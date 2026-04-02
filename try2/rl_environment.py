"""
rl_environment.py  v6
══════════════════════
Microgrid EMS — all known bugs fixed:

Fix 1: Price model aligned with solar
  Night 00-05 = $0.10 (charge)  ←  no solar, cheap
  Day   06-17 = $0.20 (idle)    ←  solar covers load
  Eve   18-23 = $0.40 (discharge)← no solar, expensive

Fix 2: Rule-Based uses PRICE as primary signal (not solar)
  price <= 0.10 → charge
  price >= 0.40 → discharge
  else          → idle (let solar cover load)

Fix 3: Physical boundary enforcement
  SoC >= cap  + charge action → forced idle + penalty
  SoC <= 0    + discharge     → forced idle + penalty

Fix 4: info records EXECUTED action (after override), not raw PPO output

State (6-dim, all normalised [0,1]):
  predicted_load    LSTM
  predicted_solar   LSTM
  price_norm        normalised price
  battery_soc       0=empty 1=full
  hour_norm         0=midnight 1=23:00
  net_load_norm     (load-solar)/max_load

Action (discrete): 0=discharge  1=idle  2=charge
Episode: 24 hours (1 day)
"""

import numpy as np
import pandas as pd
import torch, pickle
import gymnasium as gym
from gymnasium import spaces

from ML_LSTM import (LSTMForecaster, FEATURE_COLS, SEQ_LEN, TARGET_COLS,
                     add_features, generate_solar, price_model,
                     SOLAR_CAP, PRICE_MIN, PRICE_MAX, PRICE_SELL)

EPISODE_HOURS = 24    # 1-day episode: PPO sees full price cycle in one episode


class BatteryStorageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_csv:     str   = "saved_models/test_data.csv",
        model_path:   str   = "saved_models/lstm_best.pth",
        scalers_path: str   = "saved_models/scalers.pkl",
        battery_cap:  float = 10.0,
        charge_rate:  float = 2.5,
        eta_charge:   float = 0.95,
        eta_discharge:float = 0.95,
    ):
        super().__init__()
        self.battery_cap   = battery_cap
        self.charge_rate   = charge_rate
        self.eta_charge    = eta_charge
        self.eta_discharge = eta_discharge

        df = pd.read_csv(data_csv, index_col=0, parse_dates=True)
        self.df       = add_features(df).dropna().reset_index(drop=False)
        self.max_load = self.df["load"].max()
        self.n_steps  = len(self.df)

        with open(scalers_path, "rb") as f:
            sc = pickle.load(f)
        self.X_scaler = sc["X_scaler"]
        self.y_scaler = sc["y_scaler"]

        self.ml_model = LSTMForecaster(input_size=len(FEATURE_COLS),
                                       output_size=len(TARGET_COLS))
        self.ml_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True))
        self.ml_model.eval()

        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6,  dtype=np.float32))
        self.action_space = spaces.Discrete(3)   # 0=discharge 1=idle 2=charge

        self.current_step = self.episode_start = None
        self.battery_soc  = self.total_cost    = None

    # ── helpers ──────────────────────────────────────────────────

    def _predict(self, idx):
        start  = max(0, idx - SEQ_LEN)
        window = self.df.iloc[start:idx][FEATURE_COLS].values
        if len(window) < SEQ_LEN:
            window = np.vstack([np.zeros((SEQ_LEN-len(window), len(FEATURE_COLS))), window])
        x   = torch.tensor(self.X_scaler.transform(window)[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            out = self.y_scaler.inverse_transform(self.ml_model(x).numpy())[0]
        return max(float(out[0]), 0.0), float(np.clip(out[1], 0.0, SOLAR_CAP))

    def _get_hour(self, idx=None):
        i = idx if idx is not None else self.current_step
        row = self.df.iloc[i]
        if "datetime" in self.df.columns:
            return int(pd.to_datetime(row["datetime"]).hour)
        return i % 24

    def _get_obs(self):
        pred_load, pred_solar = self._predict(self.current_step)
        hour  = self._get_hour()
        price = price_model(hour)
        net_load = max(pred_load - pred_solar, 0.0)
        return np.array([
            np.clip(pred_load  / self.max_load,                      0, 1),
            np.clip(pred_solar / SOLAR_CAP,                          0, 1),
            (price - PRICE_MIN) / (PRICE_MAX - PRICE_MIN),
            self.battery_soc / self.battery_cap,
            hour / 23.0,
            np.clip(net_load / self.max_load,                        0, 1),
        ], dtype=np.float32)

    # ── Gym API ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = self.n_steps - EPISODE_HOURS * 2 - 10
        self.episode_start = int(np.random.randint(SEQ_LEN, max_start))
        self.current_step  = self.episode_start
        self.battery_soc   = float(np.random.uniform(0.1, 0.9) * self.battery_cap)
        self.total_cost    = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        row          = self.df.iloc[self.current_step]
        actual_load  = max(float(row["load"]), 0.0)
        hour         = self._get_hour()
        price_buy    = price_model(hour)
        actual_solar = generate_solar(hour, noise=True)
        net_load     = max(actual_load - actual_solar, 0.0)

        # ── Fix 3: enforce physical boundaries ───────────────────
        executed_action = action
        impossible      = False

        if action == 2 and self.battery_soc >= self.battery_cap - 0.01:
            executed_action = 1   # full → idle
            impossible      = True
        elif action == 0 and self.battery_soc <= 0.01:
            executed_action = 1   # empty → idle
            impossible      = True

        # ── battery physics (using executed_action) ───────────────
        if executed_action == 2:       # charge
            max_in        = self.battery_cap - self.battery_soc
            energy_stored = min(self.charge_rate * self.eta_charge, max_in)
            grid_for_batt = energy_stored / self.eta_charge
            self.battery_soc  += energy_stored
            battery_to_load    = 0.0

        elif executed_action == 0:     # discharge
            energy_from  = min(self.charge_rate, self.battery_soc)
            battery_to_load   = energy_from * self.eta_discharge
            self.battery_soc -= energy_from
            grid_for_batt      = 0.0

        else:                          # idle
            battery_to_load = grid_for_batt = 0.0

        # floating-point clamp
        self.battery_soc = float(np.clip(self.battery_soc, 0.0, self.battery_cap))

        # ── energy balance ────────────────────────────────────────
        grid_power = net_load - battery_to_load + grid_for_batt

        elec_cost = (price_buy * grid_power if grid_power >= 0
                     else -PRICE_SELL * (-grid_power))

        # ── reward shaping ────────────────────────────────────────
        reward = -elec_cost

        if impossible:
            reward -= 0.5   # teach PPO battery limits

        # +bonus for correct actions
        if executed_action == 2:
            if price_buy <= 0.10:      reward += 0.5   # charge at night ✓
            elif actual_solar > 1.0:   reward += 0.3   # charge with solar ✓

        if executed_action == 0 and price_buy >= 0.40 and battery_to_load > 0:
            reward += 0.5              # discharge at peak ✓

        # -penalty for suboptimal states
        if self.battery_soc < 0.5 and price_buy >= 0.40:
            reward -= 0.3              # empty at peak = missed opportunity

        if executed_action == 2 and price_buy >= 0.40:
            reward -= 0.5              # charging at peak = wasteful

        self.total_cost   += elec_cost
        self.current_step += 1
        terminated = (self.current_step - self.episode_start) >= EPISODE_HOURS

        info = {
            "actual_load":     actual_load,
            "actual_solar":    actual_solar,
            "net_load":        net_load,
            "hour":            hour,
            "price_buy":       price_buy,
            "battery_soc":     self.battery_soc,
            "grid_power":      grid_power,
            "cost":            elec_cost,
            "total_cost":      self.total_cost,
            # Fix 4: record EXECUTED action for correct visualisation
            "action_requested": action,
            "action_executed":  executed_action,
            "action_raw":       executed_action - 1,  # maps to {-1,0,1} for plots
            "was_overridden":   impossible,
        }
        return self._get_obs(), reward, terminated, False, info


# ══════════════════════════════════════════════════════════════════
# ENVIRONMENT VALIDATION  (6 tests)
# ══════════════════════════════════════════════════════════════════

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import os


def _fixed_reset(env, seed=0):
    obs, _ = env.reset(seed=seed)
    # start at midnight (hour=0) for clean visualisation
    for i in range(SEQ_LEN, min(SEQ_LEN + 48, env.n_steps)):
        if env._get_hour(i) == 0:
            env.current_step = env.episode_start = i
            break
    env.battery_soc = env.battery_cap * 0.5
    env.total_cost  = 0.0
    return env._get_obs()


def _run_policy(env, policy_fn, n_steps=168, seed=0):
    obs = _fixed_reset(env, seed)
    rec = {k: [] for k in ["costs","socs","solars","loads",
                             "prices","actions_exec","actions_req"]}
    for _ in range(n_steps):
        action = policy_fn(obs, env)
        obs, _, done, _, info = env.step(action)
        rec["costs"].append(info["cost"])
        rec["socs"].append(info["battery_soc"])
        rec["solars"].append(info["actual_solar"])
        rec["loads"].append(info["actual_load"])
        rec["prices"].append(info["price_buy"])
        rec["actions_exec"].append(info["action_executed"])
        rec["actions_req"].append(info["action_requested"])
        if done: break
    rec["total_cost"] = sum(rec["costs"])
    return rec


def _rule(obs, env):
    """Correct rule: price is PRIMARY signal, solar is secondary."""
    p = price_model(env._get_hour())
    s = generate_solar(env._get_hour(), noise=False)
    if   p <= 0.10: return 2   # night: cheap → charge
    elif p >= 0.40: return 0   # evening: peak → discharge
    elif s > 1.5:   return 2   # midday solar surplus → charge
    else:           return 1   # shoulder: idle


# ── test functions ──────────────────────────────────────────────

def test1_physical_logic(env):
    print("\n[Test 1] Physical logic")
    passed = True

    # find night hour (price=0.10)
    for i in range(SEQ_LEN, SEQ_LEN + 48):
        if price_model(env._get_hour(i)) == 0.10:
            env.current_step = env.episode_start = i
            env.battery_soc  = 0.0; env.total_cost = 0.0; break

    h = env._get_hour()
    _, _, _, _, ic = env.step(2)
    print(f"  Hour {h:02d} price=${price_model(h):.2f} | charge cost=${ic['cost']:.4f} | SoC→{ic['battery_soc']:.2f}")
    if ic["battery_soc"] > 0: print("  OK  : SoC increased")
    else: print("  FAIL: SoC did not increase"); passed = False

    # find evening peak (price=0.40)
    soc_save = ic["battery_soc"]
    for i in range(env.current_step, env.current_step + 24):
        if i < env.n_steps and price_model(env._get_hour(i)) == 0.40:
            env.current_step = env.episode_start = i; break

    step_pk = env.current_step
    env.battery_soc = soc_save; _, _, _, _, i_idle = env.step(1)
    env.current_step = step_pk
    env.battery_soc  = soc_save; _, _, _, _, i_dis  = env.step(0)

    h2 = env._get_hour(step_pk)
    print(f"  Hour {h2:02d} price=${price_model(h2):.2f} | idle=${i_idle['cost']:.4f} | discharge=${i_dis['cost']:.4f}")
    if i_dis["cost"] < i_idle["cost"]: print("  OK  : Discharge cheaper than idle at peak")
    else: print("  FAIL: Discharge should be cheaper at peak"); passed = False

    print(f"  {'[PASS]' if passed else '[FAIL]'} Physical logic")
    return passed


def test2_numerical_bounds(env, n=168):
    print("\n[Test 2] Numerical bounds")
    obs = _fixed_reset(env)
    sv = ov = 0
    for _ in range(n):
        a = env.action_space.sample()
        obs, _, done, _, info = env.step(a)
        if info["battery_soc"] < -0.001 or info["battery_soc"] > env.battery_cap + 0.001: sv += 1
        if np.any(obs < -0.001) or np.any(obs > 1.001): ov += 1
        if done: break
    passed = sv == 0 and ov == 0
    print(f"  {'OK  ' if sv==0 else 'FAIL'}: SoC violations={sv}")
    print(f"  {'OK  ' if ov==0 else 'FAIL'}: Obs violations={ov}")
    print(f"  {'[PASS]' if passed else '[FAIL]'} Numerical bounds")
    return passed


def test3_solar_pattern(env):
    print("\n[Test 3] Solar pattern")
    night = [generate_solar(h, False) for h in [0,1,2,3,4,5,22,23]]
    day   = [generate_solar(h, False) for h in [9,10,11,12,13]]
    p1 = all(v == 0 for v in night)
    p2 = all(v > 0  for v in day)
    print(f"  {'OK  ' if p1 else 'FAIL'}: Night solar=0")
    print(f"  {'OK  ' if p2 else 'FAIL'}: Day solar>0 (noon={generate_solar(12,False):.2f} kW)")
    passed = p1 and p2
    print(f"  {'[PASS]' if passed else '[FAIL]'} Solar pattern")
    return passed


def test4_price_alignment(env):
    """
    Verify price model is economically correct:
    night < day < evening
    AND solar peaks during DAY (not peak price time)
    """
    print("\n[Test 4] Price-solar alignment")
    passed = True

    p_night   = price_model(2)
    p_day     = price_model(12)
    p_evening = price_model(20)
    s_noon    = generate_solar(12, False)
    s_night   = generate_solar(2,  False)

    print(f"  Price: night=${p_night:.2f} | day=${p_day:.2f} | evening=${p_evening:.2f}")
    print(f"  Solar: noon={s_noon:.2f} kW | night={s_night:.2f} kW")

    if p_night < p_day < p_evening:
        print("  OK  : Price increases through the day (night < day < evening)")
    else:
        print("  FAIL: Price order wrong — should be night < day < evening")
        passed = False

    if s_noon > 0 and p_day < p_evening:
        print("  OK  : Solar peaks during DAY (cheap/medium price), not during peak price")
    else:
        print("  FAIL: Solar and peak price are misaligned")
        passed = False

    print(f"  {'[PASS]' if passed else '[FAIL]'} Price-solar alignment")
    return passed


def test5_policy_ranking(env):
    print("\n[Test 5] Policy ranking — Rule-Based < Idle")
    r_idle   = _run_policy(env, lambda o, e: 1)
    r_charge = _run_policy(env, lambda o, e: 2)
    r_rule   = _run_policy(env, _rule)
    ci, cc, cr = r_idle["total_cost"], r_charge["total_cost"], r_rule["total_cost"]
    print(f"  Always Idle:   ${ci:.4f}")
    print(f"  Always Charge: ${cc:.4f}")
    print(f"  Rule-Based:    ${cr:.4f}")
    passed = cr < ci
    saving = ci - cr
    print(f"  {'OK  ' if passed else 'FAIL'}: Rule saves ${saving:.4f} ({saving/ci*100:.1f}%)")
    print(f"  {'[PASS]' if passed else '[FAIL]'} Policy ranking")
    return passed, ci, cr, cc, r_rule


def test6_boundary_enforcement(env):
    print("\n[Test 6] Boundary enforcement")
    passed = True

    # full battery + charge
    env.reset(seed=0); env.current_step = env.episode_start = SEQ_LEN
    env.battery_soc = env.battery_cap; env.total_cost = 0.0
    soc_b = env.battery_soc
    _, r, _, _, info = env.step(2)
    print(f"  Full + charge: SoC {soc_b:.2f}→{info['battery_soc']:.2f} | override={info['was_overridden']} | reward={r:.3f}")
    if info["was_overridden"] and abs(info["battery_soc"] - soc_b) < 0.01 and r < 0:
        print("  OK  : Overridden to idle, penalty applied")
    else:
        print("  FAIL: Boundary not enforced for full battery"); passed = False

    # empty battery + discharge
    env.reset(seed=0); env.current_step = env.episode_start = SEQ_LEN
    env.battery_soc = 0.0; env.total_cost = 0.0
    _, r2, _, _, info2 = env.step(0)
    print(f"  Empty + discharge: SoC 0.00→{info2['battery_soc']:.2f} | override={info2['was_overridden']} | reward={r2:.3f}")
    if info2["was_overridden"] and info2["battery_soc"] < 0.01 and r2 < 0:
        print("  OK  : Overridden to idle, penalty applied")
    else:
        print("  FAIL: Boundary not enforced for empty battery"); passed = False

    print(f"  {'[PASS]' if passed else '[FAIL]'} Boundary enforcement")
    return passed


def test7_reward_shaping(env):
    print("\n[Test 7] Reward shaping signals")
    passed = True

    # off-peak night: charge should reward > idle
    for i in range(SEQ_LEN, SEQ_LEN + 48):
        if price_model(env._get_hour(i)) == 0.10:
            env.current_step = env.episode_start = i
            env.battery_soc = 0.0; env.total_cost = 0.0; break
    _, r_c, _, _, _ = env.step(2)
    env.current_step = env.episode_start = i
    env.battery_soc = 0.0; env.total_cost = 0.0
    _, r_i, _, _, _ = env.step(1)
    print(f"  Night charge reward: {r_c:.4f} | idle: {r_i:.4f}")
    if r_c > r_i: print("  OK  : Charge rewarded more than idle at night")
    else: print("  FAIL: Charge should beat idle at night"); passed = False

    # evening peak: discharge should reward > idle
    for i in range(SEQ_LEN, SEQ_LEN + 48):
        if price_model(env._get_hour(i)) == 0.40:
            env.current_step = env.episode_start = i
            env.battery_soc = env.battery_cap * 0.8; env.total_cost = 0.0; break
    _, r_d, _, _, _ = env.step(0)
    env.current_step = env.episode_start = i
    env.battery_soc = env.battery_cap * 0.8; env.total_cost = 0.0
    _, r_i2, _, _, _ = env.step(1)
    print(f"  Peak discharge reward: {r_d:.4f} | idle: {r_i2:.4f}")
    if r_d > r_i2: print("  OK  : Discharge rewarded more than idle at peak")
    else: print("  FAIL: Discharge should beat idle at peak"); passed = False

    print(f"  {'[PASS]' if passed else '[FAIL]'} Reward shaping")
    return passed


def plot_validation(env, rule_rec):
    os.makedirs("results", exist_ok=True)
    n = len(rule_rec["costs"]); hours = np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Environment Validation — 24h Snapshot (Rule-Based Policy, midnight start)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(hours, rule_rec["loads"],  color="#1f77b4", lw=1.2, label="Load (kW)")
    ax.fill_between(hours, rule_rec["solars"], color="#f0c040", alpha=0.6, label="Solar (kW)")
    ax.set_ylabel("Power (kW)"); ax.set_title("(1) Load vs Solar Generation", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.step(hours, rule_rec["prices"], color="#d62728", lw=1.5, where="post")
    ax.axhline(0.10, color="#2ca02c", ls=":", lw=0.8, label="Charge zone ($0.10)")
    ax.axhline(0.40, color="#d62728", ls=":", lw=0.8, label="Discharge zone ($0.40)")
    ax.set_ylabel("Price ($/kWh)"); ax.set_title("(2) Electricity Price — night cheap, evening peak", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(hours, rule_rec["socs"], color="#2ca02c", lw=1.5, label="SoC")
    ax.fill_between(hours, rule_rec["socs"], 0, color="#2ca02c", alpha=0.15)
    ax.axhline(env.battery_cap * 0.5, color="gray", ls="--", lw=0.8, label="50% SoC")
    ax.set_ylabel("SoC (kWh)"); ax.set_title("(3) Battery SoC (Rule-Based)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[3]
    # Fix 4: show EXECUTED actions, not requested
    exec_actions = np.array(rule_rec["actions_exec"])
    req_actions  = np.array(rule_rec["actions_req"])
    overridden   = exec_actions != req_actions
    cfg = {0:("Discharge","#1f77b4"), 1:("Idle","#aaaaaa"), 2:("Charge","#e07b39")}
    for av, (an, ac) in cfg.items():
        mask = np.where(exec_actions == av)[0]
        ax.scatter(mask, [av]*len(mask), s=10, color=ac, label=an, alpha=0.8, zorder=3)
    # mark overridden steps
    ov_mask = np.where(overridden)[0]
    if len(ov_mask):
        ax.scatter(ov_mask, exec_actions[ov_mask], s=60, marker="x",
                   color="red", label=f"Overridden ({len(ov_mask)})", zorder=4)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["Discharge","Idle","Charge"])
    ax.set_xlabel("Hour"); ax.set_title("(4) EXECUTED Actions (× = overridden by boundary)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig("results/env_validation.png", dpi=150, bbox_inches="tight")
    print("\n  Figure saved: results/env_validation.png")
    plt.show()


# ── entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  ENVIRONMENT VALIDATION  v6")
    print("=" * 62)

    env = BatteryStorageEnv()
    p1          = test1_physical_logic(env)
    p2          = test2_numerical_bounds(env)
    p3          = test3_solar_pattern(env)
    p4          = test4_price_alignment(env)
    p5, ci, cr, cc, rule_rec = test5_policy_ranking(env)
    p6          = test6_boundary_enforcement(env)
    p7          = test7_reward_shaping(env)

    print("\n" + "=" * 62)
    print("  VALIDATION SUMMARY")
    print("=" * 62)
    tests = [("Physical logic",        p1),
             ("Numerical bounds",      p2),
             ("Solar pattern",         p3),
             ("Price-solar alignment", p4),
             ("Policy ranking",        p5),
             ("Boundary enforcement",  p6),
             ("Reward shaping",        p7)]
    all_pass = all(p for _, p in tests)
    for name, p in tests:
        print(f"  {'[PASS]' if p else '[FAIL]'}  {name}")
    print()
    if all_pass:
        print("  ✅ All 7 tests passed — environment is correct.")
        print("     Run train_ppo.py next.")
    else:
        print("  ❌ Some tests failed — fix before training.")
    print("\nGenerating validation plot...")
    plot_validation(env, rule_rec)