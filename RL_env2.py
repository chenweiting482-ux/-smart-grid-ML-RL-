"""
rl_environment.py — Microgrid EMS (v5, discrete action + reward shaping)
══════════════════════════════════════════════════════════════════════════
Key changes from v4:
  1. DISCRETE action space {0=discharge, 1=idle, 2=charge}
     → PPO learns much faster on discrete actions for this problem
     → Literature confirms discrete PPO outperforms continuous on battery dispatch
  2. Reward shaping: immediate signal for correct behavior
     → Charge bonus when price is low OR solar is high
     → Discharge bonus when price is high AND battery has energy
     → SoC penalty when battery is empty at peak time (missed opportunity)
  3. State includes soc_change: shows PPO that actions affect state
  4. Episode = 24 hours (1 day) — shorter episodes for faster credit assignment

State (6-dim, all in [0,1]):
    predicted_load   LSTM output
    predicted_solar  LSTM output
    price_norm       current price normalised
    battery_soc      0=empty, 1=full
    hour_norm        0=midnight, 1=23:00
    net_load_norm    (load - solar) / max_load  ← new: direct signal

Action (discrete):
    0 = discharge (full rate)
    1 = idle
    2 = charge   (full rate)

Reward:
    base   = -electricity_cost
    shaping= +charge_bonus + discharge_bonus - missed_opportunity_penalty
"""

import numpy as np
import pandas as pd
import torch
import pickle
import gymnasium as gym
from gymnasium import spaces

from LSTM import (
    LSTMForecaster, FEATURE_COLS, SEQ_LEN,
    TARGET_COLS, add_features,
    generate_solar, price_model, SOLAR_CAP
)

PRICE_BUY_MIN  = 0.10
PRICE_BUY_MAX  = 0.40
PRICE_SELL     = 0.05
EPISODE_HOURS  = 24    # 1 day — shorter = faster credit assignment for PPO


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

        # load data
        df = pd.read_csv(data_csv, index_col=0, parse_dates=True)
        self.df       = add_features(df).dropna().reset_index(drop=False)
        self.max_load = self.df["load"].max()
        self.n_steps  = len(self.df)

        # load LSTM
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        self.X_scaler = scalers["X_scaler"]
        self.y_scaler = scalers["y_scaler"]

        self.ml_model = LSTMForecaster(
            input_size=len(FEATURE_COLS), output_size=len(TARGET_COLS))
        self.ml_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True))
        self.ml_model.eval()

        # ── DISCRETE action space ──
        # 0=discharge  1=idle  2=charge
        self.action_space = spaces.Discrete(3)

        # ── 6-dim state ──
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6,  dtype=np.float32),
        )

        self.current_step  = None
        self.episode_start = None
        self.battery_soc   = None
        self.total_cost    = None

    # ── helpers ──────────────────────────

    def _predict(self, step_idx):
        start  = max(0, step_idx - SEQ_LEN)
        window = self.df.iloc[start:step_idx][FEATURE_COLS].values
        if len(window) < SEQ_LEN:
            pad    = np.zeros((SEQ_LEN - len(window), len(FEATURE_COLS)))
            window = np.vstack([pad, window])
        w_s = self.X_scaler.transform(window)
        x   = torch.tensor(w_s[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            out_s = self.ml_model(x).numpy()[0]
        out        = self.y_scaler.inverse_transform(out_s.reshape(1, -1))[0]
        pred_load  = max(float(out[0]), 0.0)
        pred_solar = float(np.clip(out[1], 0.0, SOLAR_CAP))
        return pred_load, pred_solar

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

        norm_load  = float(np.clip(pred_load  / self.max_load,       0.0, 1.0))
        norm_solar = float(np.clip(pred_solar / SOLAR_CAP,           0.0, 1.0))
        norm_price = float((price - PRICE_BUY_MIN) /
                           (PRICE_BUY_MAX - PRICE_BUY_MIN))
        norm_soc   = float(self.battery_soc / self.battery_cap)
        norm_hour  = float(hour / 23.0)

        # net load: direct signal showing surplus/deficit
        net_load   = max(pred_load - pred_solar, 0.0)
        norm_net   = float(np.clip(net_load / self.max_load, 0.0, 1.0))

        return np.array([norm_load, norm_solar, norm_price,
                         norm_soc, norm_hour, norm_net], dtype=np.float32)

    # ── Gym API ──────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = self.n_steps - EPISODE_HOURS * 2 - 10
        self.episode_start = int(np.random.randint(SEQ_LEN, max_start))
        self.current_step  = self.episode_start
        self.battery_soc   = float(
            np.random.uniform(0.1, 0.9) * self.battery_cap)
        self.total_cost    = 0.0
        return self._get_obs(), {}

    def _get_valid_actions(self):
        """
        Returns which actions are physically possible given current SoC.
        Used to override impossible actions before they enter physics.
        """
        can_charge    = self.battery_soc < self.battery_cap - 0.01
        can_discharge = self.battery_soc > 0.01
        valid = [1]   # idle always valid
        if can_charge:    valid.append(2)
        if can_discharge: valid.append(0)
        return valid

    def step(self, action: int):
        row         = self.df.iloc[self.current_step]
        actual_load = max(float(row["load"]), 0.0)
        hour        = self._get_hour()
        price_buy   = price_model(hour)
        actual_solar = generate_solar(hour, noise=True)
        net_load     = max(actual_load - actual_solar, 0.0)

        # ── enforce physical constraints ─
        # If action is physically impossible, override to idle
        # and apply a penalty so PPO learns not to attempt it
        impossible_action = False
        if action == 2 and self.battery_soc >= self.battery_cap - 0.01:
            action = 1   # battery full → can't charge → force idle
            impossible_action = True
        elif action == 0 and self.battery_soc <= 0.01:
            action = 1   # battery empty → can't discharge → force idle
            impossible_action = True

        # ── battery physics ──────────────
        if action == 2:    # charge
            max_chargeable = self.battery_cap - self.battery_soc
            energy_stored  = min(self.charge_rate * self.eta_charge,
                                 max_chargeable)
            grid_for_batt  = energy_stored / self.eta_charge
            self.battery_soc += energy_stored
            battery_to_load  = 0.0

        elif action == 0:  # discharge
            max_discharge    = self.battery_soc
            energy_from_batt = min(self.charge_rate, max_discharge)
            battery_to_load  = energy_from_batt * self.eta_discharge
            self.battery_soc -= energy_from_batt
            grid_for_batt    = 0.0

        else:              # idle
            battery_to_load = 0.0
            grid_for_batt   = 0.0

        # clamp SoC to valid range (floating point safety)
        self.battery_soc = float(np.clip(self.battery_soc, 0.0, self.battery_cap))

        # ── energy balance ───────────────
        grid_power = net_load - battery_to_load + grid_for_batt

        if grid_power >= 0:
            elec_cost = price_buy * grid_power
        else:
            elec_cost = -PRICE_SELL * (-grid_power)

        # ── reward shaping ───────────────
        reward = -elec_cost

        # Penalty for attempting an impossible action
        # This directly teaches PPO the battery boundaries
        if impossible_action:
            reward -= 0.5

        # Shaping 1: charge at off-peak or high solar
        if action == 2:
            if price_buy <= 0.10:
                reward += 0.5
            elif actual_solar > 1.0:
                reward += 0.4

        # Shaping 2: discharge at peak price (only if actually discharged)
        if action == 0 and price_buy >= 0.40 and battery_to_load > 0:
            reward += 0.5

        # Shaping 3: penalise being empty at peak (missed opportunity)
        if self.battery_soc < 0.5 and price_buy >= 0.40:
            reward -= 0.3

        # Shaping 4: penalise charging at peak price
        if action == 2 and price_buy >= 0.40:
            reward -= 0.5

        self.total_cost   += elec_cost
        self.current_step += 1
        terminated = (self.current_step - self.episode_start) >= EPISODE_HOURS

        info = {
            "actual_load":  actual_load,
            "actual_solar": actual_solar,
            "net_load":     net_load,
            "hour":         hour,
            "price_buy":    price_buy,
            "battery_soc":  self.battery_soc,
            "grid_power":   grid_power,
            "cost":         elec_cost,
            "total_cost":   self.total_cost,
            "action_raw":   float(action - 1),  # map to [-1,0,1] for compatibility
        }
        return self._get_obs(), reward, terminated, False, info


# ════════════════════════════════════════════════════════════════
# ENVIRONMENT VALIDATION
# ════════════════════════════════════════════════════════════════

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def _fixed_reset(env, seed=0):
    obs, _ = env.reset(seed=seed)
    env.current_step  = SEQ_LEN
    env.episode_start = SEQ_LEN
    env.battery_soc   = env.battery_cap * 0.5
    env.total_cost    = 0.0
    return env._get_obs()


def _run_policy(env, policy_fn, n_steps=EPISODE_HOURS, seed=0):
    obs = _fixed_reset(env, seed=seed)
    record = {"costs":[], "socs":[], "solars":[], "loads":[],
              "prices":[], "actions":[], "obs":[]}
    for _ in range(n_steps):
        action = policy_fn(obs, env)
        obs, _, done, _, info = env.step(action)
        record["costs"].append(info["cost"])
        record["socs"].append(info["battery_soc"])
        record["solars"].append(info["actual_solar"])
        record["loads"].append(info["actual_load"])
        record["prices"].append(info["price_buy"])
        record["actions"].append(int(action))
        record["obs"].append(obs.copy())
        if done: break
    record["total_cost"] = sum(record["costs"])
    return record


def test1_physical_logic(env):
    print("\n[Test 1] Physical logic — charge cheap, discharge expensive")
    passed = True

    # find off-peak hour
    for i in range(SEQ_LEN, len(env.df) - 5):
        if price_model(env._get_hour(i)) == 0.10:
            env.current_step  = i
            env.episode_start = i
            env.battery_soc   = 0.0
            env.total_cost    = 0.0
            break

    hour_c = env._get_hour()
    _, _, _, _, info_c = env.step(2)   # charge
    print(f"  Hour {hour_c:02d}: price=${price_model(hour_c):.2f} | "
          f"Charge cost=${info_c['cost']:.4f} | SoC: 0→{info_c['battery_soc']:.2f} kWh")

    if info_c['battery_soc'] > 0:
        print("  OK  : SoC increased after charging")
    else:
        print("  FAIL: SoC did not increase"); passed = False

    # find peak hour
    soc_saved = info_c['battery_soc']
    step_saved = env.current_step
    for i in range(env.current_step, len(env.df) - 2):
        if price_model(env._get_hour(i)) == 0.40:
            env.current_step = i; break

    # idle vs discharge
    env.battery_soc = soc_saved
    step_pk = env.current_step
    _, _, _, _, i_idle = env.step(1)

    env.current_step = step_pk
    env.battery_soc  = soc_saved
    _, _, _, _, i_dis = env.step(0)

    print(f"  Hour {env._get_hour(step_pk):02d}: price=${price_model(env._get_hour(step_pk)):.2f} | "
          f"Idle=${i_idle['cost']:.4f} | Discharge=${i_dis['cost']:.4f}")

    if i_dis['cost'] < i_idle['cost']:
        print("  OK  : Discharge reduces cost at peak")
    else:
        print("  FAIL: Discharge should be cheaper at peak"); passed = False

    print(f"  {'[PASS]' if passed else '[FAIL]'} Physical logic")
    return passed


def test2_numerical_bounds(env, n_steps=168):
    print("\n[Test 2] Numerical bounds")
    obs = _fixed_reset(env)
    soc_v = obs_v = 0
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, _, done, _, info = env.step(action)
        if info['battery_soc'] < -0.001 or info['battery_soc'] > env.battery_cap + 0.001:
            soc_v += 1
        if np.any(obs < -0.001) or np.any(obs > 1.001):
            obs_v += 1
        if done: break

    passed = soc_v == 0 and obs_v == 0
    print(f"  {'OK  ' if soc_v==0 else 'FAIL'}: SoC violations={soc_v}")
    print(f"  {'OK  ' if obs_v==0 else 'FAIL'}: Obs violations={obs_v}")
    print(f"  {'[PASS]' if passed else '[FAIL]'} Numerical bounds")
    return passed


def test3_solar_pattern(env):
    print("\n[Test 3] Solar pattern")
    night = [generate_solar(h, False) for h in range(0,6)]
    day   = [generate_solar(h, False) for h in range(9,16)]
    p1 = all(v == 0 for v in night)
    p2 = all(v > 0  for v in day)
    print(f"  {'OK  ' if p1 else 'FAIL'}: Night hours zero")
    print(f"  {'OK  ' if p2 else 'FAIL'}: Day hours positive (noon={generate_solar(12,False):.2f} kW)")
    passed = p1 and p2
    print(f"  {'[PASS]' if passed else '[FAIL]'} Solar pattern")
    return passed


def test4_policy_ranking(env):
    print("\n[Test 4] Policy ranking — Rule-Based should cost < Idle")

    def rule(obs, e):
        h = e._get_hour(); s = generate_solar(h, False); p = price_model(h)
        if   p >= 0.40: return 0   # peak price → discharge first
        elif p <= 0.10: return 2   # off-peak   → charge
        elif s > 1.0:   return 2   # solar surplus → charge
        else:           return 1   # shoulder   → idle

    # run over 168 steps (7 days) for more stable comparison
    r_idle  = _run_policy(env, lambda o,e: 1, n_steps=168)
    r_charge= _run_policy(env, lambda o,e: 2, n_steps=168)
    r_rule  = _run_policy(env, rule,           n_steps=168)

    ci, cc, cr = r_idle["total_cost"], r_charge["total_cost"], r_rule["total_cost"]
    print(f"  Always Idle:   ${ci:.4f}")
    print(f"  Always Charge: ${cc:.4f}")
    print(f"  Rule-Based:    ${cr:.4f}")

    passed = cr < ci
    saving = ci - cr
    print(f"  {'OK  ' if passed else 'FAIL'}: Rule saves ${saving:.4f} ({saving/ci*100:.1f}%)")
    print(f"  {'[PASS]' if passed else '[FAIL]'} Policy ranking")
    return passed, ci, cr, cc, r_rule


def test5_reward_shaping(env):
    """Verify reward shaping gives correct signal: charging off-peak gets bonus."""
    print("\n[Test 5] Reward shaping — correct bonus/penalty signals")
    passed = True

    # find off-peak step
    for i in range(SEQ_LEN, SEQ_LEN + 48):
        if price_model(env._get_hour(i)) == 0.10:
            env.current_step  = i
            env.episode_start = i
            env.battery_soc   = 0.0
            env.total_cost    = 0.0
            break

    obs = env._get_obs()
    _, r_charge, _, _, _ = env.step(2)

    env.current_step  = i
    env.episode_start = i
    env.battery_soc   = 0.0
    env.total_cost    = 0.0
    _, r_idle, _, _, _ = env.step(1)

    print(f"  Off-peak charge reward: {r_charge:.4f}")
    print(f"  Off-peak idle   reward: {r_idle:.4f}")
    if r_charge > r_idle:
        print("  OK  : Charging off-peak rewarded more than idle")
    else:
        print("  FAIL: Charging should be rewarded more than idle off-peak")
        passed = False

    for i in range(SEQ_LEN, SEQ_LEN + 48):
        if price_model(env._get_hour(i)) == 0.40:
            env.current_step  = i
            env.episode_start = i
            env.battery_soc   = env.battery_cap * 0.8
            env.total_cost    = 0.0
            break

    _, r_dis,  _, _, _ = env.step(0)

    env.current_step  = i
    env.episode_start = i
    env.battery_soc   = env.battery_cap * 0.8
    env.total_cost    = 0.0
    _, r_idle2, _, _, _ = env.step(1)

    print(f"  Peak discharge reward: {r_dis:.4f}")
    print(f"  Peak idle      reward: {r_idle2:.4f}")
    if r_dis > r_idle2:
        print("  OK  : Discharging at peak rewarded more than idle")
    else:
        print("  FAIL: Discharging at peak should reward more than idle")
        passed = False

    print(f"  {'[PASS]' if passed else '[FAIL]'} Reward shaping")
    return passed


def test6_boundary_enforcement(env):
    """
    When battery is FULL, charge action must be overridden to idle.
    When battery is EMPTY, discharge action must be overridden to idle.
    SoC must never exceed battery_cap or go below 0.
    """
    print("\n[Test 6] Boundary enforcement — no charge when full, no discharge when empty")
    passed = True

    # --- Test: charge when full ---
    env.reset(seed=0)
    env.current_step  = SEQ_LEN
    env.episode_start = SEQ_LEN
    env.battery_soc   = env.battery_cap   # FULL
    env.total_cost    = 0.0

    soc_before = env.battery_soc
    obs, r, _, _, info = env.step(2)   # attempt charge
    soc_after = info["battery_soc"]

    print(f"  Full battery + charge action: SoC {soc_before:.2f} → {soc_after:.2f}")
    if abs(soc_after - soc_before) < 0.01:
        print("  OK  : SoC did not increase beyond capacity")
    else:
        print("  FAIL: SoC exceeded battery capacity")
        passed = False

    if r < 0:   # should have received penalty
        print("  OK  : Penalty applied for impossible charge action")
    else:
        print("  FAIL: No penalty for impossible charge action")
        passed = False

    # --- Test: discharge when empty ---
    env.reset(seed=0)
    env.current_step  = SEQ_LEN
    env.episode_start = SEQ_LEN
    env.battery_soc   = 0.0   # EMPTY
    env.total_cost    = 0.0

    soc_before = env.battery_soc
    obs, r2, _, _, info2 = env.step(0)   # attempt discharge
    soc_after2 = info2["battery_soc"]

    print(f"  Empty battery + discharge action: SoC {soc_before:.2f} → {soc_after2:.2f}")
    if abs(soc_after2 - soc_before) < 0.01:
        print("  OK  : SoC did not go below 0")
    else:
        print("  FAIL: SoC went below 0")
        passed = False

    if r2 < 0:
        print("  OK  : Penalty applied for impossible discharge action")
    else:
        print("  FAIL: No penalty for impossible discharge action")
        passed = False

    print(f"  {'[PASS]' if passed else '[FAIL]'} Boundary enforcement")
    return passed


def plot_validation(env, rule_record, n=168):
    import os; os.makedirs("results", exist_ok=True)
    hours = np.arange(len(rule_record["costs"]))

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Environment Validation — 7-Day Snapshot (Rule-Based Policy)",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(hours, rule_record["loads"],  color="#1f77b4", lw=1.2, label="Load (kW)")
    ax.fill_between(hours, rule_record["solars"], color="#f0c040", alpha=0.6, label="Solar (kW)")
    ax.set_ylabel("Power (kW)"); ax.set_title("(1) Load vs Solar", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.step(hours, rule_record["prices"], color="#d62728", lw=1.5, where="post")
    ax.set_ylabel("Price ($/kWh)"); ax.set_title("(2) Electricity Price", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(hours, rule_record["socs"], color="#2ca02c", lw=1.5)
    ax.fill_between(hours, rule_record["socs"], 0, color="#2ca02c", alpha=0.15)
    ax.axhline(y=env.battery_cap*0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("SoC (kWh)"); ax.set_title("(3) Battery SoC (Rule-Based)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[3]
    actions = np.array(rule_record["actions"])
    colors_a = {0:"#1f77b4", 1:"#aaaaaa", 2:"#e07b39"}
    labels_a = {0:"Discharge", 1:"Idle", 2:"Charge"}
    for a_val in [0, 1, 2]:
        mask = np.where(actions == a_val)[0]
        ax.scatter(mask, [a_val]*len(mask), s=8,
                   color=colors_a[a_val], label=labels_a[a_val], alpha=0.8)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["Discharge","Idle","Charge"])
    ax.set_xlabel("Hour"); ax.set_title("(4) Rule-Based Actions", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig("results/env_validation.png", dpi=150, bbox_inches="tight")
    print("\n  Figure saved: results/env_validation.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("  ENVIRONMENT VALIDATION v5 (Discrete Actions)")
    print("=" * 60)

    env = BatteryStorageEnv()
    p1           = test1_physical_logic(env)
    p2           = test2_numerical_bounds(env)
    p3           = test3_solar_pattern(env)
    p4, ci, cr, cc, rule_rec = test4_policy_ranking(env)
    p5           = test5_reward_shaping(env)
    p6           = test6_boundary_enforcement(env)

    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    tests = [
        ("Physical logic",        p1),
        ("Numerical bounds",      p2),
        ("Solar pattern",         p3),
        ("Policy ranking",        p4),
        ("Reward shaping",        p5),
        ("Boundary enforcement",  p6),
    ]
    all_pass = all(p for _, p in tests)
    for name, p in tests:
        print(f"  {'[PASS]' if p else '[FAIL]'}  {name}")

    print()
    if all_pass:
        print("  ✅ All tests passed.")
        print("     Run train_ppo.py next.")
    else:
        print("  ❌ Some tests failed — fix before training.")

    print("\nGenerating validation plot...")
    plot_validation(env, rule_rec)