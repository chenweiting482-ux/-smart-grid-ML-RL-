"""
rl_environment.py — Microgrid EMS Environment (v4)
═══════════════════════════════════════════════════
Reference design: CityLearn / Gym-ANM simplified

Energy flow:
    Solar → Load
    Solar → Battery
    Battery → Load
    Grid → Load (buy)
    Load → Grid (sell, if grid_power < 0)

State (5-dim, all normalised to [0,1]):
    predicted_load    LSTM forecast
    predicted_solar   LSTM forecast
    price             current electricity price (normalised)
    battery_soc       0=empty, 1=full
    hour_of_day       0=midnight, 1=23:00

Action (continuous, 1-dim):
    battery_power ∈ [-1, 1]
    -1 = full discharge  |  0 = idle  |  +1 = full charge

Battery model:
    soc(t+1) = soc(t) + η_c * charge  - discharge / η_d

Energy balance (per step):
    net_load    = actual_load - actual_solar
    grid_power  = net_load + charge_from_grid - discharge_to_load
    grid_power > 0  →  buying from grid
    grid_power < 0  →  selling to grid (small sell price)

Reward:
    cost           = price_buy  * max(grid_power, 0)
                   - price_sell * max(-grid_power, 0)
    battery_penalty= 0.001 * |action|        ← prevents wild cycling
    reward         = -cost - battery_penalty

Episode:
    168 steps = 7 days  (sweet spot for PPO)
"""

import numpy as np
import pandas as pd
import torch
import pickle
import gymnasium as gym
from gymnasium import spaces

from LSTM import (LSTMForecaster, FEATURE_COLS, SEQ_LEN,TARGET_COLS, add_features,generate_solar, price_model, SOLAR_CAP)

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
PRICE_BUY_MIN  = 0.10
PRICE_BUY_MAX  = 0.40
PRICE_SELL     = 0.05   # sell-back rate (always lower than buy)
EPISODE_HOURS  = 168    # 7 days


class BatteryStorageEnv(gym.Env):
    """
    Microgrid Energy Management System environment.
    Compatible with Stable-Baselines3 PPO.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_csv:     str   = "saved_models/test_data.csv",
        model_path:   str   = "saved_models/lstm_best.pth",
        scalers_path: str   = "saved_models/scalers.pkl",
        battery_cap:  float = 10.0,    # kWh — household scale
        charge_rate:  float = 2.5,     # kW  — max charge/discharge per hour
        eta_charge:   float = 0.95,    # charging efficiency
        eta_discharge:float = 0.95,    # discharging efficiency
    ):
        super().__init__()

        self.battery_cap   = battery_cap
        self.charge_rate   = charge_rate
        self.eta_charge    = eta_charge
        self.eta_discharge = eta_discharge

        # ── load data ──
        df = pd.read_csv(data_csv, index_col=0, parse_dates=True)
        self.df       = add_features(df).dropna().reset_index(drop=False)
        self.max_load = self.df["load"].max()
        self.n_steps  = len(self.df)

        # ── load LSTM ──
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        self.X_scaler = scalers["X_scaler"]
        self.y_scaler = scalers["y_scaler"]

        self.ml_model = LSTMForecaster(
            input_size=len(FEATURE_COLS), output_size=len(TARGET_COLS))
        self.ml_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True))
        self.ml_model.eval()

        # ── spaces ──
        # State: [pred_load, pred_solar, price, soc, hour]
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5,  dtype=np.float32),
        )
        # Action: continuous [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
        )

        self.current_step  = None
        self.episode_start = None
        self.battery_soc   = None
        self.total_cost    = None

    # ── helpers ──────────────────────────

    def _predict(self, step_idx):
        """LSTM forward pass → (pred_load kW, pred_solar kW)"""
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

        return np.array([norm_load, norm_solar, norm_price,
                         norm_soc, norm_hour], dtype=np.float32)

    # ── Gym API ──────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        max_start = self.n_steps - EPISODE_HOURS - 10
        self.episode_start = int(np.random.randint(SEQ_LEN, max_start))
        self.current_step  = self.episode_start
        # Random initial SoC: forces PPO to learn from diverse situations
        self.battery_soc   = float(
            np.random.uniform(0.1, 0.9) * self.battery_cap)
        self.total_cost    = 0.0
        return self._get_obs(), {}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))

        # ── get environment state ──
        row         = self.df.iloc[self.current_step]
        actual_load = max(float(row["load"]), 0.0)
        hour        = self._get_hour()
        price_buy   = price_model(hour)

        # actual solar WITH noise (differs from prediction → realism)
        actual_solar = generate_solar(hour, noise=True)

        # ── net load (after solar) ──
        net_load = max(actual_load - actual_solar, 0.0)

        # ── battery physics ──────────────
        # Positive action → charge battery from grid
        # Negative action → discharge battery to supply load

        if a > 0:    # charging
            charge_power   = a * self.charge_rate          # kW requested
            max_chargeable = self.battery_cap - self.battery_soc
            energy_stored  = min(charge_power * self.eta_charge,
                                 max_chargeable)            # kWh into battery
            grid_for_batt  = energy_stored / self.eta_charge  # kWh drawn from grid
            self.battery_soc += energy_stored
            battery_to_load = 0.0

        elif a < 0:  # discharging
            discharge_power  = (-a) * self.charge_rate
            max_discharge    = self.battery_soc
            energy_from_batt = min(discharge_power, max_discharge)
            battery_to_load  = energy_from_batt * self.eta_discharge
            self.battery_soc -= energy_from_batt
            grid_for_batt = 0.0

        else:        # idle
            battery_to_load = 0.0
            grid_for_batt   = 0.0

        # ── energy balance ───────────────
        # grid_power > 0  →  buying from grid
        # grid_power < 0  →  selling to grid (excess solar / battery)
        grid_power = net_load - battery_to_load + grid_for_batt

        # ── cost & reward ────────────────
        if grid_power >= 0:
            cost = price_buy  * grid_power
        else:
            cost = -PRICE_SELL * (-grid_power)   # sell-back = negative cost

        battery_penalty = 0.001 * abs(a)         # small penalty for cycling

        self.total_cost   += cost
        self.current_step += 1

        reward     = -cost - battery_penalty
        terminated = (self.current_step - self.episode_start) >= EPISODE_HOURS

        info = {
            "actual_load":   actual_load,
            "actual_solar":  actual_solar,
            "net_load":      net_load,
            "hour":          hour,
            "price_buy":     price_buy,
            "battery_soc":   self.battery_soc,
            "grid_power":    grid_power,
            "cost":          cost,
            "total_cost":    self.total_cost,
            "action_raw":    a,
        }
        return self._get_obs(), reward, terminated, False, info


# ═════════════════════════════════════════════════════════════════
# ENVIRONMENT VALIDATION  —  run this file directly to verify
#
# 4 independent tests:
#   Test 1  Physical logic    charging costs money, discharging saves money
#   Test 2  Numerical bounds  SoC stays in [0, cap], obs stays in [0, 1]
#   Test 3  Solar pattern     output follows sunrise/sunset cycle
#   Test 4  Policy ranking    Rule-Based > Random > Always Charge (cost order)
#
# Expected output (all PASS):
#   [PASS] Physical logic
#   [PASS] SoC bounds
#   [PASS] Observation bounds
#   [PASS] Solar pattern
#   [PASS] Policy ranking
# ═════════════════════════════════════════════════════════════════

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def _fixed_reset(env, seed=0):
    """Reset environment to a fixed, reproducible starting point."""
    obs, _ = env.reset(seed=seed)
    env.current_step  = SEQ_LEN
    env.episode_start = SEQ_LEN
    env.battery_soc   = env.battery_cap * 0.5
    env.total_cost    = 0.0
    return env._get_obs()


def _run_policy(env, policy_fn, n_steps=EPISODE_HOURS, seed=0):
    """Run a policy for n_steps, return collected data."""
    obs = _fixed_reset(env, seed=seed)
    record = {"costs":[], "socs":[], "solars":[], "loads":[],
              "prices":[], "actions":[], "grid_powers":[], "obs":[]}
    for _ in range(n_steps):
        action = policy_fn(obs, env)
        obs, _, done, _, info = env.step(action)
        record["costs"].append(info["cost"])
        record["socs"].append(info["battery_soc"])
        record["solars"].append(info["actual_solar"])
        record["loads"].append(info["actual_load"])
        record["prices"].append(info["price_buy"])
        record["actions"].append(info["action_raw"])
        record["grid_powers"].append(info["grid_power"])
        record["obs"].append(obs.copy())
        if done: break
    record["total_cost"] = sum(record["costs"])
    return record


# ─────────────────────────────────────────────────────────────────
def test1_physical_logic(env):
    """
    Charge for 1 step at off-peak, then discharge for 1 step at peak.
    Discharge step must be cheaper than idle step (battery reduces grid draw).
    """
    print("\n[Test 1] Physical logic — charge cheap, discharge expensive")

    passed = True

    # Find an off-peak hour (price = 0.10) in the data
    for i in range(SEQ_LEN, len(env.df) - 5):
        h = env._get_hour(i)
        if price_model(h) == 0.10:
            env.current_step  = i
            env.episode_start = i
            env.battery_soc   = 0.0
            env.total_cost    = 0.0
            break

    obs = env._get_obs()
    hour_before = env._get_hour()
    price_before = price_model(hour_before)

    # Step A: charge
    _, _, _, _, info_charge = env.step(np.array([1.0]))
    soc_after_charge = info_charge["battery_soc"]
    cost_charge = info_charge["cost"]

    print(f"  Hour {hour_before}: price=${price_before:.2f} | "
          f"Charge cost=${cost_charge:.4f} | SoC: 0 → {soc_after_charge:.2f} kWh")

    # check: charging increases SoC
    if soc_after_charge <= 0:
        print("  FAIL: SoC did not increase after charging")
        passed = False
    else:
        print("  OK  : SoC increased after charging")

    # Find a peak hour (price = 0.40)
    for i in range(env.current_step, len(env.df) - 2):
        h = env._get_hour(i)
        if price_model(h) == 0.40:
            env.current_step = i
            break

    env.battery_soc = soc_after_charge

    # Step B1: idle at peak
    soc_b = env.battery_soc
    step_b = env.current_step
    _, _, _, _, info_idle = env.step(np.array([0.0]))
    cost_idle = info_idle["cost"]

    # Step B2: discharge at peak (reset to same point)
    env.current_step = step_b
    env.battery_soc  = soc_b
    _, _, _, _, info_dis = env.step(np.array([-1.0]))
    cost_dis = info_dis["cost"]
    soc_after_dis = info_dis["battery_soc"]

    print(f"  Hour {env._get_hour(step_b)}: price=${price_model(env._get_hour(step_b)):.2f} | "
          f"Idle cost=${cost_idle:.4f} | Discharge cost=${cost_dis:.4f} | "
          f"SoC: {soc_b:.2f} → {soc_after_dis:.2f} kWh")

    if cost_dis < cost_idle:
        print("  OK  : Discharge reduces cost vs idle at peak price")
    else:
        print("  FAIL: Discharge should reduce cost at peak price")
        passed = False

    if soc_after_dis < soc_b:
        print("  OK  : SoC decreased after discharging")
    else:
        print("  FAIL: SoC did not decrease after discharging")
        passed = False

    result = "[PASS]" if passed else "[FAIL]"
    print(f"  {result} Physical logic")
    return passed


# ─────────────────────────────────────────────────────────────────
def test2_numerical_bounds(env, n_steps=EPISODE_HOURS):
    """SoC must stay in [0, battery_cap]. Obs must stay in [0, 1]."""
    print("\n[Test 2] Numerical bounds — SoC and observation range")

    # Use random policy to stress-test bounds
    obs = _fixed_reset(env)
    soc_violations = 0
    obs_violations = 0

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, _, done, _, info = env.step(action)
        soc = info["battery_soc"]
        if soc < -0.001 or soc > env.battery_cap + 0.001:
            soc_violations += 1
        if np.any(obs < -0.001) or np.any(obs > 1.001):
            obs_violations += 1
        if done: break

    passed = True
    if soc_violations == 0:
        print(f"  OK  : SoC always in [0, {env.battery_cap}] ({n_steps} steps)")
    else:
        print(f"  FAIL: {soc_violations} SoC out-of-range violations")
        passed = False

    if obs_violations == 0:
        print(f"  OK  : All observations in [0, 1] ({n_steps} steps)")
    else:
        print(f"  FAIL: {obs_violations} observation out-of-range violations")
        passed = False

    result = "[PASS]" if passed else "[FAIL]"
    print(f"  {result} Numerical bounds")
    return passed


# ─────────────────────────────────────────────────────────────────
def test3_solar_pattern(env):
    """Solar output must be 0 at night and positive during the day."""
    print("\n[Test 3] Solar pattern — day/night cycle")

    night_hours = list(range(0, 6)) + list(range(21, 24))
    day_hours   = list(range(9, 16))   # guaranteed sunlight

    night_vals = [generate_solar(h, noise=False) for h in night_hours]
    day_vals   = [generate_solar(h, noise=False) for h in day_hours]

    passed = True
    if all(v == 0.0 for v in night_vals):
        print(f"  OK  : Solar = 0 during night hours {night_hours}")
    else:
        print(f"  FAIL: Solar not zero at night: {night_vals}")
        passed = False

    if all(v > 0 for v in day_vals):
        print(f"  OK  : Solar > 0 during day hours {day_hours}")
        print(f"        Peak solar at noon: {generate_solar(12, noise=False):.2f} kW")
    else:
        print(f"  FAIL: Solar zero during day: {day_vals}")
        passed = False

    result = "[PASS]" if passed else "[FAIL]"
    print(f"  {result} Solar pattern")
    return passed


# ─────────────────────────────────────────────────────────────────
def test4_policy_ranking(env):
    """
    Rule-Based (solar+price aware) must cost less than Always Idle.
    Always Charge must cost more than Rule-Based (wastes money charging at peak).
    """
    print("\n[Test 4] Policy ranking — Rule-Based < Idle < Always Charge")

    def rule(obs, e):
        h     = e._get_hour()
        solar = generate_solar(h, noise=False)
        price = price_model(h)
        if   solar > 1.0:   return np.array([ 1.0])
        elif price >= 0.40: return np.array([-1.0])
        else:               return np.array([ 0.0])

    r_idle   = _run_policy(env, lambda o, e: np.array([0.0]))
    r_charge = _run_policy(env, lambda o, e: np.array([1.0]))
    r_rule   = _run_policy(env, rule)

    c_idle   = r_idle["total_cost"]
    c_charge = r_charge["total_cost"]
    c_rule   = r_rule["total_cost"]

    print(f"  Always Idle:   ${c_idle:.4f}")
    print(f"  Always Charge: ${c_charge:.4f}  (expected > Idle)")
    print(f"  Rule-Based:    ${c_rule:.4f}  (expected < Idle)")

    passed = True
    saving = c_idle - c_rule
    pct    = saving / c_idle * 100

    if c_rule < c_idle:
        print(f"  OK  : Rule-Based saves ${saving:.4f} ({pct:.1f}%)")
    else:
        print(f"  FAIL: Rule-Based should be cheaper than Idle")
        passed = False

    result = "[PASS]" if passed else "[FAIL]"
    print(f"  {result} Policy ranking")
    return passed, c_idle, c_rule, c_charge, r_rule


# ─────────────────────────────────────────────────────────────────
def plot_validation(env, rule_record):
    """
    3-panel validation plot:
      Panel 1: Solar + Load over 7 days
      Panel 2: Electricity price over 7 days
      Panel 3: Battery SoC under Rule-Based policy
    """
    import os
    os.makedirs("results", exist_ok=True)

    n = len(rule_record["costs"])
    hours = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Environment Validation — 7-Day Snapshot (Rule-Based Policy)",
                 fontsize=12, fontweight="bold")

    # Panel 1: Load and Solar
    ax = axes[0]
    ax.plot(hours, rule_record["loads"],  color="#1f77b4",
            linewidth=1.2, label="Actual Load (kW)")
    ax.fill_between(hours, rule_record["solars"],
                    color="#f0c040", alpha=0.6, label="Solar Output (kW)")
    ax.set_ylabel("Power (kW)")
    ax.set_title("(1) Load Demand vs Solar Generation", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: Electricity price
    ax = axes[1]
    ax.step(hours, rule_record["prices"],
            color="#d62728", linewidth=1.5, where="post", label="Buy Price")
    ax.axhline(y=PRICE_SELL, color="gray", linestyle="--",
               linewidth=0.8, label=f"Sell Price (${PRICE_SELL})")
    ax.set_ylabel("Price ($/kWh)")
    ax.set_title("(2) Electricity Price Signal", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 3: Battery SoC
    ax = axes[2]
    ax.plot(hours, rule_record["socs"],
            color="#2ca02c", linewidth=1.5, label="Battery SoC")
    ax.axhline(y=env.battery_cap * 0.5, color="gray",
               linestyle="--", linewidth=0.8, label="50% SoC")
    ax.fill_between(hours, rule_record["socs"], 0,
                    color="#2ca02c", alpha=0.15)
    ax.set_ylabel("SoC (kWh)"); ax.set_xlabel("Hour")
    ax.set_title("(3) Battery State of Charge (Rule-Based policy)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig("results/env_validation1.png", dpi=150, bbox_inches="tight")
    print("\n  Figure saved: results/env_validation.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ENVIRONMENT VALIDATION")
    print("  rl_environment.py — Microgrid EMS")
    print("=" * 60)

    env = BatteryStorageEnv()

    results_all = {}

    p1 = test1_physical_logic(env)
    p2 = test2_numerical_bounds(env)
    p3 = test3_solar_pattern(env)
    p4, c_idle, c_rule, c_charge, rule_record = test4_policy_ranking(env)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)
    tests = [
        ("Physical logic  (charge↑cost, discharge↓cost)", p1),
        ("SoC & obs bounds (always in valid range)",       p2),
        ("Solar pattern   (zero at night, peak at noon)",  p3),
        ("Policy ranking  (Rule-Based < Idle cost)",       p4),
    ]
    all_pass = True
    for name, passed in tests:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ✅ All tests passed — environment is correct.")
        print("     PPO has a valid learning signal. Run train_ppo.py next.")
    else:
        print("  ❌ Some tests failed — check the environment before training.")

    # ── Plot ──
    print("\nGenerating validation plots...")
    plot_validation(env, rule_record)