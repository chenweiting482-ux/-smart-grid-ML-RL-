"""
evaluate.py — Policy Comparison for Microgrid EMS (v4)
Compares 4 strategies over 30 days (720 hours).

Policies:
  1. Always Idle      — battery never used
  2. Always Charge    — always charge at full power
  3. Rule-Based       — store solar at midday, discharge at peak price
  4. PPO (Ours)       — trained RL agent

Figures produced:
  results/policy_comparison.png   4-panel comparison
  results/summary_table.csv       cost/saving table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from stable_baselines3 import PPO
from RL_env2 import (BatteryStorageEnv, SEQ_LEN,price_model, generate_solar)


# ─────────────────────────────────────────
# Policies
# ─────────────────────────────────────────

def policy_idle(obs, env):
    return np.array([0.0])

def policy_charge(obs, env):
    return np.array([1.0])

def policy_rule(obs, env):
    """
    Logic mirroring the intended learning signal:
      Midday (high solar)  → charge battery (+1)
      Peak price (18-23)   → discharge battery (-1)
      Otherwise            → idle (0)
    """
    hour  = env._get_hour()
    price = price_model(hour)
    solar = generate_solar(hour, noise=False)

    if   solar > 1.0:   return np.array([ 1.0])   # store solar
    elif price >= 0.40: return np.array([-1.0])   # sell at peak
    else:               return np.array([ 0.0])

def policy_ppo(obs, env, model=None):
    action, _ = model.predict(obs, deterministic=True)
    return action


# ─────────────────────────────────────────
# Run one episode (fixed start for fair comparison)
# ─────────────────────────────────────────

def run_episode(policy_fn, n_steps=720, seed=42):
    env = BatteryStorageEnv()
    obs, _ = env.reset(seed=seed)
    env.current_step  = SEQ_LEN
    env.episode_start = SEQ_LEN
    env.battery_soc   = env.battery_cap * 0.5
    env.total_cost    = 0.0
    obs = env._get_obs()

    costs, loads, solars, socs, prices, actions = [], [], [], [], [], []

    for _ in range(n_steps):
        action = policy_fn(obs, env)
        obs, reward, done, _, info = env.step(action)

        costs.append(info["cost"])
        loads.append(info["actual_load"])
        solars.append(info["actual_solar"])
        socs.append(info["battery_soc"])
        prices.append(info["price_buy"])
        actions.append(info["action_raw"])

        if done:
            break

    return {
        "total_cost": sum(costs),
        "mean_cost":  np.mean(costs),
        "costs":      np.array(costs),
        "loads":      np.array(loads),
        "solars":     np.array(solars),
        "socs":       np.array(socs),
        "prices":     np.array(prices),
        "actions":    np.array(actions),
    }


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main(model_path="saved_models/best_model", n_steps=720):

    ppo_model = PPO.load(model_path)

    policies = {
        "Always Idle":   policy_idle,
        "Always Charge": policy_charge,
        "Rule-Based":    policy_rule,
        "PPO (Ours)":    lambda obs, env: policy_ppo(obs, env, model=ppo_model),
    }

    # run all policies
    results = {}
    print(f"Evaluating {n_steps} hours ({n_steps//24} days)...\n")
    for name, fn in policies.items():
        r = run_episode(fn, n_steps=n_steps, seed=42)
        results[name] = r
        print(f"  {name:<18s}  Total cost: ${r['total_cost']:.4f}")

    # savings vs baseline
    base = results["Always Idle"]["total_cost"]
    print("\nSavings vs Always Idle:")
    for name, r in results.items():
        saved = base - r["total_cost"]
        print(f"  {name:<18s}  Save ${saved:.4f}  ({saved/base*100:+.1f}%)")

    # PPO action stats
    ppo_a = results["PPO (Ours)"]["actions"]
    print(f"\nPPO action stats:")
    print(f"  Mean={ppo_a.mean():.3f}  Std={ppo_a.std():.3f}")
    print(f"  Charging   (a> 0.05): {(ppo_a >  0.05).sum()} steps")
    print(f"  Idle    (|a|<=0.05): {(np.abs(ppo_a)<=0.05).sum()} steps")
    print(f"  Discharging(a<-0.05): {(ppo_a < -0.05).sum()} steps")

    # ─────────────────────────────────────
    # 5-panel figure
    # ─────────────────────────────────────
    colors = {
        "Always Idle":   "#aaaaaa",
        "Always Charge": "#e07b39",
        "Rule-Based":    "#5b8db8",
        "PPO (Ours)":    "#2ca02c",
    }
    hours = np.arange(n_steps)

    fig, axes = plt.subplots(5, 1, figsize=(14, 17), sharex=True)
    fig.suptitle("Microgrid EMS — Policy Comparison (30 Days)",
                 fontsize=13, fontweight="bold", y=0.98)

    # 1. Cumulative cost (core result)
    ax = axes[0]
    for name, r in results.items():
        cum = np.cumsum(r["costs"])
        ax.plot(hours[:len(cum)], cum,
                label=name, color=colors[name], linewidth=2)
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title("(1) Cumulative Electricity Cost  [Core Result]", fontsize=10)
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)

    # 2. Solar generation (disturbance variable)
    ax = axes[1]
    ref_solar = results["Always Idle"]["solars"]
    ax.fill_between(hours[:len(ref_solar)], ref_solar,
                    color="#f0c040", alpha=0.6, label="Solar output")
    ax.plot(hours[:len(ref_solar)], ref_solar,
            color="#d4a000", linewidth=0.8)
    ax.set_ylabel("Solar (kW)")
    ax.set_title("(2) Solar Generation (Disturbance Variable)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. Electricity price (disturbance variable)
    ax = axes[2]
    ref_prices = results["Always Idle"]["prices"]
    ax.step(hours[:len(ref_prices)], ref_prices,
            color="#d62728", linewidth=1.2, where="post")
    ax.set_ylabel("Price ($/kWh)")
    ax.set_title("(3) Electricity Price (Disturbance Variable)", fontsize=10)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.2f"))
    ax.grid(alpha=0.3)

    # 4. Battery SoC comparison
    ax = axes[3]
    for name in ["Rule-Based", "PPO (Ours)"]:
        ax.plot(hours[:len(results[name]["socs"])], results[name]["socs"],
                label=name, color=colors[name], linewidth=1.4)
    ax.axhline(y=5.0, color="gray", linestyle="--",
               linewidth=0.8, label="50% SoC")
    ax.set_ylabel("Battery SoC (kWh)")
    ax.set_title("(4) Battery State of Charge", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 5. PPO continuous action
    ax = axes[4]
    ppo_actions = results["PPO (Ours)"]["actions"]
    ax.plot(hours[:len(ppo_actions)], ppo_actions,
            color="#2ca02c", linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.fill_between(hours[:len(ppo_actions)], ppo_actions, 0,
                    where=ppo_actions >  0.05,
                    color="#e07b39", alpha=0.4, label="Charging")
    ax.fill_between(hours[:len(ppo_actions)], ppo_actions, 0,
                    where=ppo_actions < -0.05,
                    color="#1f77b4", alpha=0.4, label="Discharging")
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Full Discharge", "Idle", "Full Charge"])
    ax.set_xlabel("Hour")
    ax.set_title("(5) PPO Continuous Action  [-1=discharge  0=idle  +1=charge]",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/policy_comparison.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved: results/policy_comparison.png")

    # summary table
    rows = []
    for name, r in results.items():
        saved = base - r["total_cost"]
        rows.append({
            "Strategy":        name,
            "Total Cost ($)":  f"{r['total_cost']:.4f}",
            "Mean Cost/h ($)": f"{r['mean_cost']:.4f}",
            "Saving ($)":      f"{saved:.4f}",
            "Saving (%)":      f"{saved/base*100:+.1f}%",
        })
    summary = pd.DataFrame(rows).set_index("Strategy")
    print("\nSummary:"); print(summary.to_string())
    summary.to_csv("results/summary_table.csv")
    print("\nTable saved: results/summary_table.csv")
    plt.show()


if __name__ == "__main__":
    main()