"""
train_ppo.py — PPO training for Microgrid EMS (v4)
Produces: results/training_curve.png + results/training_report.txt
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from RL_env import BatteryStorageEnv


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, log_freq=2000):
        super().__init__(verbose=0)
        self.log_freq   = log_freq
        self.steps_log  = []
        self.reward_log = []

    def _on_step(self) -> bool:
        if (self.n_calls % self.log_freq == 0
                and len(self.model.ep_info_buffer) > 0):
            mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            self.steps_log.append(self.n_calls)
            self.reward_log.append(mean_r)
            print(f"  Step {self.n_calls:>7d} | Mean episode reward: {mean_r:.2f}")
        return True


def load_eval_results(path="logs/evaluations.npz"):
    if not os.path.exists(path):
        return [], []
    d = np.load(path)
    return d["timesteps"].tolist(), d["results"].mean(axis=1).tolist()


def plot_training_curve(steps, rewards, eval_steps, eval_rewards):
    fig, ax = plt.subplots(figsize=(12, 5))
    if steps:
        ax.plot(steps, rewards, color="#1f77b4", lw=1.5,
                alpha=0.7, label="Train reward (rolling mean)")
    if eval_steps:
        ax.plot(eval_steps, eval_rewards, color="#2ca02c", lw=2,
                marker="o", ms=4, label="Eval reward (deterministic)")
        best_idx = int(np.argmax(eval_rewards))
        ax.annotate(
            f"Best: {eval_rewards[best_idx]:.1f}\n@ step {eval_steps[best_idx]:,}",
            xy=(eval_steps[best_idx], eval_rewards[best_idx]),
            xytext=(eval_steps[best_idx], eval_rewards[best_idx] * 0.85),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9, color="red",
        )
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Reward  (= -(7-day cost + battery penalty))")
    ax.set_title("PPO Training — Microgrid EMS with Solar + Price Signals", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/training_curve.png", dpi=150, bbox_inches="tight")
    print("Saved: results/training_curve.png")
    plt.show()


def write_report(total_timesteps, eval_steps, eval_rewards, hyperparams):
    if eval_rewards:
        bi = int(np.argmax(eval_rewards))
        best_r, best_s, final_r = eval_rewards[bi], eval_steps[bi], eval_rewards[-1]
    else:
        best_r = final_r = float("nan"); best_s = 0

    lines = [
        "=" * 60,
        "  PPO Training Report — Microgrid EMS",
        "=" * 60,
        "",
        "[System Design]",
        "  ML:  LSTM dual-output (load + solar prediction)",
        "  RL:  PPO with continuous action [-1, 1]",
        "",
        "[State Space (5-dim, all normalised to [0,1])]",
        "  predicted_load    LSTM forecast",
        "  predicted_solar   LSTM forecast",
        "  price             current electricity price",
        "  battery_soc       battery state of charge",
        "  hour_of_day       time signal",
        "",
        "[Action]",
        "  Continuous: battery_power in [-1, 1]",
        "  -1=full discharge | 0=idle | +1=full charge",
        "",
        "[Energy Flow]",
        "  Solar -> Load -> Battery -> Grid",
        "  grid_power = net_load - battery_discharge + battery_charge",
        "",
        "[Reward]",
        "  reward = -cost - 0.001*|action|",
        "  cost   = buy_price * max(grid_power, 0)",
        "         - sell_price * max(-grid_power, 0)",
        "",
        "[Episode]",
        "  Length: 168 steps = 7 days",
        "",
        "[Hyperparameters]",
        f"  Total timesteps: {total_timesteps:,}",
    ] + [f"  {k}: {v}" for k, v in hyperparams.items()] + [
        "",
        "[Results]",
        f"  Best eval reward:  {best_r:.2f}  (step {best_s:,})",
        f"  Final eval reward: {final_r:.2f}",
        f"  Best model:        saved_models/best_model.zip",
        "=" * 60,
    ]
    os.makedirs("results", exist_ok=True)
    with open("results/training_report.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nSaved: results/training_report.txt")


def train_ppo(total_timesteps=200_000, n_envs=4,
              save_path="saved_models/ppo_battery"):

    print("=" * 55)
    print("  Train PPO — Microgrid EMS (Solar + Price Signals)")
    print("=" * 55)

    train_env = make_vec_env(lambda: Monitor(BatteryStorageEnv()), n_envs=n_envs)
    eval_env  = Monitor(BatteryStorageEnv())

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs",         exist_ok=True)

    logger_cb = TrainingLoggerCallback(log_freq=2000)
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path="saved_models/",
        log_path="logs/",
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    hyperparams = {
        "learning_rate": 3e-4,
        "n_steps":       2048,
        "batch_size":    64,
        "n_epochs":      10,
        "gamma":         0.999,
        "gae_lambda":    0.95,
        "ent_coef":      0.012,
        "clip_range":    0.2,
    }

    model = PPO("MlpPolicy", train_env, **hyperparams, verbose=0)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[logger_cb, eval_cb],
        progress_bar=True,
    )
    model.save(save_path)
    print(f"\nFinal model saved: {save_path}.zip")

    eval_steps, eval_rewards = load_eval_results()
    plot_training_curve(logger_cb.steps_log, logger_cb.reward_log,
                        eval_steps, eval_rewards)
    write_report(total_timesteps, eval_steps, eval_rewards, hyperparams)

    # ── Quick post-training evaluation ──
    quick_eval(save_path)
    print("\nDone! Next: python evaluate.py")


# ─────────────────────────────────────────
# Quick post-training evaluation
# Runs immediately after training finishes.
# Compares PPO vs Rule-Based vs Idle over ONE 7-day episode.
# Purpose: fast sanity check that PPO actually learned something.
# ─────────────────────────────────────────

def quick_eval(model_path="saved_models/ppo_battery"):
    print("\n" + "=" * 55)
    print("  QUICK POST-TRAINING EVALUATION (7 days)")
    print("=" * 55)

    from RL_env import (BatteryStorageEnv, SEQ_LEN, EPISODE_HOURS,price_model, generate_solar)

    # load best model (saved by EvalCallback)
    best_path = "saved_models/best_model"
    try:
        ppo = PPO.load(best_path)
        print(f"  Loaded: {best_path}.zip")
    except Exception:
        ppo = PPO.load(model_path)
        print(f"  Loaded: {model_path}.zip  (best_model not found)")

    env = BatteryStorageEnv()

    def fixed_reset():
        obs, _ = env.reset(seed=0)
        env.current_step  = SEQ_LEN
        env.episode_start = SEQ_LEN
        env.battery_soc   = env.battery_cap * 0.5
        env.total_cost    = 0.0
        return env._get_obs()

    def run(policy_fn, label):
        obs = fixed_reset()
        total = 0.0
        n_charge = n_discharge = n_idle = 0
        for _ in range(EPISODE_HOURS):
            action = policy_fn(obs)
            obs, _, done, _, info = env.step(action)
            total += info["cost"]
            a = info["action_raw"]
            if   a >  0.05: n_charge    += 1
            elif a < -0.05: n_discharge += 1
            else:           n_idle      += 1
            if done: break
        print(f"  {label:<22s}  cost=${total:.4f}  "
              f"[C={n_charge} I={n_idle} D={n_discharge}]")
        return total

    c_idle = run(lambda o: np.array([0.0]),  "Always Idle")
    c_rule = run(
        lambda o: (
            np.array([ 1.0]) if generate_solar(env._get_hour(), False) > 1.0
            else np.array([-1.0]) if price_model(env._get_hour()) >= 0.40
            else np.array([ 0.0])
        ),
        "Rule-Based"
    )
    c_ppo  = run(lambda o: ppo.predict(o, deterministic=True)[0], "PPO (best_model)")

    print()
    saving_rule = c_idle - c_rule
    saving_ppo  = c_idle - c_ppo
    print(f"  Rule-Based saved ${saving_rule:.4f} ({saving_rule/c_idle*100:.1f}% vs Idle)")
    print(f"  PPO        saved ${saving_ppo:.4f}  ({saving_ppo/c_idle*100:.1f}% vs Idle)")

    if c_ppo < c_idle:
        print("\n  ✅ PPO learned a cost-reducing strategy!")
    else:
        print("\n  ⚠️  PPO did not beat Idle — consider more training steps.")

    if c_ppo <= c_rule:
        print("  ✅ PPO matched or beat Rule-Based!")
    else:
        diff = c_ppo - c_rule
        print(f"  ℹ️  PPO is ${diff:.4f} more expensive than Rule-Based.")
        print("     More training steps or tuning may close this gap.")

    print("=" * 55)


if __name__ == "__main__":
    train_ppo()