"""
train_ppo.py — PPO training (v5, discrete action)
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

from RL_env2 import BatteryStorageEnv, EPISODE_HOURS


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
        bi = int(np.argmax(eval_rewards))
        ax.annotate(
            f"Best: {eval_rewards[bi]:.2f}\n@ step {eval_steps[bi]:,}",
            xy=(eval_steps[bi], eval_rewards[bi]),
            xytext=(eval_steps[bi], eval_rewards[bi] * 0.85),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9, color="red",
        )
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Reward (shaped, 24-hour episode)")
    ax.set_title("PPO Training — Microgrid EMS (Discrete Action)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/training_curve.png", dpi=150, bbox_inches="tight")
    print("Saved: results/training_curve.png")
    plt.show()


def write_report(total_timesteps, eval_steps, eval_rewards, hyperparams):
    if eval_rewards:
        bi = int(np.argmax(eval_rewards))
        br, bs, fr = eval_rewards[bi], eval_steps[bi], eval_rewards[-1]
    else:
        br = fr = float("nan"); bs = 0

    lines = [
        "=" * 60,
        "  PPO Training Report v5 — Microgrid EMS",
        "=" * 60,
        "",
        "[Key Design Choices]",
        "  Action: DISCRETE {0=discharge, 1=idle, 2=charge}",
        "  Episode: 24 hours (1 day) — fast credit assignment",
        "  Reward: base(-cost) + shaping(charge/discharge bonuses)",
        "  State: 6-dim [load, solar, price, soc, hour, net_load]",
        "",
        "[Reward Shaping]",
        "  +0.3 * stored/cap  when charging at off-peak ($0.10)",
        "  +0.2 * stored/cap  when charging with solar surplus",
        "  +0.3 * discharged  when discharging at peak ($0.40)",
        "  -0.1               when SoC<0.5 at peak (missed chance)",
        "  -0.2               when charging at peak (wasteful)",
        "",
        f"[Results]",
        f"  Best eval reward:  {br:.2f}  (step {bs:,})",
        f"  Final eval reward: {fr:.2f}",
        f"  Best model:        saved_models/best_model.zip",
        "=" * 60,
    ]
    os.makedirs("results", exist_ok=True)
    with open("results/training_report.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nSaved: results/training_report.txt")


def quick_eval(model_path="saved_models/ppo_battery"):
    print("\n" + "=" * 55)
    print("  QUICK POST-TRAINING EVALUATION (24 hours)")
    print("=" * 55)

    from RL_env2 import (BatteryStorageEnv, SEQ_LEN, EPISODE_HOURS,
                                 price_model, generate_solar)

    try:
        ppo = PPO.load("saved_models/best_model")
        print("  Loaded: saved_models/best_model.zip")
    except Exception:
        ppo = PPO.load(model_path)
        print(f"  Loaded: {model_path}.zip")

    env = BatteryStorageEnv()

    def fixed_reset():
        obs, _ = env.reset(seed=0)
        env.current_step  = SEQ_LEN
        env.episode_start = SEQ_LEN
        env.battery_soc   = env.battery_cap * 0.5
        env.total_cost    = 0.0
        return env._get_obs()

    def run(label, policy_fn):
        obs = fixed_reset()
        total = 0.0; nc = nd = ni = 0
        for _ in range(EPISODE_HOURS):
            a = policy_fn(obs)
            obs, _, done, _, info = env.step(a)
            total += info["cost"]
            if   a == 2: nc += 1
            elif a == 0: nd += 1
            else:        ni += 1
            if done: break
        print(f"  {label:<22s}  cost=${total:.4f}  "
              f"[C={nc:>3d} I={ni:>3d} D={nd:>3d}]")
        return total

    c_idle = run("Always Idle",   lambda o: 1)

    def rule_action(o):
        h = env._get_hour(); s = generate_solar(h,False); p = price_model(h)
        if s > 1.0:  return 2
        elif p >= 0.40: return 0
        else:           return 1
    c_rule = run("Rule-Based",    rule_action)
    c_ppo  = run("PPO (best)",    lambda o: int(ppo.predict(o, deterministic=True)[0]))

    print()
    si = c_idle - c_ppo
    sr = c_idle - c_rule
    print(f"  Idle→PPO  saving: ${si:.4f} ({si/c_idle*100:.1f}%)")
    print(f"  Idle→Rule saving: ${sr:.4f} ({sr/c_idle*100:.1f}%)")

    if c_ppo < c_idle:
        print("\n  ✅ PPO learned to reduce cost vs Idle!")
    else:
        print("\n  ⚠️  PPO didn't beat Idle. Try more training steps.")

    if c_ppo <= c_rule * 1.1:
        print("  ✅ PPO is within 10% of Rule-Based — good result!")
    else:
        gap = c_ppo - c_rule
        print(f"  ℹ️  PPO is ${gap:.4f} more than Rule-Based. More steps may help.")
    print("=" * 55)


def train_ppo(total_timesteps=300_000, n_envs=4,
              save_path="saved_models/ppo_battery"):

    print("=" * 55)
    print("  Train PPO v5 — Discrete Action + Reward Shaping")
    print("=" * 55)
    print(f"  Episode length: {EPISODE_HOURS}h | Action: discrete {{0,1,2}}")

    train_env = make_vec_env(lambda: Monitor(BatteryStorageEnv()), n_envs=n_envs)
    eval_env  = Monitor(BatteryStorageEnv())

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs",         exist_ok=True)

    logger_cb = TrainingLoggerCallback(log_freq=2000)
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path="saved_models/",
        log_path="logs/",
        eval_freq=max(5_000 // n_envs, 1),   # evaluate more often
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    hyperparams = {
        "learning_rate": 3e-4,
        "n_steps":       512,      # shorter rollout matches 24h episode
        "batch_size":    64,
        "n_epochs":      10,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "ent_coef":      0.01,
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
    quick_eval(save_path)
    print("\nDone! Next: python evaluate.py")


if __name__ == "__main__":
    train_ppo()