"""
Atari DQN — Train and Play with Stable-Baselines3
==================================================
Group 6 — ALE/Venture-v5
Challenge 1: Machine Learning — Universidad Distrital Francisco José de Caldas
Prof. Carlos Andrés Sierra, M.Sc.
 
Environment notes (Venture-v5):
  - High difficulty early; sparse positive signals in a noisy action space.
  - The agent must navigate rooms and collect treasures while avoiding monsters.
  - Rewards are sparse and require sustained exploration.
  - Frame stacking is critical to capture monster motion and anticipate threats.
 
How it works (high level):
  1. The environment renders raw pixel frames (84 × 84 grayscale after preprocessing).
  2. The agent stacks the last 4 frames to capture motion.
  3. A CNN policy learns which action maximises future reward via the Bellman equation:
       Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]
  4. An epsilon-greedy schedule balances exploration vs exploitation during training.
 
Usage
-----
  # Train with built-in defaults (baseline)
  python venture_dqn.py --mode train --model-path models/venture_dqn
 
  # Train a specific named experiment from the JSON config
  python venture_dqn.py --mode train --experiment exp_02_lr_high --model-path models/venture_exp02
 
  # Watch the trained agent play
  python venture_dqn.py --mode play --model-path models/venture_dqn --episodes 3
 
  # Run all experiments from sweep_configs.json and keep the best model
  python venture_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/venture_best
 
  # Inspect saved model hyperparameters
  python venture_dqn.py --mode inspect --model-path models/venture_dqn
 
  # Monitor all sweep runs in TensorBoard
  python -m tensorboard.main --logdir logs/venture_dqn/sweep --port 6006
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import shutil
from pathlib import Path
 
import torch
import numpy as np
import ale_py
import gymnasium as gym
 
gym.register_envs(ale_py)  # register ALE environments in the gymnasium namespace
 
from torch.utils.tensorboard import SummaryWriter
 
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
 
# Group 6 environment
ENV_ID = "ALE/Venture-v5"
 
# Number of consecutive frames stacked together as a single observation.
# Critical for Venture: monsters move fast and the agent needs motion context.
N_STACK = 4
 
 
# ─── Reward Shaping Wrapper ──────────────────────────────────────────────────
 
class VentureRewardWrapper(gym.Wrapper):
    """Custom reward shaping for ALE/Venture-v5.
 
    Problem without shaping:
      The agent learns to loop in a single room because any positive reward
      (even tiny) is better than risking death in new rooms. This is a classic
      reward hacking / local optimum caused by sparse signals.
 
    Shaping strategy:
      1. TREASURE BONUS (2x multiplier): amplify the original game reward for
         collecting a treasure. Makes collecting worth the risk of entering.
      2. DEATH PENALTY (-2.0): small penalty on life loss to discourage passive
         survival loops (the agent already gets terminal_on_life_loss, this
         reinforces the signal).

    Note on reward clipping:
      Standard Atari preprocessing clips rewards to {-1, 0, +1}. We keep the
      shaped reward outside that clipping so the model can still see the reward
      magnitude for rare positive events.
    """
 
    TREASURE_MULT        =  2.0
    DEATH_PENALTY        = -5.0

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._prev_lives: int | None = None

    def reset(self, **kwargs):
        self._prev_lives = None
        obs, info = self.env.reset(**kwargs)
        self._prev_lives = info.get("lives", 3)
        return obs, info
 
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = float(reward)

        if reward > 0:
            shaped_reward = reward * self.TREASURE_MULT
 
        current_lives = info.get("lives", self._prev_lives)
        if self._prev_lives is not None and current_lives < self._prev_lives:
            shaped_reward += self.DEATH_PENALTY
        self._prev_lives = current_lives
 
        return obs, shaped_reward, terminated, truncated, info
 
 
# ─── TensorBoard Callback ────────────────────────────────────────────────────
 
class TensorBoardCallback(BaseCallback):
    """Custom callback that logs per-episode metrics to TensorBoard.
 
    Attaches to the same SummaryWriter that SB3 creates internally, so our
    custom scalars land in the exact same event file as the built-in
    rollout/ and train/ metrics.
 
    Scalars added by this callback:
      - training/episode_reward : total reward accumulated in each episode
      - training/epsilon        : current exploration rate (ε), logged every step
 
    SB3 built-in scalars (also visible in the same run):
      - rollout/ep_rew_mean : rolling mean reward over the last 100 episodes
      - train/loss          : TD-error loss
      - train/learning_rate : current learning rate
    """
 
    def __init__(self) -> None:
        super().__init__()
        self._writer: SummaryWriter | None = None
        self._episode_reward = 0.0
 
    def _on_training_start(self) -> None:
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.model._logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._writer = fmt.writer
                return
        self._writer = None
 
    def _on_step(self) -> bool:
        if self._writer is None:
            return True
 
        self._episode_reward += float(self.locals["rewards"][0])
 
        # Log epsilon every step → smooth decay curve in TensorBoard.
        self._writer.add_scalar(
            "training/epsilon",
            self.model.exploration_rate,
            self.num_timesteps,
        )
 
        if self.locals["dones"][0]:
            self._writer.add_scalar(
                "training/episode_reward",
                self._episode_reward,
                self.num_timesteps,
            )
            self._episode_reward = 0.0
 
        return True
 
 
# ─── Environment Builders ────────────────────────────────────────────────────
 
def build_training_environment(seed: int, reward_shaping: bool = True) -> VecFrameStack:
    """Create a vectorised, preprocessed Atari environment for training.
 
    Wrapper order (inner → outer):
      1. gym.make(ENV_ID)           — raw ALE environment
      2. VentureRewardWrapper       — reward shaping (room bonus, death penalty)
      3. AtariWrapper               — grayscale, resize, frameskip, clip_reward
      4. DummyVecEnv                — vectorisation required by SB3
      5. VecFrameStack              — stack last 4 frames as one observation
 
    Reward shaping is applied BEFORE AtariWrapper's clip so the sign of each
    bonus/penalty survives clipping and guides learning correctly.
 
    Args:
        seed:            Random seed for reproducibility.
        reward_shaping:  Apply VentureRewardWrapper (default True).
 
    Returns:
        A VecFrameStack-wrapped vectorised environment ready for DQN.
    """
    def _make_shaped_env():
        base = gym.make(ENV_ID)
        if reward_shaping:
            base = VentureRewardWrapper(base)
        # Preserve shaped reward magnitude in the environment; clip in wrapper if needed.
        base = AtariWrapper(base, terminal_on_life_loss=True, clip_reward=False)
        return base
 
    env = DummyVecEnv([_make_shaped_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env
 
 
def build_playing_environment() -> VecFrameStack:
    """Create a human-rendered Atari environment for watching the agent play.
 
    Differences from the training environment:
      - render_mode="human" opens a visible game window.
      - clip_reward=False: show the real score instead of {-1, 0, +1}.
 
    Returns:
        A VecFrameStack-wrapped environment with a human-visible window.
    """
    def _make_single_env() -> AtariWrapper:
        base_env = gym.make(ENV_ID, render_mode="human")
        return AtariWrapper(base_env, terminal_on_life_loss=True, clip_reward=False)
 
    env = DummyVecEnv([_make_single_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env
 
 
# ─── Core Logic ──────────────────────────────────────────────────────────────
 
def train_agent(
    model_path: str,
    timesteps: int,
    seed: int,
    tensorboard_log: str,
    hparams: dict | None = None,
) -> float:
    """Train a DQN agent on Venture-v5 and save the model.
 
    Baseline hyperparameter rationale (tuned for Venture's sparse reward signal):
 
      learning_rate         1e-4   — standard SB3-Zoo Atari lr; stable for CNNs
      buffer_size           100_000 — larger buffer helps with Venture's sparse
                                      rewards: diverse experiences improve learning
      learning_starts       25_000 — fill 25% of buffer before updating; ensures
                                      enough varied transitions before first update
      batch_size            32     — smaller batches work well with sparse rewards
      gamma                 0.99   — standard Atari discount factor
      train_freq            4      — one update every 4 env steps (DQN paper)
      target_update         2_000  — moderate sync interval; balances stability
                                     and tracking of policy improvement
      exploration_fraction  0.25   — longer exploration critical for sparse Venture;
                                     ε decays over 25% of training steps
      exploration_final_eps 0.01   — 1% random floor; standard Atari value
 
    Args:
        model_path:      Path (without .zip) where the trained model is saved.
        timesteps:       Total environment steps to train for.
        seed:            Random seed for reproducibility.
        tensorboard_log: Directory where TensorBoard event files are written.
        hparams:         Optional hyperparameter dict; uses built-in defaults when None.
 
    Returns:
        Mean episode reward over the last episodes in SB3's episode info buffer.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
 
    if hparams is None:
        hparams = dict(
            env_id=ENV_ID,
            learning_rate=5e-5,
            buffer_size=200_000,
            learning_starts=25_000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            target_update_interval=2_000,
            exploration_fraction=0.5,
            exploration_final_eps=0.01,
            reward_shaping=True,
            timesteps=timesteps,
            seed=seed,
        )
 
    # Write hparams to TensorBoard → visible in the HPARAMS tab.
    _tb_writer = SummaryWriter(log_dir=tensorboard_log)
    _tb_writer.add_hparams(hparams, metric_dict={"hparam/episode_reward": 0})
    _tb_writer.close()
 
    reward_shaping = hparams.get("reward_shaping", True)
    env = build_training_environment(seed=seed, reward_shaping=reward_shaping)
 
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=hparams["learning_rate"],
        buffer_size=hparams["buffer_size"],
        learning_starts=hparams["learning_starts"],
        batch_size=hparams["batch_size"],
        tau=1.0,
        gamma=hparams["gamma"],
        train_freq=hparams["train_freq"],
        gradient_steps=1,
        target_update_interval=hparams["target_update_interval"],
        exploration_fraction=hparams["exploration_fraction"],
        exploration_final_eps=hparams["exploration_final_eps"],
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )
 
    model.learn(
        total_timesteps=timesteps,
        callback=TensorBoardCallback(),
        progress_bar=True,
    )
    model.save(model_path)
    env.close()
    print(f"Model saved → {model_path}.zip")
 
    if model.ep_info_buffer:
        return float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
    return 0.0
 
 
def play_agent(model_path: str, episodes: int) -> None:
    """Load a trained model and watch it play in a visible game window.
 
    Args:
        model_path: Path to the saved model (with or without .zip extension).
        episodes:   Number of full games to play before exiting.
 
    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip\n"
            "Run with --mode train first to create a model."
        )
 
    env = build_playing_environment()
    model = DQN.load(model_path, env=env)
 
    completed = 0
    obs = env.reset()
    episode_reward = 0.0
 
    while completed < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])
 
        if dones[0]:
            if infos[0].get("lives", 0) == 0:
                completed += 1
                print(f"Episode {completed}/{episodes}  reward: {episode_reward:.2f}")
                episode_reward = 0.0
 
    env.close()
 
 
# ─── Hyperparameter Sweep ────────────────────────────────────────────────────
 
def run_sweep(
    sweep_path: str,
    default_timesteps: int,
    seed: int,
    base_log_dir: str,
    best_model_path: str,
) -> None:
    """Run all experiments defined in a JSON config file and save the best model.
 
    Each experiment uses the ``timesteps`` value from its JSON entry.
    TensorBoard logs for every run are written to
    ``<base_log_dir>/sweep/<experiment_name>/``.
 
    After all experiments finish, only the model with the highest mean final
    episode reward is kept at ``best_model_path``.
 
    Args:
        sweep_path:        Path to the JSON file containing experiment configs.
        default_timesteps: Fallback timestep budget.
        seed:              Random seed applied to every experiment.
        base_log_dir:      Root TensorBoard log directory.
        best_model_path:   Where to save the winning model (without .zip).
    """
    with open(sweep_path) as f:
        configs = json.load(f)
 
    tmp_model_dir = Path("models") / "_sweep_tmp"
    tmp_model_dir.mkdir(parents=True, exist_ok=True)
 
    results: list[tuple[str, float]] = []
    total = len(configs)
 
    for idx, cfg in enumerate(configs, start=1):
        name = cfg.get("name", f"exp_{idx:02d}")
        note = cfg.get("note", "")
        exp_timesteps = cfg.get("timesteps", default_timesteps)
        print(f"\nExperiment {idx}/{total}: {name}  ({exp_timesteps:,} steps)")
        if note:
            print(f"  Note: {note}")
        print("=" * 60)
 
        hparams = {
            "env_id":                 ENV_ID,
            "learning_rate":          cfg["learning_rate"],
            "buffer_size":            cfg["buffer_size"],
            "learning_starts":        cfg["learning_starts"],
            "batch_size":             cfg["batch_size"],
            "gamma":                  cfg["gamma"],
            "train_freq":             cfg["train_freq"],
            "target_update_interval": cfg["target_update_interval"],
            "exploration_fraction":   cfg["exploration_fraction"],
            "exploration_final_eps":  cfg["exploration_final_eps"],
            "timesteps":              exp_timesteps,
            "seed":                   seed,
        }
 
        model_path = str(tmp_model_dir / name)
        log_dir    = f"{base_log_dir}/sweep/{name}"
 
        score = train_agent(
            model_path=model_path,
            timesteps=exp_timesteps,
            seed=seed,
            tensorboard_log=log_dir,
            hparams=hparams,
        )
        results.append((name, score))
        print(f"  → final mean reward: {score:.2f}")
 
    # Rank and display results
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = results[0]
 
    print(f"\n{'='*60}")
    print("Sweep complete — results ranked by final mean reward:")
    for rank, (name, score) in enumerate(results, start=1):
        marker = "  ← BEST" if rank == 1 else ""
        print(f"  {rank:2d}. {name:<40s}  {score:7.2f}{marker}")
    print(f"{'='*60}")
 
    # Save best model, clean up temp directory
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_model_dir / f"{best_name}.zip"), f"{best_model_path}.zip")
    shutil.rmtree(tmp_model_dir)
 
    print(f"\nBest model ({best_name}, score={best_score:.2f}) saved → {best_model_path}.zip")
    print(f"TensorBoard logs: {base_log_dir}/sweep/")
 
 
def inspect_model(model_path: str) -> None:
    """Load a saved model and print its hyperparameters.
 
    Args:
        model_path: Path to the saved model (with or without .zip extension).
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")
 
    model = DQN.load(model_path)
 
    params = {
        "policy":                  model.policy_class.__name__,
        "learning_rate":           model.learning_rate,
        "buffer_size":             model.buffer_size,
        "learning_starts":         model.learning_starts,
        "batch_size":              model.batch_size,
        "tau":                     model.tau,
        "gamma":                   model.gamma,
        "train_freq":              model.train_freq,
        "gradient_steps":          model.gradient_steps,
        "target_update_interval":  model.target_update_interval,
        "exploration_fraction":    model.exploration_fraction,
        "exploration_final_eps":   model.exploration_final_eps,
        "num_timesteps_trained":   model.num_timesteps,
    }
 
    print(f"\n── Saved model: {model_path}.zip")
    for key, value in params.items():
        print(f"  {key:30s}: {value}")
    print("─" * 55 + "\n")
 
 
# ─── CLI ─────────────────────────────────────────────────────────────────────
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or watch a DQN agent on ALE/Venture-v5 (Group 6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "play", "inspect", "sweep"], required=True,
        help="'train' single run | 'play' watch agent | 'inspect' print params | "
             "'sweep' run all experiments from --sweep-file.",
    )
    parser.add_argument(
        "--sweep-file", default="sweep_configs.json",
        help="Path to JSON file with experiment configs.",
    )
    parser.add_argument(
        "--experiment", default=None,
        help="Name of a single experiment in --sweep-file to run with --mode train.",
    )
    parser.add_argument(
        "--model-path", default="models/venture_dqn",
        help="Path to save (train) or load (play/inspect) the model (without .zip).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training steps. Overrides the JSON value when set.",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of full games to play in play mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensorboard-log", default="logs/venture_dqn",
        help="Directory for TensorBoard logs.",
    )
    return parser.parse_args()
 
 
def main():
    print("GPU Available NVIDIA SI:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    print("CUDA Version:", torch.version.cuda)
    args = parse_args()
 
    if args.mode == "train":
        hparams = None
        timesteps = args.timesteps or 300_000
 
        if args.experiment:
            with open(args.sweep_file) as f:
                configs = {c["name"]: c for c in json.load(f)}
            if args.experiment not in configs:
                raise ValueError(
                    f"Experiment '{args.experiment}' not found in {args.sweep_file}.\n"
                    f"Available: {', '.join(configs)}"
                )
            cfg = configs[args.experiment]
            timesteps = args.timesteps or cfg.get("timesteps", 300_000)
            hparams = {
                "env_id":                 ENV_ID,
                "learning_rate":          cfg["learning_rate"],
                "buffer_size":            cfg["buffer_size"],
                "learning_starts":        cfg["learning_starts"],
                "batch_size":             cfg["batch_size"],
                "gamma":                  cfg["gamma"],
                "train_freq":             cfg["train_freq"],
                "target_update_interval": cfg["target_update_interval"],
                "exploration_fraction":   cfg["exploration_fraction"],
                "exploration_final_eps":  cfg["exploration_final_eps"],
                "timesteps":              timesteps,
                "seed":                   args.seed,
            }
            print(f"Loaded experiment '{args.experiment}' from {args.sweep_file}")
 
        train_agent(
            model_path=args.model_path,
            timesteps=timesteps,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log,
            hparams=hparams,
        )
 
    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes)
 
    elif args.mode == "sweep":
        run_sweep(
            sweep_path=args.sweep_file,
            default_timesteps=args.timesteps or 300_000,
            seed=args.seed,
            base_log_dir=args.tensorboard_log,
            best_model_path=args.model_path,
        )
 
    else:
        inspect_model(model_path=args.model_path)
 
 
if __name__ == "__main__":
    main()