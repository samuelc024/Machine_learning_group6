## CHALLENGE 1 - REINFORCEMENT LEARNING FOR ATARI (VENTURE)

Venture is a classic Atari arcade game where the player explores a series of dungeon-like rooms, collects treasure, avoids enemies, and tries to survive long enough to maximize the score. The game focuses on moving carefully through each room, timing actions well, and making quick decisions under pressure.

### About the Challenge

The goal of this challenge is to solve Venture using Deep Q Network (DQN) reinforcement learning. The agent must learn how to interact with the Atari environment, improve its policy over time, and achieve better game performance through experience. More details about the setup and requirements are available in [Challenge1.pdf](Challenge1.pdf).

## Requirements

- Linux recommended
- Python 3.12 or newer
- Optional: NVIDIA GPU with CUDA for faster training

## Required Dependencies

Main dependencies used by this project:

- torch
- numpy
- ale-py
- gymnasium
- stable-baselines3
- tensorboard
- tqdm
- rich

Note: argparse and pathlib are part of Python standard library.

## Installation and Virtual Environment (venv)

From the challenge_1 folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy ale-py gymnasium stable-baselines3 tensorboard tqdm rich
```

If you prefer Poetry (pyproject.toml is already included):

```bash
poetry install
poetry run python venture_dqn.py --help
```

## How to Use venture_dqn.py

The script supports 4 modes: train, play, inspect, and sweep.

### 1) Train with baseline defaults

```bash
python venture_dqn.py --mode train --model-path models/venture_dqn
```

If you do not provide --timesteps, training uses 300000 steps by default.

### 2) Recommended training budget

For Venture, it is recommended to train with at least 1000000 timesteps for more stable behavior:

```bash
python venture_dqn.py --mode train --model-path models/venture_dqn --timesteps 1000000
```

To train like our best run use:

```bash
python venture_dqn.py --mode train --model-path models/venture_dqn --timesteps 1500000
```

### 3) Train a specific experiment from sweep file

```bash
python venture_dqn.py --mode train --experiment exp_02_lr_high --sweep-file sweep_configs.json --model-path models/venture_exp02
```

### 4) Run full sweep and keep best model

```bash
python venture_dqn.py --mode sweep --sweep-file sweep_configs.json --model-path models/venture_best
```

### 5) Watch a trained agent

```bash
python venture_dqn.py --mode play --model-path models/venture_dqn --episodes 3
```

To see our best agent use [Venture Model 3](models\venture3.zip)

```bash
python venture_dqn.py --mode play --model-path models/venture3 --episodes 3
```

### 6) Monitor runs with TensorBoard

```bash
python -m tensorboard.main --logdir logs/venture_dqn --port 6006
```

## Default Hyperparameters (baseline train mode)

When running train mode without --experiment, the script uses these defaults:

- env_id: ALE/Venture-v5
- learning_rate: 1e-4
- buffer_size: 150000
- learning_starts: 20000
- batch_size: 64
- gamma: 0.99
- train_freq: 4
- gradient_steps: 2
- target_update_interval: 8000
- exploration_fraction: 0.60
- exploration_final_eps: 0.05
- reward_shaping: True
- terminal_on_life_loss: False
- optimize_memory_usage: True
- handle_timeout_termination: False when optimize_memory_usage=True
- seed: 42

## Useful CLI Defaults

- --model-path: models/venture_dqn
- --tensorboard-log: logs/venture_dqn
- --episodes: 3
- --seed: 42
- --sweep-file: sweep_configs.json
- --timesteps: None in CLI (falls back to 300000 in train/sweep if not provided)

## Output Structure

- Models are saved in models/ as .zip files
- TensorBoard logs are saved in logs/venture_dqn/

## Practical Notes

- The script prints GPU and CUDA availability at startup.
- On RAM-limited systems, keep optimize_memory_usage=True to reduce replay buffer memory footprint.

### 

## VIDEO

[Group 6 / Challenge 1 Video](https://drive.google.com/drive/folders/1nqwb9-S4W8uKuwVLx-iSxD-YRPQ_60Bs?usp=drive_link).
