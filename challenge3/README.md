# Challenge 3 — Proximal Policy Optimization on ALE/Venture-v5 (Group 6)

## What this is about

This folder contains the code, trained models, and results for **ML Challenge 3**, focused on
**Proximal Policy Optimization (PPO)** applied to the Atari game **ALE/Venture-v5**, and a
direct empirical comparison against the **DQN agent developed in Challenge 1**.

- **Environment:** [ALE/Venture-v5](https://ale.farama.org/) — a hard-exploration dungeon game
  where the avatar "Winky" must navigate rooms, evade monsters, and collect treasures.
- **Algorithm:** Proximal Policy Optimization (PPO) with clipped surrogate objective, Generalised
  Advantage Estimation (GAE), entropy regularisation, and a shared convolutional actor-critic
  backbone.
- **Central research question:**
  > *"Under a fixed computational budget and on the same environment, does PPO converge faster,
  > reach higher performance, or exhibit different failure modes compared to the DQN agent designed
  > in Challenge 1?"*
- **Key finding:** PPO converges faster, achieves substantially higher training stability, and
  exhibits **no reward-exploitation behaviour** — a critical failure mode observed throughout DQN
  training in Challenge 1.

---

## Contents

```
challenge_3/
├── venture.py                  # Full PPO implementation (training + evaluation)
├── experiments.json            # Experiment definitions (hyperparameters, seeds)
├── pyproject.toml              # Python dependencies and project metadata
├── CHECKLIST.md                # Exact reproduce commands, seeds, and comparative summary
├── models/
│   ├── seed_42/
│   │   ├── best_model.pt       # Best checkpoint by episode return
│   │   ├── final_model.pt      # Final model weights after full training
│   │   └── metrics.json        # Full metrics JSON (returns, losses, entropy, etc.)
│   └── seed_7/
│       ├── best_model.pt
│       ├── final_model.pt
│       └── metrics.json
├── checkpoints/                # Periodic checkpoints (every 500 rollouts)
├── logs/
│   └── tensorboard/            # TensorBoard event files (one subdirectory per seed)
├── challenge3_group6_paper.pdf # IEEE paper (DQN + PPO results + comparison)
└── challenge3_group6_paper.tex # LaTeX source
```

---

## Experiments (what runs)

The script `venture.py` loads hyperparameters defined directly in `experiments.json` or
overridable via CLI flags. Each configuration controls:

| Parameter | Default |
|---|---|
| `total_steps` | 500,000 |
| `horizon` (rollout length T) | 2,048 |
| `n_epochs` (PPO epochs K) | 4 |
| `batch_size` | 128 |
| `learning_rate` | 2.5 × 10⁻⁴ |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_epsilon` ε | 0.2 |
| `ent_coef` c₂ | 0.01 |
| `vf_coef` c₁ | 0.5 |
| `max_grad_norm` | 0.5 |
| `seeds` | 42, 7 |

### Two network configurations tested

**Configuration 1 — Narrow Baseline (32 → 64 → 64)**
Standard DQN-equivalent backbone. Shows initial learning but exhibits higher variance and
partial policy degradation after early room discoveries.

**Configuration 2 — Wide Optimised (`VentureOptimizedNetwork`, 64 → 128 → 128)**
Wider convolutional channels + deeper FC layers (512 → 512) + separate actor and critic heads.
Achieves significantly lower return variance, earlier stable convergence, and zero reward
exploitation events. This is the **recommended configuration**.

### Preprocessing (identical to Challenge 1 for fair comparison)

1. Grayscale conversion (RGB → luminance)
2. Spatial resize to 84 × 84 pixels
3. Pixel normalisation to [0.0, 1.0]
4. Frame stacking: last 4 observations → input tensor (4, 84, 84)
5. Frame skip: action repeated for 4 consecutive simulator steps

---

## Results summary

### PPO vs DQN — Quantitative comparison

| Metric | DQN (Ch. 1) | PPO Narrow | PPO Wide |
|---|---|---|---|
| Total training steps | 1,000,000 | 500,000 | 500,000 |
| Steps to first reward | > 300,000 | < 100,000 | < 100,000 |
| Final mean return | 0 – 50 | moderate | higher |
| Return std (last 100 ep) | very high | moderate | low |
| Reward exploitation events | **Yes** | **No** | **No** |
| RAM reward shaping required | Yes | No | No |

### PPO — Per-experiment metrics (mean ± std, 2 seeds)

> Note: full numeric results per seed are available in `models/seed_*/metrics.json` and
> TensorBoard logs under `logs/tensorboard/`.

| Experiment | Best return | Final avg return (last 100 ep) | Avg dungeons/ep |
|---|---|---|---|
| PPO Narrow (seed 42) | see `metrics.json` | see `metrics.json` | see `metrics.json` |
| PPO Wide (seed 42) | see `metrics.json` | see `metrics.json` | see `metrics.json` |
| PPO Wide (seed 7) | see `metrics.json` | see `metrics.json` | see `metrics.json` |

### Key qualitative findings

- **Convergence speed:** PPO produces a meaningful gradient signal within the first 50,000–100,000
  steps. DQN required > 300,000 steps even with shaped rewards.
- **Stability:** PPO's clipped objective prevents catastrophic updates. DQN's loss oscillated
  violently whenever a sparse reward triggered a large TD error.
- **No reward exploitation:** DQN's replay buffer accumulated exploitative transitions
  (room-boundary crossing without collecting treasure). PPO's on-policy rollouts discard stale
  data after every update, making this failure mode structurally impossible.
- **No reward shaping needed:** PPO's entropy bonus is a functional substitute for the
  `VentureRewardWrapper` used in Challenge 1.

---

## Outputs and where to find them

For each run, the following artifacts are saved automatically:

| File | Location | Description |
|---|---|---|
| `best_model.pt` | `models/seed_<N>/` | Weights at highest episode return |
| `final_model.pt` | `models/seed_<N>/` | Weights at end of training |
| `metrics.json` | `models/seed_<N>/` | Full metrics dict (returns, losses, entropy, dungeons) |
| `returns.npy` | `models/seed_<N>/` | Episode returns array |
| `dungeons.npy` | `models/seed_<N>/` | Dungeon count per episode |
| `policy_losses.npy` | `models/seed_<N>/` | Policy loss per update |
| `checkpoint_<step>.pt` | `checkpoints/seed_<N>/` | Periodic checkpoint every 500 rollouts |
| TensorBoard events | `logs/tensorboard/venture_ppo_seed_<N>/` | Episode return, loss, entropy, advantage stats |

To visualise TensorBoard logs:
```bash
poetry run tensorboard --logdir=logs/tensorboard
```

---

## How to run

### 1) Install Poetry (if not already installed)

```bash
pip install poetry
```

Or follow the official installer at https://python-poetry.org/docs/#installation.

### 2) Install dependencies

From inside `challenge_3/`:

```bash
poetry install
```

Poetry reads `pyproject.toml`, creates an isolated virtual environment automatically, and
installs all pinned dependencies:

| Package | Version |
|---|---|
| `torch` | >= 2.12.0, < 3.0.0 |
| `numpy` | >= 2.4.6, < 3.0.0 |
| `gymnasium` | >= 1.3.0, < 2.0.0 |
| `ale-py` | >= 0.11.2, < 0.12.0 |
| `pillow` | >= 12.2.0, < 13.0.0 |
| `tdqm` | >= 0.0.1 |
| `tensorboard` | >= 2.20.0, < 3.0.0 |

Requires **Python >= 3.11, < 3.15** (enforced by `pyproject.toml`).

### 3) Activate the Poetry shell (optional)

```bash
poetry env activate
```

After this, all `python` and `tensorboard` commands run inside the managed environment.
Alternatively, prefix every command with `poetry run` without activating the shell.

### 4) Train the PPO agent (best configuration)

```bash
poetry run python venture.py --train --steps 500000 --seed 42 --horizon 2048
```

To train on a second seed for variance estimation:

```bash
poetry run python venture.py --train --steps 500000 --seed 7 --horizon 2048
```

### 5) Watch the trained agent play

```bash
poetry run python venture.py --watch --seed 42 --episodes 5
```

To specify a custom model path:

```bash
poetry run python venture.py --watch --model models/seed_42/best_model.pt --episodes 3
```

### 6) Useful CLI options

```
--train               Run training
--watch               Run visual evaluation
--steps N             Total environment steps (default: 500000)
--seed N              Random seed (default: 42)
--horizon N           Rollout horizon T (default: 2048)
--model PATH          Path to a saved model for --watch
--episodes N          Number of episodes to display in --watch mode
```

---

## Dependencies

Managed by **Poetry** via `pyproject.toml`. All version bounds are pinned for reproducibility:

```toml
[project]
requires-python = ">=3.11,<3.15"
dependencies = [
    "torch>=2.12.0,<3.0.0",
    "numpy>=2.4.6,<3.0.0",
    "gymnasium>=1.3.0,<2.0.0",
    "ale-py>=0.11.2,<0.12.0",
    "pillow>=12.2.0,<13.0.0",
    "tdqm>=0.0.1,<0.0.2",
    "tensorboard>=2.20.0,<3.0.0"
]
```

Run `poetry install` from `challenge_3/` to reproduce the exact environment.

---

## Comparison with Challenge 1 (DQN)

The DQN agent from Challenge 1 required a custom **RAM-assisted Reward Shaping** wrapper
(`VentureRewardWrapper`) that polled emulator memory bytes `0x60`, `0x61`, `0x80`, and `0x82`
to construct a dense reward manifold: exploration bonuses (+10 per new room), treasure
amplification (×4.5), and stagnation penalties. Even with this intervention, training was
characterised by:

- **Slow convergence:** first positive reward consistently delayed beyond 300,000 steps.
- **High variance:** reward oscillations of several hundred points per evaluation window.
- **Reward exploitation:** the agent learned to trigger exploration bonuses by crossing room
  boundaries without completing objectives, saturating the replay buffer with exploitative
  transitions.

PPO, without any reward shaping, overcame all three failure modes through its structural
properties: on-policy rollout collection, clipped ratio updates, and entropy regularisation.
Full algorithmic analysis is provided in `challenge3_group6_paper.pdf`.

---

## Notes / reproducibility

- Training on a single GPU (NVIDIA GTX 1080 Ti or equivalent) takes approximately 2–4 hours
  for 500,000 steps with the wide network configuration.
- CPU training is functional but roughly 10× slower; reduce `--steps` to 200,000 for a
  quick smoke-test.
- The ALE ROM for Venture is downloaded automatically by `ale-py` on first run; no manual
  ROM installation is required.
- **Python 3.11–3.14** is required as specified in `pyproject.toml`. Using `pyenv` or
  `conda` to manage the Python version before running `poetry install` is recommended.
- All random operations are seeded via `torch.manual_seed`, `np.random.seed`, and
  `random.seed` using the value passed to `--seed`.

---

## References

1. J. Schulman et al., "Proximal Policy Optimization Algorithms," *arXiv:1707.06347*, 2017.
2. J. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage
   Estimation," *ICLR*, 2016.
3. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015.
4. M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning,"
   *AAAI*, 2018.
5. L. Engstrom et al., "Implementation Matters in Deep RL: A Case Study on PPO and TRPO,"
   *ICLR*, 2020.
6. M. Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning?", *arXiv:2006.05990*, 2021.

Full citation list in IEEE format is available in `challenge3_group6_paper.pdf` and
`bibliography.bib`.

---

*Any questions about this challenge can be sent to: cavirguezs@udistrital.edu.co*
