# CHECKLIST.md - Challenge 3 (Group 6)
**Environment:** `ALE/Venture-v5`  
**Algorithm:** PPO (Proximal Policy Optimization)  
**Group:** 6

---

## 📌 Exact command to reproduce the best PPO run

```bash
# Install dependencies (using Poetry)
poetry install

# Run training with best hyperparameters
poetry run python ppo_venture.py --train --steps 500000 --seed 42 --horizon 2048

#view tensorboard

tensorboard --logdir=logs/tensorboard