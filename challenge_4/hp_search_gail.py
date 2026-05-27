import subprocess
import itertools
import csv
import time
import os
import torch
import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)

from models import AtariActorCritic

def evaluate_model(path, episodes=5, max_steps=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym.make("ALE/Venture-v5", render_mode=None, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)
    n_actions = env.action_space.n

    model = AtariActorCritic(n_actions).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except Exception:
        return None

    model.eval()
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=123 + ep)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_steps:
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = logits.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
            steps += 1
        scores.append(total)
    env.close()
    return float(np.mean(scores))

def run_grid():
    # Grid breve
    gen_lrs = [1e-4, 3e-4]
    disc_lrs = [1e-6, 5e-6]
    disc_steps_list = [1, 3]

    combos = list(itertools.product(gen_lrs, disc_lrs, disc_steps_list))

    results_file = os.path.join(sys_path, "hp_results.csv")
    with open(results_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gen_lr", "disc_lr", "disc_steps", "save_prefix", "eval_score"])

        for i, (g_lr, d_lr, d_steps) in enumerate(combos):
            prefix = f"hp_g{int(g_lr*1e6)}_d{int(d_lr*1e9)}_s{d_steps}_{int(time.time())}"
            print(f"Experiment {i+1}/{len(combos)}: gen_lr={g_lr}, disc_lr={d_lr}, disc_steps={d_steps}")

            cmd = [
                sys.executable, "group6/train_gail.py",
                "--episodes", "20",
                "--max-steps", "500",
                "--demos-path", "group6/demos_humano.npz",
                "--bc-model-path", "group6/bc_policy.pt",
                "--gen-lr", str(g_lr),
                "--disc-lr", str(d_lr),
                "--disc-steps", str(d_steps),
                "--entropy-coef", str(0.01),
                "--save-prefix", prefix
            ]

            ret = subprocess.run(cmd, cwd=os.path.dirname(sys_path))
            if ret.returncode != 0:
                print("Training failed for this config, skipping evaluation.")
                writer.writerow([g_lr, d_lr, d_steps, prefix, "FAIL"])
                continue

            model_path = os.path.join(os.getcwd(), f"{prefix}_generator.pt")
            score = evaluate_model(model_path, episodes=5, max_steps=500)
            print(f"Eval score: {score}")
            writer.writerow([g_lr, d_lr, d_steps, prefix, score])

    print(f"Grid search finished. Results saved to {results_file}")

if __name__ == "__main__":
    run_grid()
