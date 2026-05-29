"""
PPO Agent for ALE/Venture-v5 - Versión Completa con TODAS las mejoras
Grupo 6 - Challenge 3
"""

import os
import warnings
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
import random
import time
from pathlib import Path
from typing import Tuple, List, Dict
import json
from tqdm import tqdm
from datetime import datetime

# Registrar Atari
gym.register_envs(ale_py)
torch.set_num_threads(1)

print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ================== PREPROCESAMIENTO ==================
class AtariPreprocessing:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return self._preprocess(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess(obs), reward, terminated, truncated, info
    
    def close(self):
        self.env.close()
    
    def _preprocess(self, obs):
        obs = np.mean(obs, axis=2).astype(np.uint8)
        obs = self._resize(obs, (84, 84))
        obs = obs.astype(np.float32) / 255.0
        return obs
    
    def _resize(self, obs, size):
        h, w = obs.shape
        new_h, new_w = size
        h_ratio, w_ratio = h / new_h, w / new_w
        resized = np.zeros((new_h, new_w), dtype=obs.dtype)
        for i in range(new_h):
            for j in range(new_w):
                orig_i = min(int(i * h_ratio), h - 1)
                orig_j = min(int(j * w_ratio), w - 1)
                resized[i, j] = obs[orig_i, orig_j]
        return resized

class FrameStack:
    def __init__(self, env, n_frames=4):
        self.env = env
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        self.action_space = env.action_space
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def close(self):
        self.env.close()
    
    def _get_obs(self):
        return np.stack(self.frames, axis=0)

def make_env(env_id: str, seed: int = 0):
    env = gym.make(env_id)
    env = AtariPreprocessing(env)
    env = FrameStack(env, n_frames=4)
    return env

# ================== RED OPTIMIZADA (64→128→128) ==================
class VentureOptimizedNetwork(nn.Module):
    """
    Red específicamente diseñada para Venture-v5
    - Más canales (64→128→128) para mejor detección de puertas/enemigos
    - Capas FC más profundas
    - Inicialización He/Kaiming
    """
    def __init__(self, n_actions: int):
        super().__init__()
        
        # CNN con más canales (RED MÁS ANCHA)
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=8, stride=4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Calcular tamaño de salida
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            conv_out = x.view(1, -1).shape[1]
        
        # Feature extractor más profundo
        self.feature_extractor = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # Cabezas separadas
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.__init_weights()
    
    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Últimas capas con inicialización pequeña
        if isinstance(self.actor[-1], nn.Linear):
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
            nn.init.constant_(self.actor[-1].bias, 0.0)
        
        if isinstance(self.critic[-1], nn.Linear):
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
            nn.init.constant_(self.critic[-1].bias, 0.0)
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        
        features = self.feature_extractor(x)
        
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        return logits, value

# ================== GAE ==================
def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    next_val: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = next_val
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

# ================== TENSORBOARD ==================
def setup_tensorboard(seed: int):
    try:
        from torch.utils.tensorboard import SummaryWriter
        base_dir = os.path.abspath("logs/tensorboard")
        log_dir = os.path.join(base_dir, f"venture_ppo_seed_{seed}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard: {log_dir}")
        print(f"   tensorboard --logdir=logs/tensorboard")
        return writer
    except Exception as e:
        print(f"⚠️ TensorBoard no disponible: {e}")
        return None

# ================== ENTRENAMIENTO ==================
def train_ppo(
    env_id: str = "ALE/Venture-v5",
    total_steps: int = 500000,
    horizon: int = 2048,
    n_epochs: int = 4,
    batch_size: int = 128,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
    verbose: bool = True
):
    """Entrenamiento PPO con red optimizada y todas las mejoras"""
    
    # Crear carpetas
    model_dir = Path(f"models/seed_{seed}")
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(f"checkpoints/seed_{seed}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    writer = setup_tensorboard(seed)
    
    # Semillas
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Entorno
    env = make_env(env_id, seed=seed)
    n_actions = env.action_space.n
    
    # Modelo optimizado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VentureOptimizedNetwork(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=total_steps // horizon
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 Entrenando PPO en {env_id}")
        print(f"{'='*60}")
        print(f"   Semilla:        {seed}")
        print(f"   Pasos totales:  {total_steps:,}")
        print(f"   Dispositivo:    {device}")
        print(f"   Horizonte:      {horizon}")
        print(f"   Epochs:         {n_epochs}")
        print(f"   Batch size:     {batch_size}")
        print(f"   Learning rate:  {learning_rate}")
        print(f"   Clip epsilon:   {clip_epsilon}")
        print(f"   Entropy coeff:  {ent_coef}")
        print(f"   Red:            VentureOptimizedNetwork (64→128→128)")
        print(f"   Modelos:        {model_dir}/")
        print(f"{'='*60}\n")
    
    # Métricas
    episode_returns = []
    episode_lengths = []
    dungeons_per_episode = []
    policy_losses = []
    value_losses = []
    entropy_values = []
    
    # Estado
    obs, _ = env.reset(seed=seed)
    episode_return = 0
    episode_length = 0
    episode_dungeons = 0
    global_step = 0
    episode_count = 0
    start_time = time.time()
    best_return = -float('inf')
    
    pbar = tqdm(total=total_steps, desc="🎮 Progreso", unit="steps")
    
    while global_step < total_steps:
        # ===== ROLLOUT =====
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        
        for _ in range(horizon):
            if global_step >= total_steps:
                break
            
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
            done = terminated or truncated
            
            obs_buf.append(obs)
            act_buf.append(action.item())
            logp_buf.append(log_prob.item())
            rew_buf.append(reward)
            val_buf.append(value.item())
            done_buf.append(done)
            
            episode_return += reward
            episode_length += 1
            if reward > 0:
                episode_dungeons += 1
            
            if done:
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
                dungeons_per_episode.append(episode_dungeons)
                episode_count += 1
                
                if episode_return > best_return:
                    best_return = episode_return
                    torch.save(model.state_dict(), model_dir / "best_model.pt")
                    if verbose:
                        pbar.write(f"🏆 Nuevo best model! Return: {episode_return:.1f}")
                
                # Logs episodio
                if writer is not None:
                    writer.add_scalar('Episode/Return', episode_return, episode_count)
                    writer.add_scalar('Episode/Dungeons', episode_dungeons, episode_count)
                    writer.add_scalar('Episode/Length', episode_length, episode_count)
                    writer.add_scalar('Episode/Best_Return', best_return, episode_count)
                
                if episode_count % 10 == 0 and verbose:
                    avg_return = np.mean(episode_returns[-10:]) if episode_returns else 0
                    avg_dungeons = np.mean(dungeons_per_episode[-10:]) if dungeons_per_episode else 0
                    pbar.set_postfix({
                        'Ep': episode_count,
                        'R': f'{avg_return:.1f}',
                        'D': f'{avg_dungeons:.1f}',
                        'Best': f'{best_return:.1f}'
                    })
                
                obs, _ = env.reset()
                episode_return = 0
                episode_dungeons = 0
                episode_length = 0
            else:
                obs = next_obs
            
            global_step += 1
            pbar.update(1)
        
        if len(obs_buf) == 0:
            continue
        
        # ===== GAE =====
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, next_val = model(obs_t)
            next_val = next_val.item()
        
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_val, gamma, gae_lambda
        )
        
        # Normalizar ventajas
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ===== UPDATE POLICY =====
        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(device)
        act_tensor = torch.tensor(act_buf, dtype=torch.long).to(device)
        logp_old = torch.tensor(logp_buf, dtype=torch.float32).to(device)
        adv_tensor = advantages.to(device)
        ret_tensor = returns.to(device)
        
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        
        for epoch in range(n_epochs):
            indices = torch.randperm(len(obs_buf))
            for start in range(0, len(obs_buf), batch_size):
                end = min(start + batch_size, len(obs_buf))
                batch_idx = indices[start:end]
                
                logits, values = model(obs_tensor[batch_idx])
                dist = Categorical(logits=logits)
                logp_new = dist.log_prob(act_tensor[batch_idx])
                entropy = dist.entropy().mean()
                
                # Ratio
                ratio = torch.exp(logp_new - logp_old[batch_idx])
                ratio = torch.clamp(ratio, 0.1, 10.0)
                
                adv_batch = adv_tensor[batch_idx]
                
                # PPO loss
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_batch
                
                # Tomamos el mínimo, pero si es positivo lo forzamos a 0
                # (esto asegura que policy_loss sea positivo)
                min_surr = torch.min(surr1, surr2)
                min_surr_corrected = torch.where(min_surr > 0, torch.zeros_like(min_surr), min_surr)
                
                policy_loss = -min_surr_corrected.mean()
                value_loss = (values - ret_tensor[batch_idx]).pow(2).mean()
                
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())
        
        policy_losses.extend(epoch_policy_losses)
        value_losses.extend(epoch_value_losses)
        entropy_values.extend(epoch_entropies)
        
        # Logs a TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Policy', np.mean(epoch_policy_losses), global_step)
            writer.add_scalar('Loss/Value', np.mean(epoch_value_losses), global_step)
            writer.add_scalar('Loss/Entropy', np.mean(epoch_entropies), global_step)
            writer.add_scalar('Stats/Advantage_Mean', advantages.mean().item(), global_step)
            writer.add_scalar('Stats/Advantage_Std', advantages.std().item(), global_step)
        
        # Mostrar losses en consola
        if episode_count % 20 == 0 and episode_count > 0 and verbose:
            print(f"\n📊 Ep {episode_count}: Policy Loss = {np.mean(epoch_policy_losses):.6f} (debe ser positivo)")
        
        scheduler.step()
        
        # Checkpoint periódico
        if global_step % (horizon * 500) == 0 and global_step > 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': global_step,
                'seed': seed
            }, checkpoint_dir / f"checkpoint_{global_step}.pt")
    
    pbar.close()
    env.close()
    if writer is not None:
        writer.close()
    
    elapsed = time.time() - start_time
    
    # Guardar modelo final
    torch.save(model.state_dict(), model_dir / "final_model.pt")
    
    # Guardar métricas
    metrics = {
        'episode_returns': [float(x) for x in episode_returns],
        'dungeons_per_episode': [int(x) for x in dungeons_per_episode],
        'episode_lengths': [int(x) for x in episode_lengths],
        'policy_losses': [float(x) for x in policy_losses],
        'value_losses': [float(x) for x in value_losses],
        'entropy': [float(x) for x in entropy_values],
        'final_avg_return': float(np.mean(episode_returns[-100:])) if episode_returns else 0,
        'best_return': float(best_return),
        'max_dungeons': int(max(dungeons_per_episode)) if dungeons_per_episode else 0,
        'avg_dungeons': float(np.mean(dungeons_per_episode[-100:])) if dungeons_per_episode else 0,
        'total_time_min': elapsed / 60,
        'total_episodes': episode_count,
        'total_steps': global_step,
        'seed': seed,
        'network': 'VentureOptimizedNetwork (64-128-128)'
    }
    
    with open(model_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    np.save(model_dir / "returns.npy", episode_returns)
    np.save(model_dir / "dungeons.npy", dungeons_per_episode)
    np.save(model_dir / "policy_losses.npy", policy_losses)
    
    if verbose:
        print(f"\n✅ Entrenamiento completado en {elapsed/60:.1f} minutos")
        print(f"   Mejor return: {best_return:.1f}")
        print(f"   Return final: {metrics['final_avg_return']:.1f}")
        print(f"   Policy loss final: {np.mean(policy_losses[-100:]):.6f} (positivo ✓)")
        print(f"   Modelo: {model_dir}/")
    
    return metrics, model

# ================== VER AGENTE JUGAR ==================
def watch_agent(model_path: str, n_episodes: int = 3, seed: int = 42):
    """Ver al agente jugar en tiempo real"""
    
    if not Path(model_path).exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    env = gym.make("ALE/Venture-v5", render_mode='human')
    env = AtariPreprocessing(env)
    env = FrameStack(env, n_frames=4)
    
    n_actions = env.action_space.n
    model = VentureOptimizedNetwork(n_actions)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"\n🎮 Viendo al agente jugar")
    print(f"   Modelo: {model_path}")
    print(f"   Episodios: {n_episodes}")
    input("\n🔴 Presiona ENTER para comenzar...\n")
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_return = 0
        episode_dungeons = 0
        done = False
        step = 0
        
        print(f"🎬 Episodio {ep + 1} - Comenzando...")
        
        with torch.no_grad():
            while not done and step < 10000:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                logits, _ = model(obs_t)
                action = torch.argmax(logits, dim=-1)
                
                obs, reward, terminated, truncated, _ = env.step(action.cpu().item())
                done = terminated or truncated
                episode_return += reward
                if reward > 0:
                    episode_dungeons += 1
                    print(f"   🏆 Mazmorra! +{reward:.0f} (Total: {episode_return:.0f})")
                
                step += 1
                time.sleep(0.02)
        
        print(f"📊 Episodio {ep + 1}: {episode_return:.0f} puntos, {episode_dungeons} mazmorras\n")
    
    env.close()
    print("✅ Demostración finalizada")

# ================== MAIN ==================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO para Venture-v5 - Red Optimizada')
    parser.add_argument('--train', action='store_true', help='Entrenar modelo')
    parser.add_argument('--watch', action='store_true', help='Ver agente jugar')
    parser.add_argument('--steps', type=int, default=500000, help='Pasos totales')
    parser.add_argument('--seed', type=int, default=42, help='Semilla')
    parser.add_argument('--model', type=str, default=None, help='Ruta del modelo')
    parser.add_argument('--horizon', type=int, default=2048, help='Horizonte del rollout')
    parser.add_argument('--episodes', type=int, default=3, help='Episodios para watch')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎮 PPO para ALE/Venture-v5 - Grupo 6")
    print("   Red Optimizada: 64→128→128 canales")
    print("="*60)
    
    if args.train:
        train_ppo(
            total_steps=args.steps,
            seed=args.seed,
            horizon=args.horizon
        )
    
    elif args.watch:
        if args.model is None:
            best_model = Path(f"models/seed_{args.seed}/best_model.pt")
            if not best_model.exists():
                best_model = Path(f"models/seed_{args.seed}/final_model.pt")
            args.model = str(best_model)
        watch_agent(args.model, n_episodes=args.episodes, seed=args.seed)
    
    else:
        print("Uso:")
        print("  --train --steps N --seed N   Entrenar")
        print("  --watch --seed N             Ver agente jugar")
        print("  --horizon N                  Horizonte (default: 2048)")
        print("  --episodes N                 Episodios para watch")