import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import ale_py
import time
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import math

gym.register_envs(ale_py)

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importamos ambas redes de tu librería
try:
    from venture_agent.models import AtariActorCritic, GAILDiscriminator  # type: ignore[import-not-found]
except ImportError:
    from models import AtariActorCritic, GAILDiscriminator

def make_env(env_id="ALE/Venture-v5"):
    env = gym.make(env_id, render_mode=None, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def compute_gail_reward(discriminator, obs_tensor):
    """Calcula la recompensa de engaño: r = -log(1 - D(s))"""
    with torch.no_grad():
        # El discriminador devuelve un valor entre 0 (IA) y 1 (Humano)
        prob_humano = discriminator(obs_tensor)
        # Añadimos un pequeño epsilon (1e-8) para evitar log(0) que haría explotar la red
        reward = -torch.log(1.0 - prob_humano + 1e-8)
    return float(reward.squeeze().cpu().item())

def train_gail(env_id="ALE/Venture-v5", demos_path="demos_humano.npz", bc_model_path="bc_policy.pt", episodes=10000, max_steps_per_ep=2000,
               gen_lr=3e-4, disc_lr=1e-6, disc_steps=3, entropy_coef=0.005, expert_batch=256, save_prefix="gail_long12"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando entrenamiento GAIL en: {device}")

    # Resolver rutas relativas respecto al directorio del script para evitar errores
    script_dir = Path(__file__).resolve().parent
    demos_path = Path(demos_path)
    # Prioridad: 1) path tal cual (absolute or relative to cwd), 2) script_dir / given, 3) script_dir / name(given)
    if not demos_path.exists():
        candidate = script_dir / demos_path
        if candidate.exists():
            demos_path = candidate
        else:
            candidate2 = script_dir / demos_path.name
            if candidate2.exists():
                demos_path = candidate2

    bc_model_path = Path(bc_model_path)
    if not bc_model_path.exists():
        candidate = script_dir / bc_model_path
        if candidate.exists():
            bc_model_path = candidate
        else:
            candidate2 = script_dir / bc_model_path.name
            if candidate2.exists():
                bc_model_path = candidate2
    # 1. Cargar datos del experto (Tus grabaciones)
    print("Cargando memoria del experto...")
    data = np.load(demos_path)
    expert_obs = torch.tensor(data["observations"], dtype=torch.float32)
    # Creamos un DataLoader para sacar lotes aleatorios del experto
    expert_loader = DataLoader(TensorDataset(expert_obs), batch_size=expert_batch, shuffle=True)
    expert_iter = iter(expert_loader)

    # 2. Inicializar el Entorno y las Redes
    env = make_env(env_id)
    n_actions = env.action_space.n

    generator = AtariActorCritic(n_actions).to(device)
    discriminator = GAILDiscriminator(n_actions=n_actions, use_action=False).to(device)

    #Cargamos los pesos del Behavioral Cloning para darle ventaja inicial al Generador
    try:
        generator.load_state_dict(torch.load(bc_model_path, map_location=device))
        print("✅ Pesos de BC cargados. El falsificador ya sabe moverse un poco.")
    except Exception as e:
        print("⚠️ No se encontró bc_policy.pt. El falsificador empezará desde cero.")

    # Optimizadores (permitimos pasar lr desde CLI)
    opt_gen = optim.Adam(generator.parameters(), lr=gen_lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=disc_lr)
    
    # Función de pérdida para el Discriminador (Clasificación Binaria)
    criterion_disc = nn.BCELoss()

    # 3. Bucle Principal de GAIL
    # Hiperparámetros para GAIL puro y entrenamiento
    gail_clip = 2.0
    disc_steps = disc_steps  # cuántos pasos de discriminador por episodio
    save_every = 50

    # Estadísticas running para normalizar recompensa GAIL (Welford)
    gail_count = 0
    gail_mean = 0.0
    gail_M2 = 0.0

    print("="*50)
    print("⚔️ INICIA EL COMBATE ADVERSARIAL ⚔️")
    print("Presiona Ctrl+C en cualquier momento para detener y guardar seguro.")
    print("="*50)

    pbar = tqdm(range(episodes), desc="Entrenando GAIL", unit="ep")

    try:
        for ep in pbar:
            obs, _ = env.reset(seed=42 + ep)
            terminado = False
            
            agent_obs_buffer = []
            log_probs_buffer = []
            values_buffer = []
            rewards_buffer = []
            # no room-tracking: usamos recompensa de imitación pura

            # A. FASE DE RECOLECCIÓN (El Falsificador juega)
            generator.eval()
            for step in range(max_steps_per_ep):
                obs_array = np.array(obs)
                obs_t = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, value = generator(obs_t)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                
                agent_obs_buffer.append(obs_array)
                log_probs_buffer.append(dist.log_prob(action))
                values_buffer.append(value)
                # 1. Ejecutamos la acción y obtenemos la recompensa nativa del juego (reward)
                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                terminado = terminated or truncated

                # 2. Recompensa de imitación GAIL pura
                gail_reward = compute_gail_reward(discriminator, obs_t)
                g = float(gail_reward)

                # Clip y normalización running (Welford)
                g = float(np.clip(g, -gail_clip, gail_clip))
                if gail_count < 10:
                    gail_scaled = math.tanh(g / 2.0)
                else:
                    gail_std = math.sqrt(gail_M2 / (gail_count)) if gail_count > 0 else 1.0
                    gail_scaled = (g - gail_mean) / (gail_std + 1e-8)
                    gail_scaled = math.tanh(gail_scaled)

                # Actualizamos estadísticas running con Welford
                gail_count += 1
                delta = g - gail_mean
                gail_mean += delta / gail_count
                gail_M2 += delta * (g - gail_mean)

                bono_juego = 0.0
                
                # En Venture, reward > 0 ocurre al recoger tesoros (o matar monstruos)
                if reward > 0:
                    
                    # Le damos un bono gigante para que se obsesione, 
                    # pero dejamos que el juego siga para que busque el siguiente.
                    bono_juego = 20.0 
                      
                # La recompensa final es el Estilo (GAIL) + El Éxito (Atari)
                recompensa_final = float(gail_scaled) + bono_juego
                
                rewards_buffer.append(recompensa_final)
                # ---------------------------------------------
                
                obs = next_obs
                if terminado: break

            # Convertimos las observaciones del agente a tensores
            agent_obs_t = torch.tensor(np.array(agent_obs_buffer), dtype=torch.float32).to(device)
            
            # --- RECOMPENSAS FUTURAS DESCONTADAS ---
            gamma = 0.99
            returns = []
            R = 0
            # Recorremos la partida desde el final hacia el principio
            for r in reversed(rewards_buffer):
                R = r + gamma * R
                returns.append(R)
            returns.reverse()
                
            # 1. Convertimos retornos a tensor (SIN NORMALIZAR) y ajustamos la dimensión
            returns_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(device)
            # -----------------------------------------------------------
            # B. FASE DE ENTRENAMIENTO DEL DISCRIMINADOR (El Detective aprende)
            discriminator.train()
            loss_disc = 0.0
            for dstep in range(disc_steps):
                opt_disc.zero_grad()
                # Obtenemos un lote de imágenes reales del experto
                try:
                    exp_obs_batch = next(expert_iter)[0].to(device)
                except StopIteration:
                    expert_iter = iter(expert_loader)
                    exp_obs_batch = next(expert_iter)[0].to(device)

                # El detective evalúa a ambos
                prob_expert = discriminator(exp_obs_batch)
                prob_agent = discriminator(agent_obs_t.detach()) # Detach para no afectar al generador aún

                # Label smoothing: experto ~0.9, agente ~0.1
                labels_expert = torch.full_like(prob_expert, 0.9, device=prob_expert.device)
                labels_agent = torch.full_like(prob_agent, 0.1, device=prob_agent.device)

                loss_expert = criterion_disc(prob_expert, labels_expert)
                loss_agent = criterion_disc(prob_agent, labels_agent)
                loss_disc = (loss_expert + loss_agent) / 2.0
                loss_disc.backward()
                opt_disc.step()

            # C. FASE DE ENTRENAMIENTO DEL GENERADOR (El Falsificador aprende)
            generator.train()
            opt_gen.zero_grad()
            
            logits, values = generator(agent_obs_t)
            
            # --- CORRECCIÓN MATEMÁTICA DE A2C ---
            # 2. Calculamos la Ventaja y LA NORMALIZAMOS a ella (No a los retornos)
            advantages = returns_t - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            log_probs = torch.stack(log_probs_buffer).unsqueeze(1)
            
            # 3. Pérdidas: El Actor usa ventajas normalizadas, el Crítico usa retornos crudos
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = (returns_t - values).pow(2).mean()
            
            # --- BONO DE ENTROPÍA ---
            # Calculamos qué tan "impredecible" está siendo el agente
            probs = torch.softmax(logits, dim=-1)
            dist_entropy = torch.distributions.Categorical(probs).entropy().mean()
            
            # Le restamos la entropía al error total.
            # Esto engaña al optimizador haciéndole creer que explorar es algo "bueno".
            coeficiente_entropia = entropy_coef
            loss_gen = actor_loss + 0.5 * critic_loss - coeficiente_entropia * dist_entropy
            # --------------------------------------------

            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
            opt_gen.step()

            # D. REPORTES EN CONSOLA (actualizar barra de progreso)
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(rewards_buffer) if len(rewards_buffer) > 0 else 0.0
                # sin seguimiento de habitaciones: coverage/avg_subs no aplican
                coverage = 0
                avg_subs = 0.0
                gail_std = math.sqrt(gail_M2 / gail_count) if gail_count > 0 else 0.0
                status = (f"Pasos: {len(agent_obs_buffer):03d} | Disc: {loss_disc.item():.4f} | Gen: {loss_gen.item():.4f} | "
                          f"GAILR: {avg_reward:.4f} | Coverage: {coverage} | AvgSubs: {avg_subs:.1f} | Gmean: {gail_mean:.4f} | Gstd: {gail_std:.4f}")
                pbar.set_postfix_str(status)
                tqdm.write(f"Episodio {ep+1:03d} | {status}")

            if (ep + 1) % save_every == 0:
                torch.save(generator.state_dict(), f"{save_prefix}_generator.pt")
                torch.save(discriminator.state_dict(), f"{save_prefix}_discriminator.pt")
                tqdm.write(f"Checkpoint guardado en episodio {ep+1:03d}")

    # Guardamos a los combatientes
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido manualmente por el usuario.")
    finally:
        print("\n💾 Guardando el progreso actual...")
        torch.save(generator.state_dict(), f"{save_prefix}_generator.pt")
        torch.save(discriminator.state_dict(), f"{save_prefix}_discriminator.pt")
        env.close()
        print("✅ ¡Modelos GAIL guardados a salvo!")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="ALE/Venture-v5")
    parser.add_argument("--demos-path", default="group6/demos_humano.npz")
    parser.add_argument("--bc-model-path", default="group6/bc_policy.pt")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--gen-lr", type=float, default=3e-4)
    parser.add_argument("--disc-lr", type=float, default=1e-6)
    parser.add_argument("--disc-steps", type=int, default=3)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--expert-batch", type=int, default=256)
    parser.add_argument("--save-prefix", type=str, default="gail_long13")
    args = parser.parse_args()

    train_gail(env_id=args.env_id, demos_path=args.demos_path, bc_model_path=args.bc_model_path,
               episodes=args.episodes, max_steps_per_ep=args.max_steps,
               gen_lr=args.gen_lr, disc_lr=args.disc_lr, disc_steps=args.disc_steps,
               entropy_coef=args.entropy_coef, expert_batch=args.expert_batch, save_prefix=args.save_prefix)