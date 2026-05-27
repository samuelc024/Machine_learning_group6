import gymnasium as gym
import ale_py
import torch
import numpy as np
import time
import os
import sys

# Registramos los entornos para Gymnasium 1.0+
gym.register_envs(ale_py)

# Importamos tu arquitectura
try:
    from models import AtariActorCritic
except ModuleNotFoundError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from models import AtariActorCritic

def evaluar_agente_bc(model_path="bc_policy.pt", episodios=3):
    print(f"Cargando el cerebro del agente desde: {model_path}...")
    
    # 1. Preparamos el entorno EXACTAMENTE como en la grabación
    env = gym.make("ALE/Venture-v5", render_mode="human", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)
    
    n_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Cargamos el modelo
    model = AtariActorCritic(n_actions).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró '{model_path}'. Asegúrate de que train_bc.py haya terminado de ejecutarse.")
        return
        
    model.eval() 
    
    print("="*50)
    print("🤖 ¡EL CLONADOR DE COMPORTAMIENTO TOMA EL CONTROL! 🤖")
    print("="*50)
    time.sleep(2)
    
    puntajes = []
    
    # 3. Bucle de juego
    for ep in range(episodios):
        obs, _ = env.reset(seed=42 + ep) # Cambiamos un poco la semilla por episodio
        terminado = False
        recompensa_total = 0.0
        
        while not terminado:
            # Convertimos la imagen a tensor para PyTorch
            obs_array = np.array(obs)
            obs_t = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)
            
            # El agente piensa su siguiente movimiento
            with torch.no_grad():
                logits, _ = model(obs_t)
                # Tomamos la acción con mayor probabilidad (Greedy)
                action = logits.argmax(dim=-1).item()
                
            # El agente ejecuta la acción
            obs, reward, terminated, truncated, _ = env.step(action)
            recompensa_total += reward
            terminado = terminated or truncated
            
        
            time.sleep(0.02) 
            
        print(f"🎬 Episodio {ep + 1} finalizado | Puntuación obtenida: {recompensa_total}")
        puntajes.append(recompensa_total)
        time.sleep(1)
        
    env.close()
    
    media = np.mean(puntajes)
    std = np.std(puntajes)
    print("="*50)
    print(f"📊 RENDIMIENTO FINAL DEL CLON (BC) 📊")
    print(f"Media ± Std: {media:.1f} ± {std:.1f} puntos")
    print("Anota estos resultados para la sección de 'Baseline' en tu reporte IEEE.")
    print("="*50)

if __name__ == "__main__":
    evaluar_agente_bc(episodios=3)