import gymnasium as gym

import ale_py

import torch

import numpy as np

import time



# Registramos el emulador

gym.register_envs(ale_py)





import sys

import os



sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



try:

    from models import AtariActorCritic

except Exception as e:

    raise ImportError(

        "No se pudo importar 'AtariActorCritic' desde 'models.py'."

    ) from e



def evaluar_agente_gail(model_path="gail_long13_generator.pt", episodios=5):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Cargando el cerebro definitivo desde: {model_path} en {device}...")

   

    # 1. Entorno idéntico al de grabación, con render_mode="human" para ver el juego

    env = gym.make("ALE/Venture-v5", render_mode="rgb_array", frameskip=1)
    
    # Añadimos el Wrapper de grabación (guardará los videos en la carpeta 'videos_gail')
    env = gym.wrappers.RecordVideo(env, video_folder="videos_gail", name_prefix="evaluacion1154", episode_trigger=lambda x: True)
    
    # Preprocesamiento habitual
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)

    n_actions = env.action_space.n

   

    # 2. Cargamos el modelo

    model = AtariActorCritic(n_actions).to(device)

    try:

        model.load_state_dict(torch.load(model_path, map_location=device))

    except FileNotFoundError:

        print(f"\n❌ ERROR: No se encontró '{model_path}'.")

        return

       

    model.eval()

   

    print("="*50)

    print("🏆 ¡EL AGENTE GAIL ENTRA A LA MAZMORRA! 🏆")

    print("="*50)

    time.sleep(2)

   

    puntajes = []

   

    # 3. Bucle de Evaluación

    for ep in range(episodios):

        obs, _ = env.reset(seed=100 + ep) # Usamos semillas diferentes para probar generalización

        terminado = False

        recompensa_total = 0.0

        pasos = 0

       

        while not terminado:

            obs_array = np.array(obs)

            obs_t = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)

           

            with torch.no_grad():

                logits, _ = model(obs_t)

                # Tomamos la acción con la probabilidad más alta (Greedy)

                action = logits.argmax(dim=-1).item()

               

            obs, reward, terminated, truncated, _ = env.step(action)

            recompensa_total += reward

            pasos += 1

            terminado = terminated or truncated

           

            time.sleep(0.02)

           

        print(f"🎬 Episodio {ep + 1} finalizado | Pasos sobrevivió: {pasos} | Puntuación obtenida: {recompensa_total}")

        puntajes.append(recompensa_total)

        time.sleep(1)

       

    env.close()

   

    media = np.mean(puntajes)

    std = np.std(puntajes)

    print("="*50)

    print(f"📊 RENDIMIENTO FINAL GAIL 📊")

    print(f"Media ± Std: {media:.1f} ± {std:.1f} puntos")

    print("="*50)



if __name__ == "__main__":

    evaluar_agente_gail(episodios=5)