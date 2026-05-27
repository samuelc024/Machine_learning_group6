import time

import ale_py
import gymnasium as gym
import numpy as np
import pygame
for i in range(1):
    def grabar_partida_humana(max_steps=2000, save_path="micro_demos/demos_humano.npz"):
        # Iniciamos el entorno con los mismos wrappers de la red neuronal
        env = gym.make("ALE/Venture-v5", render_mode="human", frameskip=1)
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84, scale_obs=False)
        env = gym.wrappers.FrameStackObservation(env, 4)

        obs, _ = env.reset(seed=42)
        pygame.init()
        
        obs_list = []
        act_list = []
        
        print("="*50)
        print("🎮 ¡PREPÁRATE PARA JUGAR VENTURE! 🎮")
        print("Controles: Flechas direccionales para moverte, ESPACIO para disparar.")
        print("Objetivo: Entra a las mazmorras, sé agresivo y busca tesoros.")
        print("Para salir antes de tiempo: Presiona ESCAPE.")
        print("="*50)
        time.sleep(4) # Tiempo para que acomodes las manos
        
        clock = pygame.time.Clock()
        pasos_actuales = 0
        
        while pasos_actuales < max_steps:
            action = 0 # NOOP (Quedarse quieto) por defecto
            
            # Capturamos el teclado
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            
            if keys[pygame.K_ESCAPE]:
                print("\nGrabación interrumpida por el usuario.")
                break
            
            # Mapeo de acciones estándar de Venture
            if keys[pygame.K_SPACE]: action = 1     # FIRE
            elif keys[pygame.K_UP]: action = 2      # UP
            elif keys[pygame.K_RIGHT]: action = 3   # RIGHT
            elif keys[pygame.K_LEFT]: action = 4    # LEFT
            elif keys[pygame.K_DOWN]: action = 5    # DOWN
            
            # Guardamos lo que ve el agente y la tecla que presionaste
            obs_list.append(np.array(obs))
            act_list.append(action)
            
            # Avanzamos el juego
            obs, reward, terminated, truncated, _ = env.step(action)
            pasos_actuales += 1
            
            # Controlamos los FPS. Como hay un frameskip de 4, 15 FPS equivale a jugar a 60 FPS reales
            clock.tick(15) 
            
            if terminated or truncated:
                print("¡Moriste o ganaste! Reiniciando la sala...")
                obs, _ = env.reset()
                time.sleep(1)
                
            if pasos_actuales % 500 == 0:
                print(f"Grabando... {pasos_actuales}/{max_steps} pasos.")
                
        env.close()
        pygame.quit()
        
        print(f"\nEmpaquetando {len(obs_list)} pasos de comportamiento puramente agresivo...")
        np.savez_compressed(save_path, 
                            observations=np.array(obs_list, dtype=np.uint8), 
                            actions=np.array(act_list, dtype=np.int64))
        print(f"Experto humano guardado en: {save_path}")

    if __name__ == "__main__":
        # 5000 pasos a esta velocidad son aproximadamente 5-6 minutos de juego concentrado
        grabar_partida_humana(max_steps=2000)