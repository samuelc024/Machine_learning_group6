import numpy as np
import matplotlib.pyplot as plt

def reproducir_video(ruta_archivo="demos_humano.npz", fps=30, max_frames=1000):
    print(f"Cargando la cinta de video desde: {ruta_archivo}...")
    datos = np.load(ruta_archivo)
    observaciones = datos['observations']
    
    total_frames = len(observaciones)
    print(f"¡Cinta cargada! Total de fotogramas: {total_frames}")
    
    frames_a_reproducir = min(max_frames, total_frames)
    tiempo_pausa = 1.0 / fps
    
    # Configuramos la ventana de reproducción
    plt.ion()  # Activar modo interactivo para animaciones
    fig, ax = plt.subplots(figsize=(6, 8))
    fig.canvas.manager.set_window_title('Reproductor de Demostraciones - Venture')
    
    # Mostramos el primer fotograma
    pantalla = ax.imshow(observaciones[0][-1], cmap='gray')
    ax.axis('off') # Ocultar los ejes numéricos
    titulo = ax.set_title("Iniciando partida...")
    
    print(f"Reproduciendo los primeros {frames_a_reproducir} frames a {fps} FPS...")
    
    # Bucle de reproducción
    for i in range(frames_a_reproducir):
        pantalla.set_data(observaciones[i][-1])
        titulo.set_text(f"Experto de DeepMind - Fotograma: {i}/{frames_a_reproducir}")
        
        # Pausa para simular los FPS del video
        plt.pause(tiempo_pausa)
        
        # Si cierras la ventana antes de que termine, evitamos que el programa colapse
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    print("Reproducción finalizada.")

if __name__ == "__main__":
    # Reproducimos los primeros 1500 pasos (unos 50 segundos de juego)
    reproducir_video(max_frames=21500)