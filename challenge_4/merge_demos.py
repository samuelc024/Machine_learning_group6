import os
import glob
import numpy as np
from pathlib import Path

def concatenar_micro_juegos(carpeta_entrada="micro_demos", archivo_salida="demos_humano.npz"):
    """
    Busca todos los archivos .npz en la carpeta de entrada, extrae las observaciones
    y las unifica en un único archivo maestro para el entrenamiento de GAIL.
    """
    script_dir = Path(__file__).resolve().parent
    ruta_carpeta = script_dir / carpeta_entrada if not Path(carpeta_entrada).is_absolute() else Path(carpeta_entrada)
    ruta_salida = script_dir / archivo_salida if not Path(archivo_salida).is_absolute() else Path(archivo_salida)

    if not ruta_carpeta.exists():
        print(f"❌ ERROR: La carpeta '{carpeta_entrada}' no existe.")
        print(f"Por favor, créala en: {ruta_carpeta}")
        print("Y guarda allí tus archivos de micro-juegos (ej. demo_1.npz, demo_2.npz...).")
        return

    # Buscar todos los archivos .npz en la carpeta
    archivos_npz = sorted(glob.glob(os.path.join(ruta_carpeta, "*.npz")))

    if not archivos_npz:
        print(f"⚠️ No se encontraron archivos .npz dentro de la carpeta '{carpeta_entrada}'.")
        return

    print("=" * 60)
    print(f"🔍 ¡Se encontraron {len(archivos_npz)} micro-juegos listos para fusionar!")
    print("=" * 60)

    lista_observaciones = []
    lista_acciones = []
    total_timesteps = 0

    for i, ruta_archivo in enumerate(archivos_npz, start=1):
        nombre_archivo = os.path.basename(ruta_archivo)
        try:
            data = np.load(ruta_archivo)
            
            if "observations" not in data:
                print(f"⚠️ Archivo omitido: '{nombre_archivo}' no contiene la clave 'observations'.")
                continue
                
            obs = data["observations"]
            lista_observaciones.append(obs)
            pasos = len(obs)
            total_timesteps += pasos
            print(f"🎮 [{i:02d}] {nombre_archivo:<25} | Pasos cargados: {pasos:5d}")
            
            # Si tus grabaciones también incluyen acciones, las guardamos por consistencia futura
            if "actions" in data:
                lista_acciones.append(data["actions"])
                
        except Exception as e:
            print(f"❌ Error leyendo {nombre_archivo}: {str(e)}")

    if not lista_observaciones:
        print("❌ ERROR: No se pudieron extraer observaciones válidas de ningún archivo.")
        return

    # Concatenar todos los arrays a lo largo del eje 0 (el eje del tiempo/pasos)
    obs_maestras = np.concatenate(lista_observaciones, axis=0)
    
    diccionario_salida = {"observations": obs_maestras}
    
    if lista_acciones and len(lista_acciones) == len(lista_observaciones):
        acc_maestras = np.concatenate(lista_acciones, axis=0)
        diccionario_salida["actions"] = acc_maestras

    # Guardar el archivo unificado
    np.savez_compressed(ruta_salida, **diccionario_salida)
    
    print("=" * 60)
    print("🏆 ¡PROCESO FINALIZADO CON ÉXITO! 🏆")
    print(f"📂 Archivo maestro generado: {ruta_salida}")
    print(f"📊 Dimensiones finales de observaciones: {obs_maestras.shape}")
    print(f"⚡ Total de experiencias humanas consolidadas: {total_timesteps} pasos.")
    print("=" * 60)
    print("¡Tu Dataset está listo para inyectarse en train_gail.py!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fusionador de Micro-Juegos para GAIL")
    parser.add_argument("--input-dir", default="micro_demos", help="Carpeta con los archivos .npz individuales")
    parser.add_argument("--output", default="demos_humano.npz", help="Nombre del archivo .npz maestro de salida")
    args = parser.parse_args()

    # Ejecutar la fusión con los parámetros indicados
    concatenar_micro_juegos(carpeta_entrada=args.input_dir, archivo_salida=args.output)