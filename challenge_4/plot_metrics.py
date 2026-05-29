import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_gail_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Error: No se encontró el archivo '{csv_path}'.")
        return

    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    # Configuración visual estilo IEEE
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Gráfica 1: Curva de Aprendizaje ---
    axes[0].plot(df["Episodio"], df["Recompensa_Media"], color="dodgerblue", linewidth=2)
    axes[0].set_title("GAIL Learning Curve (Venture)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Training Episodes", fontsize=12)
    axes[0].set_ylabel("Mean Episode Return", fontsize=12)

    # --- Gráfica 2: Precisión del Discriminador ---
    axes[1].plot(df["Episodio"], df["Precision_Disc"], color="crimson", linewidth=2, alpha=0.8)
    axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label="Equilibrio Ideal (0.5)")
    axes[1].set_title("Discriminator Accuracy Over Time", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Training Episodes", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    
    # Guardar la imagen automáticamente
    output_img = csv_path.replace('.csv', '_plots.png')
    plt.savefig(output_img, dpi=300)
    print(f"✅ Gráficas generadas exitosamente.")
    print(f"💾 Guardadas como: '{output_img}' ")
    
    # Mostrar la gráfica en una ventana emergente
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Por defecto buscará gail_long13_metrics.csv
    parser.add_argument("--csv", type=str, default="gail_long13_metrics.csv", help="Ruta al archivo CSV")
    args = parser.parse_args()
    
    plot_gail_metrics(args.csv)