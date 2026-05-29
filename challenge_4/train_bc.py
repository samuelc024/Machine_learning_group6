import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from gymnasium import error as gym_error

# Importamos el "cerebro" desde el paquete que creamos en src/
import os
import sys

# Ensure local src/ is on sys.path so this script can import local modules
# when opened/run directly.
repo_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the model class. Try normal import first; if that fails, load the
# module directly from src/models.py as a robust fallback.
try:
    from models import AtariActorCritic  # type: ignore[import-not-found]
except Exception:
    # Fallback: try to load the module file directly from src/models.py
    import importlib.util
    module_path = os.path.join(src_path, "models.py")
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location("models", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        AtariActorCritic = getattr(module, "AtariActorCritic")
    else:
        raise ImportError(
            "Could not import 'models'. Make sure 'models.py' exists.")

def make_env(env_id="ALE/Venture-v5", seed=0):
    try:
        import ale_py
        gym.register_envs(ale_py)
    except ImportError as exc:
        raise ImportError(
            "No se pudo importar 'ale-py'. Instala dependencias Atari con: "
            "pip install gymnasium[atari,accept-rom-license] ale-py"
        ) from exc


    try:
        env = gym.make(env_id, render_mode=None, frameskip=1)
    except gym_error.NamespaceNotFound as exc:
        raise RuntimeError(
            f"No se encontro el namespace del entorno para '{env_id}'. "
            "Verifica que ALE este registrado y que ale-py este instalado."
        ) from exc
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, screen_size=84, scale_obs=False)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env

def train_bc(env_id: str, demos_path: str = "demos_humano.npz",
             n_epochs: int = 20, batch_size: int = 64,
             lr: float = 1e-4, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    
    print(f"Iniciando Clonación de Comportamiento (BC) en dispositivo: {device}")
    
    # 1. Cargar las demostraciones humanas
    print(f"Cargando demostraciones desde {demos_path}...")
    data = np.load(demos_path)
    obs_t = torch.tensor(data["observations"], dtype=torch.float32)
    act_t = torch.tensor(data["actions"], dtype=torch.long)
    
    # Preparamos los datos para PyTorch (lotes mezclados aleatoriamente)
    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Configurar entorno para leer el espacio de acciones
    env = make_env(env_id)
    n_actions = env.action_space.n
    env.close()
    
    # Inicializamos la Política y el Optimizador
    model = AtariActorCritic(n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Usamos Entropía Cruzada porque predecir acciones discretas es un problema de clasificación
    criterion = nn.CrossEntropyLoss()
    
    # 3. Bucle de Entrenamiento Supervisado
    print("Comenzando el entrenamiento supervisado...")
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for obs_b, act_b in loader:
            # Enviamos los datos a la tarjeta gráfica si está disponible
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            
            # Pasamos las imágenes por la CNN
            logits, _ = model(obs_b)
            
            # Calculamos el error comparando la decisión de la red con tu acción real
            loss = criterion(logits, act_b)
            
            # Actualizamos los pesos (Retropropagación)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"BC epoch {epoch+1}/{n_epochs} | loss={avg_loss:.4f}")
        
    # Guardamos el cerebro entrenado
    torch.save(model.state_dict(), "bc_policy.pt")
    print("\n¡Modelo guardado exitosamente como 'bc_policy.pt'!")
    return model

if __name__ == "__main__":
    train_bc(env_id="ALE/Venture-v5", demos_path="demos_humano.npz", n_epochs=20)