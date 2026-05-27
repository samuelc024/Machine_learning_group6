import torch
import torch.nn as nn

class AtariActorCritic(nn.Module):
    """
    Política del agente (Generador). Toma una pila de 4 frames (84x84)
    y devuelve las probabilidades de cada acción (logits) y el valor del estado.
    """
    def __init__(self, n_actions):
        super().__init__()
        # Extractor de características (CNN estándar de DeepMind)
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        
        # Cabezas de salida
        self.actor = nn.Linear(512, n_actions) # Logits de las acciones
        self.critic = nn.Linear(512, 1)        # Valor V(s)

    def forward(self, x):
        # Es crucial normalizar los píxeles de [0, 255] a [0, 1]
        # Se asume que la entrada viene directamente del entorno Atari
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value


class GAILDiscriminator(nn.Module):
    """
    Discriminador Adversarial. 
    Toma una observación y determina P(experto | s, a).
    Retorna un valor escalar entre (0, 1).
    """
    def __init__(self, n_actions: int, use_action: bool = False):
        super().__init__()
        self.use_action = use_action
        
        # Backbone CNN compartido (arquitectura idéntica a la política para equidad)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        cnn_out = 3136 # 64 canales * 7 * 7
        
        # Si usamos la acción, el tamaño de entrada de la red FC crece
        fc_in = cnn_out + n_actions if use_action else cnn_out
        
        # Cabeza fully-connected del discriminador
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.Tanh(),          # Tanh ayuda a estabilizar gradientes en GANs
            nn.Linear(512, 1),
            nn.Sigmoid()        # Comprime la salida a una probabilidad entre 0 y 1
        )

    def forward(self, obs, actions_onehot=None):
        # Normalización de la observación
        feats = self.cnn(obs / 255.0)
        
        # Inyección opcional del contexto de la acción
        if self.use_action and actions_onehot is not None:
            feats = torch.cat([feats, actions_onehot], dim=-1)
            
        return self.fc(feats).squeeze(-1)