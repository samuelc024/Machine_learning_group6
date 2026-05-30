from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm
from .config import ModelConfig

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
 
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - handled at runtime when torch is unavailable
    torch = None  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]
    TensorDataset = Any  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _ensure_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the unsupervised AE/VAE pipeline. Install torch and rerun."
        )


@dataclass(frozen=True)
class AutoencoderScores:
    reconstruction_mse: np.ndarray
    latent_z: np.ndarray


@dataclass(frozen=True)
class VariationalAutoencoderScores:
    reconstruction_mse: np.ndarray
    kl_divergence: np.ndarray
    elbo_loss: np.ndarray
    latent_mu: np.ndarray
    latent_z: np.ndarray


if TORCH_AVAILABLE:

    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: tuple[int, int], latent_dim: int):
            super().__init__()
            hidden_1, hidden_2 = hidden_dims
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_1),
                nn.ReLU(),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
                nn.Linear(hidden_2, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_2),
                nn.ReLU(),
                nn.Linear(hidden_2, hidden_1),
                nn.ReLU(),
                nn.Linear(hidden_1, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)


    class VariationalAutoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: tuple[int, int], latent_dim: int):
            super().__init__()
            hidden_1, hidden_2 = hidden_dims
            self.encoder_backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_1),
                nn.ReLU(),
                nn.Linear(hidden_1, hidden_2),
                nn.ReLU(),
            )
            self.mu_layer = nn.Linear(hidden_2, latent_dim)
            self.logvar_layer = nn.Linear(hidden_2, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_2),
                nn.ReLU(),
                nn.Linear(hidden_2, hidden_1),
                nn.ReLU(),
                nn.Linear(hidden_1, input_dim),
            )

        def encode(self, x):
            h = self.encoder_backbone(x)
            return self.mu_layer(h), self.logvar_layer(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decoder(z)
            return reconstruction, mu, logvar


def _prepare_tensor_data(features: np.ndarray):
    _ensure_torch()
    matrix = np.asarray(features, dtype=np.float32)
    tensor = torch.from_numpy(matrix)
    return matrix, tensor


def _fit_autoencoder(model, data_tensor, epochs: int, batch_size: int, learning_rate: float) -> None:
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for _ in tqdm(range(epochs), desc="Entrenando AE"): 
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()


def score_autoencoder(features: np.ndarray, config: ModelConfig, *, random_state: int) -> AutoencoderScores:
    _ensure_torch()
    torch.manual_seed(random_state)
    matrix, data_tensor = _prepare_tensor_data(features)
    input_dim = int(matrix.shape[1])

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=config.ae_hidden_dims,
        latent_dim=config.ae_latent_dim,
    )
    _fit_autoencoder(
        model,
        data_tensor,
        epochs=config.ae_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )

    model.eval()
    with torch.no_grad():
        latent_z = model.encoder(data_tensor)
        reconstructed = model(data_tensor)
        mse = torch.mean((reconstructed - data_tensor) ** 2, dim=1)
    return AutoencoderScores(reconstruction_mse=mse.cpu().numpy(), latent_z=latent_z.cpu().numpy())


def score_variational_autoencoder(
    features: np.ndarray,
    config: ModelConfig,
    *,
    random_state: int,
) -> VariationalAutoencoderScores:
    _ensure_torch()
    torch.manual_seed(random_state)
    matrix, data_tensor = _prepare_tensor_data(features)
    input_dim = int(matrix.shape[1])

    model = VariationalAutoencoder(
        input_dim=input_dim,
        hidden_dims=config.ae_hidden_dims,
        latent_dim=config.ae_latent_dim,
    )

    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for _ in tqdm(range(config.vae_epochs), desc="Entrenando VAE"):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            reconstruction = torch.mean((reconstructed - batch) ** 2, dim=1)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            loss = torch.mean(reconstruction + config.vae_beta * kl)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstructed, mu, logvar = model(data_tensor)
        latent_z = mu
        reconstruction = torch.mean((reconstructed - data_tensor) ** 2, dim=1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        elbo = reconstruction + config.vae_beta * kl

    return VariationalAutoencoderScores(
        reconstruction_mse=reconstruction.cpu().numpy(),
        kl_divergence=kl.cpu().numpy(),
        elbo_loss=elbo.cpu().numpy(),
        latent_mu=mu.cpu().numpy(),
        latent_z=latent_z.cpu().numpy(),
    )