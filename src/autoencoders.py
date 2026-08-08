import gc
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import SEED


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mlp(dims: List[int], final_activation: nn.Module) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.SELU() if i < len(dims) - 2 else final_activation)
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Symmetric autoencoder with a SELU encoder and Sigmoid-terminated decoder."""

    def __init__(self, input_dim: int, encoder_dims: Tuple[int, ...] = (6, 4, 2),
                 l1_lambda: float = 1e-4):
        super().__init__()
        self.l1_lambda = l1_lambda
        enc = [input_dim, *encoder_dims]
        self.encoder = _mlp(enc, nn.SELU())
        dec = [*reversed(enc)]
        self.decoder = _mlp(dec, nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def l1_regularization(self):
        return self.l1_lambda * sum(p.abs().sum() for p in self.parameters())


def train_autoencoder(model, data: np.ndarray, epochs: int, batch_size: int,
                      learning_rate: float, seed: int = SEED, device=None,
                      name: str = "Autoencoder", verbose: bool = True,
                      patience: int = 3, min_delta: float = 1e-5):

    device = device or get_device()
    model = model.to(device)

    g = torch.Generator().manual_seed(seed)  # deterministic shuffling
    tensor = torch.as_tensor(np.ascontiguousarray(data, dtype=np.float32))
    loader = DataLoader(TensorDataset(tensor, tensor), batch_size=batch_size,
                        shuffle=True, generator=g)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = {"loss": [], "accuracy": [], "stopped_early": False, "epochs_run": 0}

    best_loss, bad_epochs = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        run_loss, run_acc, n = 0.0, 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by) + model.l1_regularization()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            run_loss += loss.item()
            with torch.no_grad():
                run_acc += ((out - by).abs() < 0.1).float().mean().item()
            n += 1
        epoch_loss = run_loss / n
        history["loss"].append(epoch_loss)
        history["accuracy"].append(run_acc / n)
        history["epochs_run"] = epoch + 1
        if verbose:
            print(f"[{name}] epoch {epoch + 1}/{epochs}  "
                  f"loss={epoch_loss:.6f}  recon@0.1={history['accuracy'][-1]:.4f}")

        if patience:
            if epoch_loss < best_loss - min_delta:
                best_loss, bad_epochs = epoch_loss, 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    history["stopped_early"] = True
                    if verbose:
                        print(f"[{name}] early stop at epoch {epoch + 1} "
                              f"(no improvement > {min_delta} for {patience} epochs)")
                    break
    return history


def encode_full(model, data: np.ndarray, device=None) -> np.ndarray:
    """Run the encoder over the whole dataset and return latent features."""
    device = device or get_device()
    model = model.to(device).eval()
    with torch.no_grad():
        t = torch.as_tensor(np.ascontiguousarray(data, dtype=np.float32)).to(device)
        return model.encode(t).cpu().numpy()


# ----------------------------------------------------------------- stacked AE
class _StackedAE(nn.Module):
    """One stage of the stacked AE with configurable encoder depth."""

    def __init__(self, input_dim: int, encoder_dims, l1_lambda: float = 1e-4):
        super().__init__()
        self.l1_lambda = l1_lambda
        enc = [input_dim, *encoder_dims]
        self.encoder = _mlp(enc, nn.SELU())
        self.decoder = _mlp([*reversed(enc)], nn.Sigmoid())
        self.encoder_dims = encoder_dims

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def encode_to(self, x, n_layers):
        """Return the output after ``n_layers`` encoder Linear+SELU blocks."""
        for i in range(2 * n_layers):
            x = self.encoder[i](x)
        return x

    def l1_regularization(self):
        return self.l1_lambda * sum(p.abs().sum() for p in self.parameters())


def build_stacked_features(data: np.ndarray, seed: int = SEED, device=None,
                           verbose: bool = True):
    
    device = device or get_device()
    histories = {}

    ae1 = _StackedAE(data.shape[1], (6, 4, 2))
    histories["stack1"] = train_autoencoder(ae1, data, epochs=10, batch_size=64,
                                            learning_rate=1e-4, seed=seed,
                                            device=device, name="AE1", verbose=verbose)
    ae1_recon = _reconstruct(ae1, data, device)
    ae2_in = np.concatenate([ae1_recon, data], axis=1)
    del ae1_recon, ae1                      # ~0.9 GB back
    gc.collect()

    ae2 = _StackedAE(ae2_in.shape[1], (10, 5, 2))
    histories["stack2"] = train_autoencoder(ae2, ae2_in, epochs=5, batch_size=32,
                                            learning_rate=1e-4, seed=seed,
                                            device=device, name="AE2", verbose=verbose)
    ae2_enc5 = _encode_to(ae2, ae2_in, 2, device)   # 5-D layer-2 encoding
    ae3_in = np.concatenate([ae2_enc5, ae2_in], axis=1)
    del ae2_in, ae2_enc5, ae2               # ~1.8 GB back
    gc.collect()

    ae3 = _StackedAE(ae3_in.shape[1], (15, 10, 5, 2))
    histories["stack3"] = train_autoencoder(ae3, ae3_in, epochs=5, batch_size=16,
                                            learning_rate=1e-4, seed=seed,
                                            device=device, name="AE3", verbose=verbose)
    latent = encode_full(ae3, ae3_in, device)       # 2-D final latent
    del ae3_in, ae3
    gc.collect()
    return latent, histories


def _reconstruct(model, data, device):
    model = model.to(device).eval()
    with torch.no_grad():
        t = torch.as_tensor(np.ascontiguousarray(data, dtype=np.float32)).to(device)
        return model(t).cpu().numpy()


def _encode_to(model, data, n_layers, device):
    model = model.to(device).eval()
    with torch.no_grad():
        t = torch.as_tensor(np.ascontiguousarray(data, dtype=np.float32)).to(device)
        return model.encode_to(t, n_layers).cpu().numpy()
