from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class RBMHistory:
    reconstruction_error: list


class LogSumSparseRBM:
    """Gaussian/Bernoulli visible to Bernoulli hidden RBM.

    The update contains CD-k, L2 decay, KL hidden response sparsity and
    log-sum connection sparsity. Set the lambdas to zero for a vanilla RBM.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        visible_type: str = "bernoulli",
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        cd_k: int = 1,
        weight_decay: float = 5e-3,
        logsum_lambda: float = 5e-3,
        kl_lambda: float = 1e-4,
        sparsity_target: float = 1e-2,
        logsum_eps: float = 1e-3,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.n_visible = int(n_visible)
        self.n_hidden = int(n_hidden)
        self.visible_type = visible_type
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.cd_k = int(cd_k)
        self.weight_decay = float(weight_decay)
        self.logsum_lambda = float(logsum_lambda)
        self.kl_lambda = float(kl_lambda)
        self.sparsity_target = float(sparsity_target)
        self.logsum_eps = float(logsum_eps)
        self.random_state = int(random_state)
        self.device = resolve_device(device)
        gen = torch.Generator(device="cpu").manual_seed(self.random_state)
        self.W = (0.01 * torch.randn(self.n_visible, self.n_hidden, generator=gen)).to(self.device)
        self.a = torch.zeros(self.n_visible, device=self.device)
        self.b = torch.zeros(self.n_hidden, device=self.device)
        self.history_ = RBMHistory(reconstruction_error=[])

    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.clamp(x, -30.0, 30.0))

    def hidden_prob(self, v: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(v @ self.W + self.b)

    def visible_prob(self, h: torch.Tensor) -> torch.Tensor:
        logits = h @ self.W.t() + self.a
        if self.visible_type == "gaussian":
            return logits
        return self.sigmoid(logits)

    def _sample_h(self, v: torch.Tensor):
        prob = self.hidden_prob(v)
        return prob, torch.bernoulli(prob)

    def _sample_v(self, h: torch.Tensor):
        prob = self.visible_prob(h)
        if self.visible_type == "gaussian":
            return prob, prob + 0.01 * torch.randn_like(prob)
        return prob, torch.bernoulli(prob)

    def fit(self, X: np.ndarray):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        for _epoch in range(self.epochs):
            errors = []
            for (v0_cpu,) in loader:
                v0 = v0_cpu.to(self.device)
                ph0, h = self._sample_h(v0)
                vk = v0
                hk = h
                pvk = vk
                phk = ph0
                for _ in range(self.cd_k):
                    pvk, vk = self._sample_v(hk)
                    phk, hk = self._sample_h(vk)

                positive = v0.t() @ ph0 / v0.shape[0]
                negative = vk.t() @ phk / v0.shape[0]
                grad_W = positive - negative
                grad_a = (v0 - vk).mean(dim=0)
                grad_b = (ph0 - phk).mean(dim=0)

                if self.weight_decay:
                    grad_W -= self.weight_decay * self.W
                if self.logsum_lambda:
                    grad_W -= self.logsum_lambda * torch.sign(self.W) / (self.logsum_eps + torch.abs(self.W))
                if self.kl_lambda:
                    p_hat = ph0.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
                    rho = self.sparsity_target
                    kl_base = (-rho / p_hat + (1.0 - rho) / (1.0 - p_hat))
                    kl_grad_b = kl_base * (ph0 * (1.0 - ph0)).mean(dim=0)
                    kl_grad_W = v0.t() @ kl_grad_b.expand_as(ph0) / v0.shape[0]
                    grad_W -= self.kl_lambda * kl_grad_W
                    grad_b -= self.kl_lambda * kl_grad_b

                self.W += self.lr * grad_W
                self.a += self.lr * grad_a
                self.b += self.lr * grad_b
                errors.append(torch.mean((v0 - pvk) ** 2).detach().cpu().item())
            self.history_.reconstruction_error.append(float(np.mean(errors)))
        return self

    def transform(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        bs = int(batch_size or self.batch_size)
        outs = []
        with torch.no_grad():
            for start in range(0, len(X), bs):
                v = torch.from_numpy(X[start:start + bs]).to(self.device)
                outs.append(self.hidden_prob(v).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

