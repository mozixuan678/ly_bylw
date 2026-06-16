from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DCDFVAEConfig:
    n_nodes: int
    hidden_dim: int = 48
    latent_dim: int = 16
    tcn_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.05
    tau_start: float = 1.0
    tau_min: float = 0.35
    tau_decay: float = 0.98
    hard_gumbel: bool = False
    keep_self_edges: bool = True
    prediction_horizon: int = 1
    lambda_rec: float = 1.0
    lambda_kl: float = 1e-4
    lambda_sparse: float = 8e-4
    lambda_temporal: float = 1e-4
    lambda_prior: float = 0.0
    sparse_eps: float = 1e-3
    use_tcn: bool = True
    use_gat: bool = True
    use_gumbel: bool = True
    use_cvb: bool = True
    use_sparse: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNEncoder(nn.Module):
    """Per-node causal dilated temporal encoder.

    Input x: [B, T, N]
    Output h: [B, T, N, H]
    """

    def __init__(self, hidden_dim: int, layers: int, kernel_size: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for layer in range(layers):
            dilation = 2**layer
            blocks.extend(
                [
                    CausalConv1d(in_ch, hidden_dim, kernel_size, dilation),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = hidden_dim
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, nodes = x.shape
        y = x.permute(0, 2, 1).reshape(batch * nodes, 1, time)
        y = self.net(y)
        y = y.reshape(batch, nodes, -1, time).permute(0, 3, 1, 2)
        return self.norm(y)


class PointEncoder(nn.Module):
    """Ablation replacement for TCN: no temporal convolution."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1))


class DirectedGATEdgeScorer(nn.Module):
    """GAT-like directed edge scorer.

    The additive terms match the standard GAT concat score.  A low-rank
    multiplicative term gives the scorer a pair interaction while keeping the
    memory footprint manageable for 264 ROI data.
    """

    def __init__(self, hidden_dim: int, attn_dim: int | None = None):
        super().__init__()
        attn_dim = attn_dim or hidden_dim
        self.proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.src = nn.Parameter(torch.empty(attn_dim))
        self.dst = nn.Parameter(torch.empty(attn_dim))
        self.pair = nn.Parameter(torch.empty(attn_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.src, std=0.03)
        nn.init.normal_(self.dst, std=0.03)
        nn.init.normal_(self.pair, std=0.03)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = self.proj(h)
        src_score = torch.einsum("btnd,d->btn", q, self.src)
        dst_score = torch.einsum("btnd,d->btn", q, self.dst)
        pair_score = torch.einsum("btid,d,btjd->btij", q, self.pair, q)
        logits = src_score.unsqueeze(-1) + dst_score.unsqueeze(-2) + pair_score + self.bias
        return F.leaky_relu(logits, negative_slope=0.2)


class StaticEdgeScorer(nn.Module):
    """Ablation replacement for GAT: one learned graph shared by all windows."""

    def __init__(self, n_nodes: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_nodes, n_nodes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, time, nodes, _ = h.shape
        return self.logits.view(1, 1, nodes, nodes).expand(batch, time, nodes, nodes)


def gumbel_sigmoid(
    logits: torch.Tensor,
    tau: float,
    hard: bool = False,
    training: bool = True,
    use_gumbel: bool = True,
) -> torch.Tensor:
    tau = max(float(tau), 1e-6)
    if training and use_gumbel:
        u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        logistic_noise = torch.log(u) - torch.log1p(-u)
        y = torch.sigmoid((logits + logistic_noise) / tau)
    else:
        y = torch.sigmoid(logits / tau)
    if hard:
        y_hard = (y > 0.5).to(y.dtype)
        y = y_hard.detach() - y.detach() + y
    return y


class DCDFVAE(nn.Module):
    def __init__(self, cfg: DCDFVAEConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_tcn:
            self.encoder = TCNEncoder(cfg.hidden_dim, cfg.tcn_layers, cfg.kernel_size, cfg.dropout)
        else:
            self.encoder = PointEncoder(cfg.hidden_dim, cfg.dropout)
        if cfg.use_gat:
            self.edge_scorer = DirectedGATEdgeScorer(cfg.hidden_dim)
        else:
            self.edge_scorer = StaticEdgeScorer(cfg.n_nodes)
        self.mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.dec_in = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.decoder = nn.Sequential(
            CausalConv1d(cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, dilation=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.hidden_dim, 1, kernel_size=1),
        )

    def encode(self, x: torch.Tensor, tau: float | None = None):
        h = self.encoder(x)
        logits = self.edge_scorer(h)
        if not self.cfg.keep_self_edges:
            eye = torch.eye(self.cfg.n_nodes, device=x.device, dtype=torch.bool)
            logits = logits.masked_fill(eye.view(1, 1, self.cfg.n_nodes, self.cfg.n_nodes), -20.0)
        gamma = gumbel_sigmoid(
            logits,
            tau if tau is not None else self.cfg.tau_min,
            hard=self.cfg.hard_gumbel,
            training=self.training,
            use_gumbel=self.cfg.use_gumbel,
        )
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        return h, logits, gamma, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z_tilde: torch.Tensor) -> torch.Tensor:
        batch, time, nodes, _ = z_tilde.shape
        u = self.dec_in(z_tilde)
        u = u.permute(0, 2, 3, 1).reshape(batch * nodes, self.cfg.hidden_dim, time)
        y = self.decoder(u).reshape(batch, nodes, time).permute(0, 2, 1)
        return y

    def forward(self, x: torch.Tensor, tau: float | None = None) -> dict[str, torch.Tensor]:
        _, logits, gamma, mu, logvar = self.encode(x, tau=tau)
        z = self.reparameterize(mu, logvar)
        if self.cfg.use_cvb:
            z_tilde = torch.einsum("btij,btid->btjd", gamma, z)
        else:
            z_tilde = z
        xhat = self.decode(z_tilde)
        return {
            "xhat": xhat,
            "gamma": gamma,
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_tilde": z_tilde,
        }

    def loss(
        self,
        x: torch.Tensor,
        out: dict[str, torch.Tensor],
        edge_prior: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        horizon = max(0, int(cfg.prediction_horizon))
        xhat, mu, logvar, gamma = out["xhat"], out["mu"], out["logvar"], out["gamma"]
        if horizon > 0:
            pred = xhat[:, :-horizon]
            target = x[:, horizon:]
            mu_l = mu[:, :-horizon]
            logvar_l = logvar[:, :-horizon]
            gamma_l = gamma[:, :-horizon]
        else:
            pred, target = xhat, x
            mu_l, logvar_l, gamma_l = mu, logvar, gamma
        rec = F.mse_loss(pred, target)
        kl = 0.5 * (mu_l.pow(2) + logvar_l.exp() - logvar_l - 1.0).sum(-1).mean()
        if cfg.use_sparse and cfg.lambda_sparse > 0:
            sparse = torch.log1p(gamma_l.abs() / cfg.sparse_eps).mean()
        else:
            sparse = gamma_l.new_zeros(())
        if cfg.lambda_temporal > 0 and gamma_l.shape[1] > 1:
            temporal = (gamma_l[:, 1:] - gamma_l[:, :-1]).pow(2).mean()
        else:
            temporal = gamma_l.new_zeros(())
        if edge_prior is not None and cfg.lambda_prior > 0:
            prior = F.mse_loss(gamma_l.mean(dim=(0, 1)), edge_prior)
        else:
            prior = gamma_l.new_zeros(())
        total = (
            cfg.lambda_rec * rec
            + cfg.lambda_kl * kl
            + cfg.lambda_sparse * sparse
            + cfg.lambda_temporal * temporal
            + cfg.lambda_prior * prior
        )
        return {
            "loss": total,
            "rec": rec.detach(),
            "kl": kl.detach(),
            "sparse": sparse.detach(),
            "temporal": temporal.detach(),
            "prior": prior.detach(),
        }
