"""ST-SAGE: Spatio-Temporal Small-Area Graph Estimator with Near-Repeat
Attention -- the novel algorithm this project develops for India crime
forecasting, synthesized from the reviewed literature:

  * Graph convolution over a real district k-NN adjacency (not a raster CNN)
    as the spatial backbone, since India's forecasting unit is an irregular
    administrative polygon, not a uniform pixel grid
    (cf. "Crime prediction with graph neural networks and multivariate
    normal distributions").
  * ST-ResNet-style multi-branch temporal decomposition -- closeness /
    period / trend -- fused with *learned* per-branch weights, plus a GRU
    over the closeness window for short-range dynamics
    (cf. "Hybrid ST-ResNet and LSTM approach for precise crime hotspot
    prediction").
  * A learned near-repeat / self-excitation gate operating over the graph
    (own last month + graph-neighbour last month), replacing the
    fixed-formula excitation used to generate the synthetic data with a
    trainable analogue motivated by the near-repeat / modus-operandi
    literature ("All Burglaries Are Not the Same...").
  * A Small-Area-Estimation-smoothed rate feature (see sae_smoothing.py) fed
    in as a covariate, and an explicit population-exposure *offset* in the
    output layer, so the network is a graph/temporal generalisation of
    Poisson/Negative-Binomial rate regression rather than an ad-hoc count
    regressor (cf. "Crime Against Women in India: District-Level Risk
    Estimation Using Small Area Estimation").
  * A Negative-Binomial likelihood (learned per-category dispersion) instead
    of plain Poisson, since real crime counts are over-dispersed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Minimal GCN layer: H' = act(A_norm @ H @ W + b)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F_in), adj_norm: (N, N)
        agg = torch.einsum("nm,bmf->bnf", adj_norm, x)
        return F.relu(self.linear(agg))


class STSage(nn.Module):
    def __init__(self, n_categories: int, hidden_dim: int = 32,
                 global_log_rate: torch.Tensor | None = None, max_log_delta: float = 4.0):
        super().__init__()
        self.n_categories = n_categories
        self.hidden_dim = hidden_dim
        self.max_log_delta = max_log_delta
        if global_log_rate is None:
            global_log_rate = torch.zeros(n_categories)
        self.register_buffer("global_log_rate", torch.as_tensor(global_log_rate, dtype=torch.float32))

        # Shared spatial encoder applied to every lag snapshot (raw counts
        # for that snapshot + the SAE-smoothed rate broadcast in as extra
        # channels for closeness only, where it is most informative).
        self.gcn_raw = GraphConv(n_categories, hidden_dim)
        self.gcn_sae = GraphConv(n_categories, hidden_dim)

        self.closeness_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Learned per-branch, per-channel fusion weights (ST-ResNet style).
        self.branch_weights = nn.Parameter(torch.ones(3, hidden_dim))

        # Near-repeat / self-excitation gate: own + neighbour-average last
        # month raw counts -> a sigmoid gate modulating the fused embedding.
        self.near_repeat_gate = nn.Sequential(
            nn.Linear(2 * n_categories, hidden_dim),
            nn.Sigmoid(),
        )

        self.month_embed = nn.Linear(12, 8)
        self.head_input_dim = hidden_dim + 8 + 1 + 1  # fused + month + festive + log_pop_scale
        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_categories),
        )
        # Per-category learned log-dispersion for the Negative-Binomial likelihood.
        self.log_dispersion = nn.Parameter(torch.zeros(n_categories))

    def _encode_branch(self, snapshots: torch.Tensor, adj_norm: torch.Tensor, use_gru: bool) -> torch.Tensor:
        # snapshots: (B, L, N, C)
        b, l, n, c = snapshots.shape
        flat = snapshots.reshape(b * l, n, c)
        encoded = self.gcn_raw(flat, adj_norm).reshape(b, l, n, -1)  # (B, L, N, H)
        if use_gru:
            h = encoded.permute(0, 2, 1, 3).reshape(b * n, l, -1)  # (B*N, L, H)
            _, h_n = self.closeness_gru(h)
            return h_n.squeeze(0).reshape(b, n, -1)
        return encoded.mean(dim=1)  # simple temporal pooling for period/trend branches

    def forward(self, batch: dict, adj_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        closeness_repr = self._encode_branch(batch["closeness"], adj_norm, use_gru=True)
        period_repr = self._encode_branch(batch["period"], adj_norm, use_gru=False)
        trend_repr = self._encode_branch(batch["trend"], adj_norm, use_gru=False)

        sae_repr = self.gcn_sae(batch["sae_feat"], adj_norm)  # (B, N, H)

        fused = (
            self.branch_weights[0] * closeness_repr
            + self.branch_weights[1] * period_repr
            + self.branch_weights[2] * trend_repr
            + sae_repr
        )

        neighbor_prev = torch.einsum("nm,bmc->bnc", adj_norm, batch["prev_month"])
        gate_input = torch.cat([batch["prev_month"], neighbor_prev], dim=-1)  # (B, N, 2C)
        gate = self.near_repeat_gate(gate_input)  # (B, N, H)
        fused = fused * (1.0 + gate)  # excitation multiplies, never suppresses below baseline

        month_emb = self.month_embed(batch["month_onehot"])  # (B, 8)
        n = fused.shape[1]
        month_emb = month_emb.unsqueeze(1).expand(-1, n, -1)
        festive = batch["festive"].view(-1, 1, 1).expand(-1, n, 1)
        log_pop_scale = (batch["log_exposure"] / 10.0).unsqueeze(-1)  # scaled for stability

        head_in = torch.cat([fused, month_emb, festive, log_pop_scale], dim=-1)
        raw_delta = self.head(head_in)  # (B, N, C), unbounded

        # Bound the learned log-rate around the empirical global log-rate per
        # category (tanh-clamped delta) so an under-trained network -- or a
        # rarely-seen, high-exposure district under regime shift at the
        # extrapolated end of the series -- cannot produce runaway counts.
        log_rate_multiplier = self.global_log_rate + self.max_log_delta * torch.tanh(raw_delta)

        # Exposure offset: predicted mean count = exp(log_rate_multiplier) * exposure.
        log_exposure = batch["log_exposure"].unsqueeze(-1)  # (B, N, 1)
        log_mu = log_rate_multiplier + log_exposure
        mu = torch.exp(torch.clamp(log_mu, max=15.0))
        return mu, self.log_dispersion


def negative_binomial_nll(mu: torch.Tensor, log_dispersion: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """NB2 parameterization: Var = mu + mu^2 / r, r = exp(log_dispersion) (per category)."""
    r = torch.exp(log_dispersion).clamp(min=1e-3, max=1e4)  # (C,)
    r = r.view(*([1] * (mu.dim() - 1)), -1)
    eps = 1e-8
    log_prob = (
        torch.lgamma(target + r)
        - torch.lgamma(r)
        - torch.lgamma(target + 1)
        + r * torch.log(r / (r + mu + eps))
        + target * torch.log(mu / (r + mu + eps) + eps)
    )
    return -log_prob.mean()
