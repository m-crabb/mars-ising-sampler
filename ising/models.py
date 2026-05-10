"""Locally equivariant neural rate models.

Both architectures return G(tau, i | x, t), a per-site score for changing site
i to token tau. The readout is locally equivariant: swapping x_i with tau
negates the corresponding score, so positive and negative parts define forward
and reverse CTMC rates with one model evaluation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TimestepEmbedder(nn.Module):
    """Sinusoidal time features followed by a small MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class _AttentionBlock(nn.Module):
    """Pre-norm attention block used inside the causal transformer stacks."""

    def __init__(self, hidden_dim: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        return x + self.ff(self.norm_ff(x))


class _CausalBlock(nn.Module):
    """Causal attention block with a learned position embedding."""

    def __init__(self, hidden_dim: int, n_heads: int, seq_len: int, ff_mult: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(hidden_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, hidden_dim) * 1e-2)
        self.attn_block = _AttentionBlock(hidden_dim, n_heads, ff_mult)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        return self.attn_block(self.proj_in(x) + self.pos_embed.unsqueeze(0), attn_mask) + x


class CausalStack(nn.Module):
    """Inclusive-causal transformer stack for one scan direction."""

    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int, seq_len: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_CausalBlock(hidden_dim, n_heads, seq_len) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        for block in self.blocks:
            x = block(x, mask)
        return x


class AttentionReadout(nn.Module):
    """Fuse forward/backward causal states while keeping each site hollow."""

    def __init__(self, hidden_dim: int, n_heads: int, data_dim: int, ff_mult: int = 4):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.data_dim = data_dim
        self.d_k = hidden_dim // n_heads
        self.pos_embed = nn.Parameter(torch.randn(data_dim, self.d_k) * 1e-2)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )

    def forward(self, fwd_x: Tensor, bwd_x: Tensor, cond_t: Tensor) -> Tensor:
        # Slice trick: output site i sees left context up to i-1 and right
        # context from i+1 onward, but not x_i itself.
        sliced_fwd = fwd_x[:, :-1, :]
        sliced_bwd = bwd_x[:, 1:, :]
        combined = (sliced_fwd + sliced_bwd) / math.sqrt(2.0) + cond_t
        all_keys = torch.cat([sliced_fwd, sliced_bwd], dim=1) + cond_t

        q = self.q_proj(self.norm_in(combined))
        kv = self.norm_in(all_keys)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        batch, n_sites, _ = q.shape

        def split_heads(tensor: Tensor, length: int) -> Tensor:
            return tensor.view(batch, length, self.n_heads, self.d_k).transpose(1, 2)

        q = split_heads(q, n_sites)
        k = split_heads(k, 2 * n_sites)
        v = split_heads(v, 2 * n_sites)

        pos = self.pos_embed.unsqueeze(0).unsqueeze(0)
        q = q + pos
        k = torch.cat([k[:, :, :n_sites, :] + pos, k[:, :, n_sites:, :] + pos], dim=2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        i_idx = torch.arange(n_sites, device=q.device).unsqueeze(1)
        j_idx = torch.arange(n_sites, device=q.device).unsqueeze(0)
        mask_l = j_idx > i_idx
        mask_r = j_idx < i_idx
        scores = scores.masked_fill(torch.cat([mask_l, mask_r], dim=-1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, n_sites, self.hidden_dim)
        h = combined + self.out_proj(out)
        return h + self.ff(self.norm_ff(h))


class LeTFRateMatrix(nn.Module):
    """Hollow locally equivariant transformer rate model."""

    is_locally_equivariant = True

    def __init__(
        self,
        d: int,
        vocab_size: int = 2,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.token_embedder = nn.Embedding(vocab_size, hidden_dim)
        nn.init.kaiming_uniform_(self.token_embedder.weight, a=math.sqrt(5))
        self.time_embedder = TimestepEmbedder(hidden_dim)
        self.fwd_stack = CausalStack(hidden_dim, n_layers, n_heads, 1 + d)
        self.bwd_stack = CausalStack(hidden_dim, n_layers, n_heads, 1 + d)
        self.attention_readout = AttentionReadout(hidden_dim, n_heads, d)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.omega = nn.Embedding(vocab_size, hidden_dim)
        nn.init.kaiming_uniform_(self.omega.weight, a=math.sqrt(5))

    def compute_body(self, x: Tensor, t: Tensor) -> Tensor:
        """Return hollow hidden features H_i(x_{-i}, t) for all sites."""
        x_idx = ((x + 1) / 2).long()
        x_emb = self.token_embedder(x_idx)
        cond_t = self.time_embedder(t).unsqueeze(1)
        fwd_x = self.fwd_stack(torch.cat([cond_t, x_emb], dim=1))
        bwd_x = self.bwd_stack(torch.cat([cond_t, x_emb.flip(1)], dim=1)).flip(1)
        return self.attention_readout(fwd_x, bwd_x, cond_t)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Return G(tau, i | x, t), with the current token slot set to zero."""
        h = self.output_norm(self.compute_body(x, t)) + self.time_embedder(t).unsqueeze(1)
        x_idx = ((x + 1) / 2).long()
        omega_all = self.omega.weight
        omega_xi = self.omega(x_idx)
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]
        g = torch.einsum("bdh,bdsh->bds", h, diff)
        return g.scatter(-1, x_idx.unsqueeze(-1), 0.0)


class LeConvDeepRateMatrix(nn.Module):
    """Deep hollow convolutional rate model, with optional global context."""

    is_locally_equivariant = True

    def __init__(
        self,
        lattice_size: int = 10,
        vocab_size: int = 2,
        kernel_schedule: tuple[int, ...] = (3, 5, 7, 9, 15),
        hidden_dim: int = 64,
        use_global_context: bool = False,
    ):
        super().__init__()
        self.lattice_size = lattice_size
        self.vocab_size = vocab_size
        self.kernel_schedule = tuple(kernel_schedule)
        self.hidden_dim = hidden_dim
        self.use_global_context = use_global_context
        self.token_embedder = nn.Embedding(vocab_size, hidden_dim)
        self.omega = nn.Embedding(vocab_size, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)
        nn.init.kaiming_uniform_(self.token_embedder.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.omega.weight, a=math.sqrt(5))
        self.h_0 = nn.Parameter(torch.zeros(hidden_dim))
        self.A = nn.ModuleList(
            [nn.Conv2d(hidden_dim, k * k, kernel_size=1) for k in self.kernel_schedule]
        )
        if use_global_context:
            self.global_context_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.c = nn.ParameterList([nn.Parameter(torch.zeros(k * k)) for k in self.kernel_schedule])
        for layer_idx, k in enumerate(self.kernel_schedule):
            mask = torch.ones(k, k)
            mask[k // 2, k // 2] = 0.0
            self.register_buffer(f"hollow_mask_{layer_idx}", mask)

    def _leave_one_out_global_context(self, x_emb: Tensor) -> Tensor:
        """Average all token embeddings except the site's own token."""
        n_sites = self.lattice_size * self.lattice_size
        return (x_emb.sum(dim=(2, 3), keepdim=True) - x_emb) / max(n_sites - 1, 1)

    def compute_body(self, x: Tensor, t: Tensor) -> Tensor:
        """Build hollow convolutional features on the D x D torus."""
        x_idx = ((x + 1) / 2).long()
        batch = x_idx.shape[0]
        n = self.lattice_size
        x_grid = x_idx.view(batch, n, n)
        x_emb = self.token_embedder(x_grid).permute(0, 3, 1, 2)
        global_context = None
        if self.use_global_context:
            global_context = self.global_context_proj(self._leave_one_out_global_context(x_emb))

        cond_t = self.time_embedder(t)
        x_in = x_emb + cond_t[:, :, None, None]
        h = self.h_0[None, :, None, None] + cond_t[:, :, None, None]
        h = h.expand(-1, -1, n, n).contiguous()

        for layer_idx, k_size in enumerate(self.kernel_schedule):
            # The generated kernel is zeroed at the centre, so h_i never reads
            # x_i directly. Circular padding preserves lattice translation.
            kernel_state = h if global_context is None else h + global_context
            weights = F.gelu(self.A[layer_idx](kernel_state))
            weights = weights + self.c[layer_idx].view(1, -1, 1, 1)
            weights = weights.view(batch, k_size, k_size, n, n)
            mask = getattr(self, f"hollow_mask_{layer_idx}")
            weights = weights * mask[None, :, :, None, None]
            pad = k_size // 2
            x_unfold = F.pad(x_in, (pad, pad, pad, pad), mode="circular").unfold(2, k_size, 1).unfold(3, k_size, 1)
            h = torch.einsum("bijrs,bcrsij->bcrs", weights, x_unfold)

        return h.permute(0, 2, 3, 1).reshape(batch, n * n, self.hidden_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Return locally equivariant scores for all site/token replacements."""
        h = self.compute_body(x, t)
        x_idx = ((x + 1) / 2).long()
        omega_all = self.omega.weight
        omega_xi = self.omega(x_idx)
        diff = omega_all[None, None, :, :] - omega_xi[:, :, None, :]
        g = torch.einsum("bdh,bdsh->bds", h, diff)
        return g.scatter(-1, x_idx.unsqueeze(-1), 0.0)
