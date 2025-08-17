Here is a single-file, faithful PyTorch port of Perceiver IO with an end-to-end shape test. Reference: Perceiver IO paper&#x20;

```python
# perceiver_io_pytorch.py
# Canonical PyTorch port of Perceiver IO (Jaegle et al., 2021/2022 ICLR).
# Architectural fidelity priorities:
# - Distinct Q vs KV projections with possibly different input channel dims.
# - Encoder cross-attn: queries from latents, keys/values from inputs.
# - Decoder cross-attn: queries define output structure, keys/values from latents.
# - Trainable and Fourier positional encodings.
# - Modular components mirroring the Haiku/JAX implementation.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities and Encodings
# ----------------------------

def build_linear_positions(index_dims: Tuple[int, ...],
                           output_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """Return a grid of positions with shape (*index_dims, len(index_dims))."""
    coords = [torch.linspace(output_range[0], output_range[1], steps=n) for n in index_dims]
    mesh = torch.meshgrid(*coords, indexing="ij")
    pos = torch.stack(mesh, dim=-1)  # (*index_dims, D)
    return pos


def generate_fourier_features(pos: torch.Tensor,
                              num_bands: int,
                              max_resolution: Tuple[int, ...],
                              concat_pos: bool = True,
                              sine_only: bool = False) -> torch.Tensor:
    """
    pos: (..., d)
    returns: (..., F)
    """
    device = pos.device
    d = pos.shape[-1]
    # linearly spaced frequencies up to Nyquist per dimension
    # freq_bands: (d, num_bands)
    freq_bands = torch.stack(
        [torch.linspace(1.0, res / 2.0, steps=num_bands, device=device) for res in max_resolution], dim=0
    )
    # (..., d, num_bands)
    per_pos = pos.unsqueeze(-1) * freq_bands  # broadcast
    # (..., d * num_bands)
    per_pos = per_pos.reshape(*pos.shape[:-1], d * num_bands)

    if sine_only:
        feats = torch.sin(math.pi * per_pos)
    else:
        feats = torch.cat([torch.sin(math.pi * per_pos), torch.cos(math.pi * per_pos)], dim=-1)

    if concat_pos:
        feats = torch.cat([pos, feats], dim=-1)
    return feats


class TrainablePositionEncoding(nn.Module):
    """Trainable position embeddings broadcast along batch."""
    def __init__(self, index_dim: int, num_channels: int, init_scale: float = 0.02):
        super().__init__()
        self.pos_embs = nn.Parameter(torch.empty(index_dim, num_channels))
        nn.init.trunc_normal_(self.pos_embs, std=init_scale)

    def forward(self, batch_size: int) -> torch.Tensor:
        # (B, index_dim, C)
        return self.pos_embs.unsqueeze(0).expand(batch_size, -1, -1)


class FourierPositionEncoding(nn.Module):
    """Fourier (sin/cos) features from spatial coordinates."""
    def __init__(self,
                 index_dims: Tuple[int, ...],
                 num_bands: int,
                 concat_pos: bool = True,
                 max_resolution: Optional[Tuple[int, ...]] = None,
                 sine_only: bool = False):
        super().__init__()
        self.index_dims = tuple(index_dims)
        self.num_bands = int(num_bands)
        self.concat_pos = bool(concat_pos)
        self.max_resolution = tuple(max_resolution) if max_resolution is not None else tuple(index_dims)
        self.sine_only = bool(sine_only)

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pos is None:
            base = build_linear_positions(self.index_dims).reshape(-1, len(self.index_dims))  # (N, d)
            pos = base.unsqueeze(0).expand(batch_size, -1, -1).to(self._device())
        feats = generate_fourier_features(
            pos, num_bands=self.num_bands, max_resolution=self.max_resolution,
            concat_pos=self.concat_pos, sine_only=self.sine_only
        )
        return feats  # (B, N, C)

    def _device(self):
        # best-effort device selection
        return next(self.parameters(), torch.tensor(0.)).device


class PositionEncodingProjector(nn.Module):
    def __init__(self, output_size: int, base_encoding: nn.Module):
        super().__init__()
        self.base = base_encoding
        self.proj = nn.Linear(-1, output_size, bias=True)  # placeholder, will be rebuilt lazily

        # lazy build flag
        self._built = False

    def forward(self, batch_size: int, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.base(batch_size, pos)
        if not self._built:
            in_features = base.shape[-1]
            self.proj = nn.Linear(in_features, self.proj.out_features if self.proj.out_features != -1 else in_features,
                                  bias=True).to(base.device)
            self._built = True
        return self.proj(base)


# ----------------------------
# Attention Primitives
# ----------------------------

def make_cross_attention_mask(query_mask: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
    """
    query_mask: (B, Tq) 0/1
    kv_mask:    (B, Tk) 0/1
    returns:    (B, Tq, Tk) bool
    """
    return (query_mask.unsqueeze(-1) * kv_mask.unsqueeze(1)).bool()


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with distinct projections for Q from inputs_q and K/V from inputs_kv.
    Channels may differ between inputs_q and inputs_kv. Projections are built lazily on first call.
    """
    def __init__(self,
                 num_heads: int = 8,
                 qk_channels: Optional[int] = None,
                 v_channels: Optional[int] = None,
                 output_channels: Optional[int] = None,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._output_channels = output_channels
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        # Projections will be created lazily based on observed input dims.
        self.to_q: Optional[nn.Linear] = None
        self.to_k: Optional[nn.Linear] = None
        self.to_v: Optional[nn.Linear] = None
        self.to_out: Optional[nn.Linear] = None

    def _lazy_build(self, c_q: int, c_kv: int, device) -> Tuple[int, int, int]:
        qk_channels = self._qk_channels if self._qk_channels is not None else c_q
        v_channels = self._v_channels if self._v_channels is not None else qk_channels
        out_channels = self._output_channels if self._output_channels is not None else v_channels

        if qk_channels % self.num_heads != 0:
            raise ValueError(f"qk_channels={qk_channels} not divisible by num_heads={self.num_heads}")
        if v_channels % self.num_heads != 0:
            raise ValueError(f"v_channels={v_channels} not divisible by num_heads={self.num_heads}")

        if self.to_q is None:
            self.to_q = nn.Linear(c_q, qk_channels, bias=True).to(device)
        if self.to_k is None:
            self.to_k = nn.Linear(c_kv, qk_channels, bias=True).to(device)
        if self.to_v is None:
            self.to_v = nn.Linear(c_kv, v_channels, bias=True).to(device)
        if self.to_out is None:
            self.to_out = nn.Linear(v_channels, out_channels, bias=True).to(device)

        return qk_channels, v_channels, out_channels

    def forward(self,
                inputs_q: torch.Tensor,   # (B, Tq, Cq)
                inputs_kv: torch.Tensor,  # (B, Tk, Ckv)
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Tq, Cq = inputs_q.shape
        _, Tk, Ckv = inputs_kv.shape

        qk_channels, v_channels, out_channels = self._lazy_build(Cq, Ckv, inputs_q.device)
        q_head_dim = qk_channels // self.num_heads
        v_head_dim = v_channels // self.num_heads

        q = self.to_q(inputs_q)  # (B, Tq, qk_channels)
        k = self.to_k(inputs_kv) # (B, Tk, qk_channels)
        v = self.to_v(inputs_kv) # (B, Tk, v_channels)

        # reshape to heads
        q = q.view(B, Tq, self.num_heads, q_head_dim)
        k = k.view(B, Tk, self.num_heads, q_head_dim)
        v = v.view(B, Tk, self.num_heads, v_head_dim)

        # scores: (B, H, Tq, Tk)
        scores = torch.einsum("bthd,bThd->bhtT", q, k)
        scale = 1.0 / math.sqrt(q_head_dim)
        scores = scores * scale

        if attention_mask is not None:
            # attention_mask: (B, Tq, Tk) bool
            large_neg = 1e4 if scores.dtype == torch.float16 else 1e30
            scores = scores.masked_fill(~attention_mask.unsqueeze(1), -large_neg)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # out: (B, Tq, H, v_head_dim)
        out = torch.einsum("bhtT,bThd->bthd", attn, v).contiguous()
        out = out.view(B, Tq, v_channels)

        # wipe rows with fully masked keys
        if attention_mask is not None:
            wipe = (attention_mask.sum(dim=-1) == 0).unsqueeze(-1).expand_as(out)  # (B, Tq, v_channels)
            out = torch.where(wipe, torch.zeros_like(out), out)

        out = self.to_out(out)  # (B, Tq, out_channels)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, widening_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.widening_factor = widening_factor
        self.dropout = dropout
        self.fc1: Optional[nn.Linear] = None
        self.fc2: Optional[nn.Linear] = None
        self.drop = nn.Dropout(dropout)
        self._built = False

    def _lazy_build(self, dim: int, device):
        if not self._built:
            self.fc1 = nn.Linear(dim, self.widening_factor * dim, bias=True).to(device)
            self.fc2 = nn.Linear(self.widening_factor * dim, dim, bias=True).to(device)
            self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._lazy_build(x.size(-1), x.device)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.drop(x)


class SelfAttentionBlock(nn.Module):
    """LayerNorm -> MHSA -> Dropout -> Residual; then MLP block with residual."""
    def __init__(self,
                 num_heads: int = 8,
                 widening_factor: int = 4,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 qk_channels: Optional[int] = None,
                 v_channels: Optional[int] = None):
        super().__init__()
        self.ln1 = nn.LayerNorm(-1, elementwise_affine=True)  # will rebuild lazily
        self.ln2 = nn.LayerNorm(-1, elementwise_affine=True)
        self.attn = MultiHeadAttention(num_heads=num_heads,
                                       qk_channels=qk_channels,
                                       v_channels=v_channels,
                                       output_channels=None,
                                       attn_dropout=attn_dropout,
                                       proj_dropout=dropout)
        self.mlp = MLP(widening_factor=widening_factor, dropout=dropout)

        self._built = False

    def _lazy_build(self, dim: int, device):
        if not self._built:
            self.ln1 = nn.LayerNorm(dim).to(device)
            self.ln2 = nn.LayerNorm(dim).to(device)
            self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._lazy_build(x.size(-1), x.device)
        h = x
        attn_out = self.attn(self.ln1(x), self.ln1(x), attention_mask=None)
        x = h + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention (queries from inputs_q, keys/values from inputs_kv) followed by MLP."""
    def __init__(self,
                 num_heads: int = 8,
                 widening_factor: int = 1,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 qk_channels: Optional[int] = None,
                 v_channels: Optional[int] = None,
                 use_query_residual: bool = True):
        super().__init__()
        self.use_query_residual = use_query_residual
        self.ln_q = nn.LayerNorm(-1, elementwise_affine=True)
        self.ln_kv = nn.LayerNorm(-1, elementwise_affine=True)
        self.attn = MultiHeadAttention(num_heads=num_heads,
                                       qk_channels=qk_channels,
                                       v_channels=v_channels,
                                       output_channels=None,
                                       attn_dropout=attn_dropout,
                                       proj_dropout=dropout)
        self.mlp = MLP(widening_factor=widening_factor, dropout=dropout)
        self._built = False

    def _lazy_build(self, dq: int, dkv: int, device):
        if not self._built:
            self.ln_q = nn.LayerNorm(dq).to(device)
            self.ln_kv = nn.LayerNorm(dkv).to(device)
            self._built = True

    def forward(self,
                inputs_q: torch.Tensor,   # (B, Tq, Cq)
                inputs_kv: torch.Tensor,  # (B, Tk, Ckv)
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._lazy_build(inputs_q.size(-1), inputs_kv.size(-1), inputs_q.device)
        q_norm = self.ln_q(inputs_q)
        kv_norm = self.ln_kv(inputs_kv)
        attn_out = self.attn(q_norm, kv_norm, attention_mask=attention_mask)
        x = inputs_q + attn_out if self.use_query_residual else attn_out
        x = x + self.mlp(self.ln_q(x))
        return x


# ----------------------------
# Perceiver Encoder / Decoder
# ----------------------------

class PerceiverEncoder(nn.Module):
    """
    Perceiver Encoder:
    - Latent array with trainable position embeddings: z in R^{z_index_dim x z_channels}
    - Cross-attention: queries=z, keys/values=inputs
      (q_channels=z_channels, kv_channels=input_channels)
    - Followed by num_blocks of num_self_attends_per_block self-attention layers (weights shared by block?):
      We match functional behavior by repeating the same SelfAttention spec per layer.
    """
    def __init__(self,
                 num_self_attends_per_block: int = 6,
                 num_blocks: int = 8,
                 z_index_dim: int = 512,
                 z_channels: int = 1024,
                 num_cross_attend_heads: int = 1,
                 num_self_attend_heads: int = 8,
                 cross_attend_widening: int = 1,
                 self_attend_widening: int = 1,
                 dropout: float = 0.0,
                 z_pos_enc_init_scale: float = 0.02,
                 use_query_residual: bool = True):
        super().__init__()

        self.z_index_dim = z_index_dim
        self.z_channels = z_channels
        self.num_blocks = num_blocks

        self.z_pos_enc = TrainablePositionEncoding(index_dim=z_index_dim,
                                                   num_channels=z_channels,
                                                   init_scale=z_pos_enc_init_scale)

        self.cross_attend = CrossAttentionBlock(
            num_heads=num_cross_attend_heads,
            widening_factor=cross_attend_widening,
            attn_dropout=dropout,
            dropout=dropout,
            qk_channels=None,  # defaults: qk from inputs
            v_channels=None,
            use_query_residual=use_query_residual
        )

        self.self_attends = nn.ModuleList([
            SelfAttentionBlock(num_heads=num_self_attend_heads,
                               widening_factor=self_attend_widening,
                               attn_dropout=dropout,
                               dropout=dropout,
                               qk_channels=None,
                               v_channels=None)
            for _ in range(num_self_attends_per_block)
        ])

    def latents(self, batch_size: int) -> torch.Tensor:
        # (B, N=z_index_dim, C=z_channels)
        return self.z_pos_enc(batch_size)

    def forward(self,
                inputs: torch.Tensor,     # (B, M, Cin)
                z: torch.Tensor,          # (B, N, z_channels)
                input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        input_mask: (B, M) 0/1 or bool
        """
        B, N, _ = z.shape
        _, M, _ = inputs.shape

        attention_mask = None
        if input_mask is not None:
            if input_mask.dtype != torch.bool:
                input_mask = input_mask.bool()
            query_mask = torch.ones((B, N), dtype=torch.bool, device=inputs.device)
            attention_mask = make_cross_attention_mask(query_mask, input_mask)  # (B, N, M)

        # Cross-attend: queries are latents; keys/values are inputs
        z = self.cross_attend(z, inputs, attention_mask=attention_mask)
        # Process in latent space
        for _ in range(self.num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z)
        return z


class BasicDecoder(nn.Module):
    """
    Cross-attention-based decoder.
    - inputs_query: array that defines output structure (e.g., positions or learned queries) of shape (B, O, Dq)
    - keys/values come from latent array z of shape (B, N, Dz)
    - final linear projection to output channels (E)
    """
    def __init__(self,
                 output_channels: int,
                 latent_channels: int,
                 num_heads: int = 1,
                 widening: int = 1,
                 dropout: float = 0.0,
                 use_query_residual: bool = False,
                 final_project: bool = True):
        super().__init__()
        self.cross = CrossAttentionBlock(
            num_heads=num_heads,
            widening_factor=widening,
            attn_dropout=dropout,
            dropout=dropout,
            qk_channels=None,
            v_channels=None,
            use_query_residual=use_query_residual
        )
        self.final_project = final_project
        self.out = nn.Linear(-1, output_channels)  # placeholder, to build lazily
        self._built = False

    def _lazy_build(self, dq: int, device):
        if not self._built:
            self.out = nn.Linear(dq, self.out.out_features if self.out.out_features != -1 else dq).to(device)
            self._built = True

    def forward(self,
                query: torch.Tensor,  # (B, O, Dq)
                z: torch.Tensor,      # (B, N, Dz)
                query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_mask = None
        if query_mask is not None:
            if query_mask.dtype != torch.bool:
                query_mask = query_mask.bool()
            kv_mask = torch.ones(z.size(0), z.size(1), dtype=torch.bool, device=z.device)
            attention_mask = make_cross_attention_mask(query_mask, kv_mask)  # (B, O, N)

        dec = self.cross(query, z, attention_mask=attention_mask)
        self._lazy_build(dec.size(-1), dec.device)
        return self.out(dec) if self.final_project else dec


class PerceiverIO(nn.Module):
    """
    Perceiver IO: Encoder + Decoder with optional masks.
    Expects inputs already preprocessed to 2D array (B, M, Cin) and query array (B, O, Dq).
    """
    def __init__(self,
                 encoder: PerceiverEncoder,
                 decoder: BasicDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                inputs: torch.Tensor,          # (B, M, Cin)
                output_query: torch.Tensor,    # (B, O, Dq)
                input_mask: Optional[torch.Tensor] = None,
                query_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = inputs.size(0)
        z0 = self.encoder.latents(B)
        z = self.encoder(inputs, z0, input_mask=input_mask)
        y = self.decoder(output_query, z, query_mask=query_mask)
        return y


# ----------------------------
# End-to-end verification
# ----------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Hyperparameters per objective
    B = 2
    N_in = 4096          # number of input tokens
    C_in = 256           # input feature size (input_channels)
    N_lat = 512          # z_index_dim
    C_lat = 1024         # z_channels
    M_out = 100          # number of output queries
    Dq = 512             # output_query_channels
    C_out = 3            # output channels

    # Dummy inputs and queries
    x = torch.randn(B, N_in, C_in)             # preprocessed image tokens
    q = torch.randn(B, M_out, Dq)              # output query array

    # Build model
    encoder = PerceiverEncoder(
        num_self_attends_per_block=6,
        num_blocks=2,                 # keep test fast while preserving structure
        z_index_dim=N_lat,
        z_channels=C_lat,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening=1,
        self_attend_widening=1,
        dropout=0.0,
        use_query_residual=True
    )

    decoder = BasicDecoder(
        output_channels=C_out,
        latent_channels=C_lat,
        num_heads=1,
        widening=1,
        dropout=0.0,
        use_query_residual=False,
        final_project=True
    )

    model = PerceiverIO(encoder=encoder, decoder=decoder)

    # Forward pass
    y = model(x, q)  # (B, M_out, C_out)

    # Assertions for dimensional correctness
    assert y.shape == (B, M_out, C_out), f"Unexpected output shape {y.shape}, expected {(B, M_out, C_out)}"

    print("Output shape:", tuple(y.shape))
```

Run:

```bash
python perceiver_io_pytorch.py
```

Expected:

```
Output shape: (2, 100, 3)
```
