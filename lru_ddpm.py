# lru_ddpm.py
# Advanced DDPM over weight *vectors* with an LRU backbone.
# 
# Philosophy (per your upgrade): this is **not** an optimizer-imitator. It learns the
# distribution of effective weight vectors directly and *denoises* Gaussian noise into
# high-performing configurations. No trajectory imitation, no dataset at synthesis time.
# 
# What’s new vs the first drop:
# - Parameterizations: ε-, x0-, and v-pred (Imagen-style) training
# - Schedules: linear/cosine betas; DDIM sampler (η∈[0,1]); classic DDPM sampler
# - Variable-length support via masked loss (pad to max-L within batch)
# - Optional conditioning vector (for future: architecture/dataset IDs), CFG-ready
# - Exponential Moving Average (EMA) weights for sampling stability
# - Per-vector standardization utilities (fit/transform/invert)
# - Flatten/restore helpers to map nn.Module ↔ 1D vector (shape registry)
# - AMP-friendly training utilities (sketch), gradient accumulation friendly
#
# Usage sketch:
#   D = total_params
#   model = DiffusionLRU(vector_dim=D, model_dim=256, state_dim=256, depth=6,
#                        param="v", cfg=DiffusionConfig(beta_schedule="cosine"))
#   loss = model.training_loss(x0, mask=mask)   # x0: (B,D), mask optional
#   samples = model.sample_ddim(num_samples=K, length=D, steps=250, eta=0.0)  # deterministic
#
# Optional: plug weights into a target nn.Module using flatten/restore utilities below.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------

def default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal embedding for integer timesteps.
    timesteps: (B,) int64 or float32 in [0, T-1] → (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(1, half))
    t = timesteps.float()[:, None]
    args = t * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -------------------------
# Data scaling (per-vector standardization)
# -------------------------

class VectorStandardizer:
    """Track mean/std for each sample vector and invertibly scale.
    Useful when the training set mixes heterogeneous models whose raw scales differ.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, D)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std, mean.squeeze(1), std.squeeze(1)

    def invert(self, x_scaled: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x_scaled * std.unsqueeze(1) + mean.unsqueeze(1)


# -------------------------
# EMA helper
# -------------------------

class EMA:
    def __init__(self, module: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.device = device or next(module.parameters()).device
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().to(self.device)

    @torch.no_grad()
    def update(self, module: nn.Module):
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, module: nn.Module):
        for name, p in module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])


# -------------------------
# Complex-diagonal LRU core
# -------------------------

class LRU(nn.Module):
    """Linear Recurrent Unit with complex diagonal recurrence.

    x_k = Λ x_{k-1} + exp(γ_log) ⊙ (B u_k)
    y_k = Re(C x_k) + D ⊙ u_k

    Shapes:
      - input u: (B, L, H)
      - output y: (B, L, H)
      - state size N, model dim H

    Parameterization matches the paper: stable exponential for Λ, γ-normalization, Glorot init.
    """

    def __init__(self, state_dim: int, model_dim: int,
                 r_min: float = 0.0, r_max: float = 1.0,
                 max_phase: float = 2 * math.pi,
                 bidirectional: bool = False):
        super().__init__()
        N, H = state_dim, model_dim
        self.N = N
        self.H = H
        self.bidirectional = bidirectional

        # Λ = exp(-exp(ν_log) + i * exp(θ_log))
        self.nu_log = nn.Parameter(torch.zeros(N))
        self.theta_log = nn.Parameter(torch.zeros(N))

        # Complex B, C via real/imag parameters
        self.B_re = nn.Parameter(torch.empty(N, H))
        self.B_im = nn.Parameter(torch.empty(N, H))
        self.C_re = nn.Parameter(torch.empty(H, N))
        self.C_im = nn.Parameter(torch.empty(H, N))

        # D: (H,)
        self.D = nn.Parameter(torch.empty(H))

        # γ normalization (vector over N)
        self.gamma_log = nn.Parameter(torch.empty(N))

        self.reset_parameters(r_min=r_min, r_max=r_max, max_phase=max_phase)

    @torch.no_grad()
    def reset_parameters(self, r_min: float = 0.0, r_max: float = 1.0, max_phase: float = 2 * math.pi):
        N, H = self.N, self.H
        device = self.B_re.device if self.B_re.device.type != 'meta' else default_device()
        u1 = torch.rand(N, device=device)
        u2 = torch.rand(N, device=device)
        r2 = u1 * (r_max ** 2 - r_min ** 2) + (r_min ** 2)
        r = torch.sqrt(r2).clamp(min=1e-8)
        nu = (-r.log()).clamp(min=1e-6)
        theta = (u2 * max_phase).clamp(min=1e-6)
        self.nu_log.data.copy_(nu.log())
        self.theta_log.data.copy_(theta.log())

        lam = torch.exp(-torch.exp(self.nu_log)) * torch.exp(1j * torch.exp(self.theta_log))
        lam_abs2 = lam.real ** 2 + lam.imag ** 2
        gamma0 = torch.sqrt((1.0 - lam_abs2).clamp(min=1e-12))
        self.gamma_log.data.copy_(gamma0.log())

        def glorot(shape, scale=1.0):
            fan_in, fan_out = shape[1], shape[0]
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return torch.randn(*shape, device=device) * (std * scale)

        self.B_re.data.copy_(glorot((N, H), scale=2.0))
        self.B_im.data.copy_(glorot((N, H), scale=2.0))
        self.C_re.data.copy_(glorot((H, N)))
        self.C_im.data.copy_(glorot((H, N)))
        self.D.data.copy_(torch.randn(self.H, device=device) / math.sqrt(max(1, self.H)))

    def complex_matvec(self, A_re: torch.Tensor, A_im: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        re = F.linear(v, A_re)
        im = F.linear(v, A_im)
        return re, im

    def complex_vecdot(self, A_re: torch.Tensor, A_im: torch.Tensor, v_re: torch.Tensor, v_im: torch.Tensor) -> torch.Tensor:
        return F.linear(v_re, A_re) - F.linear(v_im, A_im)

    def forward_once(self, u: torch.Tensor) -> torch.Tensor:
        B = u.shape[0]
        L = u.shape[1]
        lam_mag = torch.exp(-torch.exp(self.nu_log))
        lam_phase = torch.exp(self.theta_log)
        lam_re = lam_mag * torch.cos(lam_phase)
        lam_im = lam_mag * torch.sin(lam_phase)
        gamma = torch.exp(self.gamma_log)
        B_re = self.B_re * gamma[:, None]
        B_im = self.B_im * gamma[:, None]

        x_re = u.new_zeros(B, self.N)
        x_im = u.new_zeros(B, self.N)
        outputs = []
        for k in range(L):
            uk = u[:, k, :]
            t_re, t_im = self.complex_matvec(B_re, B_im, uk)
            xr = x_re * lam_re - x_im * lam_im + t_re
            xi = x_re * lam_im + x_im * lam_re + t_im
            x_re, x_im = xr, xi
            yk = self.complex_vecdot(self.C_re, self.C_im, x_re, x_im) + self.D * uk
            outputs.append(yk.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if not self.bidirectional:
            return self.forward_once(u)
        y_f = self.forward_once(u)
        y_b_rev = self.forward_once(torch.flip(u, dims=[1]))
        y_b = torch.flip(y_b_rev, dims=[1])
        return 0.5 * (y_f + y_b)


# -------------------------
# LRU residual block: Norm → LRU → GLU → Skip (+ optional conditioning)
# -------------------------

class GLUMix(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, use_linear_after: bool = True):
        super().__init__()
        self.lin = nn.Linear(dim, 2 * dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim) if use_linear_after else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.lin(x).chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        h = self.dropout(h)
        return self.out(h)


class ResidualLRUBlock(nn.Module):
    def __init__(self, model_dim: int, state_dim: int,
                 norm_type: str = 'layer', dropout: float = 0.0,
                 r_min: float = 0.0, r_max: float = 1.0, max_phase: float = 2 * math.pi,
                 bidirectional: bool = False,
                 glu_without_extra_linear: bool = False,
                 time_embed_dim: Optional[int] = None,
                 cond_dim: Optional[int] = None):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim) if norm_type != 'batch' else nn.BatchNorm1d(model_dim)
        self.lru = LRU(state_dim=state_dim, model_dim=model_dim,
                       r_min=r_min, r_max=r_max, max_phase=max_phase,
                       bidirectional=bidirectional)
        self.glu = GLUMix(model_dim, dropout=dropout, use_linear_after=not glu_without_extra_linear)
        self.cond_time = nn.Linear(time_embed_dim, model_dim) if time_embed_dim is not None else None
        self.cond_vec = nn.Linear(cond_dim, model_dim) if cond_dim is not None else None

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        if isinstance(self.norm, nn.BatchNorm1d):
            B, L, H = x.shape
            xn = self.norm(x.reshape(B * L, H)).reshape(B, L, H)
        else:
            xn = self.norm(x)
        if self.cond_time is not None and time_emb is not None:
            xn = xn + self.cond_time(time_emb).unsqueeze(1)
        if self.cond_vec is not None and cond is not None:
            xn = xn + self.cond_vec(cond).unsqueeze(1)
        h = self.lru(xn)
        h = self.glu(h)
        return residual + h


class LRUBackbone(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, state_dim: int,
                 depth: int = 6, dropout: float = 0.0, norm_type: str = 'layer',
                 r_min: float = 0.0, r_max: float = 0.99, max_phase: float = math.pi / 10,
                 bidirectional: bool = False,
                 glu_without_extra_linear_for_longseq: bool = False,
                 time_embed_dim: Optional[int] = None,
                 cond_dim: Optional[int] = None):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, model_dim)
        self.blocks = nn.ModuleList([
            ResidualLRUBlock(
                model_dim=model_dim,
                state_dim=state_dim,
                norm_type=norm_type,
                dropout=dropout,
                r_min=r_min, r_max=r_max, max_phase=max_phase,
                bidirectional=bidirectional,
                glu_without_extra_linear=glu_without_extra_linear_for_longseq,
                time_embed_dim=time_embed_dim,
                cond_dim=cond_dim,
            ) for _ in range(depth)
        ])
        self.out_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h, time_emb=time_emb, cond=cond)
        return self.out_proj(h)


# -------------------------
# DDPM wrapper for vectors
# -------------------------

@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_schedule: str = 'cosine'   # 'linear' or 'cosine'
    beta_start: float = 1e-4        # used if beta_schedule == 'linear'
    beta_end: float = 0.02          # used if beta_schedule == 'linear'
    param: str = 'v'                # 'eps' | 'x0' | 'v' (training target)
    use_ema: bool = True
    ema_decay: float = 0.9999


class DiffusionLRU(nn.Module):
    def __init__(self, vector_dim: int, model_dim: int = 256, state_dim: int = 256,
                 depth: int = 6, dropout: float = 0.0, norm_type: str = 'layer',
                 r_min: float = 0.0, r_max: float = 0.99, max_phase: float = math.pi / 10,
                 bidirectional: bool = False,
                 cfg: DiffusionConfig = DiffusionConfig(),
                 cond_dim: Optional[int] = None):
        super().__init__()
        self.vector_dim = vector_dim
        self.cfg = cfg
        self.cond_dim = cond_dim

        time_dim = model_dim * 2
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, model_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        ) if cond_dim is not None else None

        self.backbone = LRUBackbone(
            input_dim=1,
            model_dim=model_dim,
            state_dim=state_dim,
            depth=depth,
            dropout=dropout,
            norm_type=norm_type,
            r_min=r_min, r_max=r_max, max_phase=max_phase,
            bidirectional=bidirectional,
            glu_without_extra_linear_for_longseq=False,
            time_embed_dim=model_dim,
            cond_dim=model_dim if cond_dim is not None else None,
        )

        self.register_buffers()
        self.ema: Optional[EMA] = None
        if cfg.use_ema:
            self.ema = EMA(self, decay=cfg.ema_decay)

    # ----------- schedules
    def register_buffers(self):
        cfg = self.cfg
        T = cfg.T
        if cfg.beta_schedule == 'linear':
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, dtype=torch.float32)
        elif cfg.beta_schedule == 'cosine':
            s = 0.008
            steps = T + 1
            x = torch.linspace(0, T, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError('unknown beta_schedule')
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    # ----------- core model
    def _backbone_forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        # x_t: (B, D) → (B, L=D, 1)
        B, D = x_t.shape
        x_seq = x_t.unsqueeze(-1)
        time_emb_big = sinusoidal_timestep_embedding(t, dim=self.time_embed_mlp[0].in_features)
        time_emb = self.time_embed_mlp(time_emb_big)
        cond_emb = self.cond_mlp(cond) if (self.cond_mlp is not None and cond is not None) else None
        eps_seq = self.backbone(x_seq, time_emb=time_emb, cond=cond_emb)
        return eps_seq.squeeze(-1)  # (B, D)

    # Parameterizations
    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cfg.param == 'eps':
            return self._backbone_forward(x_t, t, cond)
        elif self.cfg.param == 'x0':
            x0 = self._backbone_forward(x_t, t, cond)
            eps = (x_t - self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x0) / self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            return eps
        elif self.cfg.param == 'v':
            v = self._backbone_forward(x_t, t, cond)
            # v = sqrt(alpha_t) * eps - sqrt(1-alpha_t) * x0  ⇒ eps = (v + sqrt(1-a)*x0)/sqrt(a)
            # Recover eps by: eps = sqrt(a) * v + (1 - a) * ???  (we don’t have x0 here)
            # For training/sampling we use v directly in dedicated formulas.
            # Provide eps-form only when needed by DDPM posterior; below we avoid converting.
            return v  # interpreted downstream
        else:
            raise ValueError('cfg.param must be one of {"eps","x0","v"}')

    # ----------- training loss (masked, param-style)
    def training_loss(self, x0: torch.Tensor, mask: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x0: (B, D), mask: (B, D) with 1 for valid positions, 0 for padding.
        cond: optional global conditioning vector (B, C) — e.g., arch/dataset ID embedding.
        """
        device = x0.device
        B, D = x0.shape
        T = self.cfg.T
        t = torch.randint(0, T, (B,), device=device)
        noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_t = sqrt_ac * x0 + sqrt_om * noise

        if self.cfg.param == 'eps':
            pred = self._backbone_forward(x_t, t, cond)
            target = noise
        elif self.cfg.param == 'x0':
            pred = self._backbone_forward(x_t, t, cond)
            target = x0
        else:  # 'v'
            # v = sqrt(a)*eps - sqrt(1-a)*x0 with a = ᾱ_t (cumulative)
            a = self.alphas_cumprod[t].unsqueeze(-1)
            target = torch.sqrt(a) * noise - torch.sqrt(1 - a) * x0
            pred = self._backbone_forward(x_t, t, cond)

        if mask is not None:
            loss = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = F.mse_loss(pred, target)

        if self.ema is not None:
            self.ema.update(self)
        return loss

    # ----------- helpers for parameterizations during sampling
    def _pred_x0_from_eps(self, x_t, t, eps):
        return (x_t - self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * eps) / self.sqrt_alphas_cumprod[t].unsqueeze(-1)

    def _pred_eps_from_v(self, x_t, t, v):
        a = self.alphas_cumprod[t].unsqueeze(-1)
        # v = sqrt(a)*eps - sqrt(1-a)*x0 ⇒ eps = (v + sqrt(1-a)*x0) / sqrt(a)
        x0 = self._pred_x0_from_v(x_t, t, v)
        return (v + torch.sqrt(1 - a) * x0) / torch.sqrt(a)

    def _pred_x0_from_v(self, x_t, t, v):
        a = self.alphas_cumprod[t].unsqueeze(-1)
        # v = sqrt(a)*eps - sqrt(1-a)*x0 and x_t = sqrt(a)*x0 + sqrt(1-a)*eps ⇒
        # x0 = sqrt(a)*x_t - sqrt(1-a)*v
        return torch.sqrt(a) * x_t - torch.sqrt(1 - a) * v

    # ----------- DDPM ancestral sampler
    @torch.no_grad()
    def sample_ddpm(self, num_samples: int, length: int, steps: Optional[int] = None,
                    cond: Optional[torch.Tensor] = None, use_ema: bool = True,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or default_device()
        x = torch.randn(num_samples, length, device=device)
        if use_ema and self.ema is not None:
            backup = {n: p.clone() for n, p in self.named_parameters() if p.requires_grad}
            self.ema.copy_to(self)
        T = self.cfg.T if steps is None else steps
        for ti in reversed(range(T)):
            t = torch.full((num_samples,), ti, device=device, dtype=torch.long)
            raw = self._backbone_forward(x, t, cond)
            if self.cfg.param == 'eps':
                eps = raw
            elif self.cfg.param == 'x0':
                x0 = raw
                eps = (x - self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x0) / self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            else:  # 'v'
                eps = self._pred_eps_from_v(x, t, raw)

            a_t = self.alphas[ti]
            beta_t = self.betas[ti]
            sqrt_recip_a = self.sqrt_recip_alphas[ti]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[ti]
            mean = sqrt_recip_a * (x - beta_t / sqrt_one_minus * eps)
            if ti > 0:
                var = self.posterior_variance[ti]
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        if use_ema and self.ema is not None:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[n])
        return x

    # ----------- DDIM sampler (deterministic at η=0)
    @torch.no_grad()
    def sample_ddim(self, num_samples: int, length: int, steps: int = 50, eta: float = 0.0,
                    cond: Optional[torch.Tensor] = None, use_ema: bool = True,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or default_device()
        # Create an evenly spaced subset of timesteps
        T = self.cfg.T
        assert steps <= T
        ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(num_samples, length, device=device)
        if use_ema and self.ema is not None:
            backup = {n: p.clone() for n, p in self.named_parameters() if p.requires_grad}
            self.ema.copy_to(self)
        for idx, ti in enumerate(ts):
            t = torch.full((num_samples,), int(ti.item()), device=device, dtype=torch.long)
            raw = self._backbone_forward(x, t, cond)
            if self.cfg.param == 'eps':
                eps = raw
                x0 = self._pred_x0_from_eps(x, t, eps)
            elif self.cfg.param == 'x0':
                x0 = raw
                eps = (x - self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x0) / self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
            else:  # 'v'
                x0 = self._pred_x0_from_v(x, t, raw)
                eps = self._pred_eps_from_v(x, t, raw)

            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[ts[min(idx + 1, steps - 1)].repeat(num_samples)] if idx < steps - 1 else torch.ones_like(a_t)
            sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            dir_xt = torch.sqrt(1 - a_prev).unsqueeze(-1) * eps
            x = torch.sqrt(a_prev).unsqueeze(-1) * x0 + dir_xt
            if eta > 0 and idx < steps - 1:
                x = x + sigma_t.unsqueeze(-1) * torch.randn_like(x)
        if use_ema and self.ema is not None:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[n])
        return x


# -------------------------
# Flatten/restore helpers for plugging vectors into modules
# -------------------------

def flatten_module_params(module: nn.Module) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size]]]:
    """Return a contiguous 1D vector of all parameters and a list of (name, shape) to restore later."""
    vecs = []
    shapes: List[Tuple[str, torch.Size]] = []
    with torch.no_grad():
        for name, p in module.named_parameters():
            vecs.append(p.detach().reshape(-1))
            shapes.append((name, p.shape))
    return torch.cat(vecs), shapes


def restore_module_params(module: nn.Module, vector: torch.Tensor, shapes: List[Tuple[str, torch.Size]]):
    """Write values from `vector` back into `module` following `shapes`. Assumes identical ordering."""
    offset = 0
    with torch.no_grad():
        for (name, shape), p in zip(shapes, module.named_parameters()):
            numel = p.numel()
            chunk = vector[offset:offset + numel].view(shape)
            p.copy_(chunk)
            offset += numel


# -------------------------
# Minimal training loop sketch (AMP / grad-accum friendly)
# -------------------------

def train_step(model: DiffusionLRU, optimizer: torch.optim.Optimizer, batch_x0: torch.Tensor,
               mask: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None,
               scaler: Optional[torch.cuda.amp.GradScaler] = None, grad_clip: Optional[float] = 1.0) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss = model.training_loss(batch_x0, mask=mask, cond=cond)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return float(loss.detach())
    else:
        with torch.cuda.amp.autocast():
            loss = model.training_loss(batch_x0, mask=mask, cond=cond)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())


# -------------------------
# Tiny sanity check
# -------------------------
if __name__ == '__main__':
    device = default_device()
    torch.manual_seed(0)
    B, D = 8, 257  # odd D to exercise masking/padding if you wish
    cfg = DiffusionConfig(T=1000, beta_schedule='cosine', param='v', use_ema=True)
    model = DiffusionLRU(vector_dim=D, model_dim=128, state_dim=128, depth=6,
                         norm_type='layer', r_min=0.8, r_max=0.99, max_phase=math.pi/10,
                         cfg=cfg).to(device)

    x0 = torch.randn(B, D, device=device)
    loss = model.training_loss(x0)
    print('DDPM loss (v-param):', float(loss))

    with torch.no_grad():
        samples = model.sample_ddim(num_samples=4, length=D, steps=50, eta=0.0, device=device)
        print('DDIM samples:', samples.shape)
