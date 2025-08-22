# train_ddpm.py
# Custom DDPM training script to REPLACE the old trajectory-based process.
# Trains DiffusionLRU purely as a denoiser over *clean* weight vectors (no path imitation).
#
# Key behaviors:
# - Loads a directory of vectors (.npy or .pt containing 1D float tensors) with optional mixed lengths
# - Pads per-batch and uses a masked DDPM loss (supports variable D)
# - Supports ε/x0/v parameterizations (default v), cosine schedule, EMA, AMP
# - Periodically samples vectors with DDIM and saves them to disk
#
# Example:
#   python train_ddpm.py \
#       --data_dir ./weights_corpus \
#       --epochs 50 --batch_size 8 --model_dim 256 --state_dim 256 --depth 6 \
#       --param v --T 1000 --schedule cosine --lr 2e-4 --save_dir ./runs/ddpm_lru
#
# Expected data layout:
#   weights_corpus/
#       modelA.npy        # shape (D,)
#       modelB.pt         # torch.tensor shape (D,)
#       ...

import os, time, json, math, argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lru_ddpm import DiffusionLRU, DiffusionConfig, train_step, VectorStandardizer


# -------------------------
# Dataset
# -------------------------

class VectorFileDataset(Dataset):
    def __init__(self, data_dir: str, exts=(".npy", ".pt")):
        self.paths = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))
                      if any(f.endswith(e) for e in exts)]
        if not self.paths:
            raise ValueError(f"No vectors found in {data_dir} (accepted: {exts})")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        if p.endswith('.npy'):
            arr = np.load(p)
            x = torch.tensor(arr, dtype=torch.float32)
        else:
            x = torch.load(p)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"{p} does not contain a tensor")
            x = x.float().view(-1)
        if x.dim() != 1:
            x = x.view(-1)
        return x


def collate_pad(batch: List[torch.Tensor]):
    lengths = [b.numel() for b in batch]
    L = max(lengths)
    B = len(batch)
    x = torch.zeros(B, L, dtype=torch.float32)
    mask = torch.zeros(B, L, dtype=torch.float32)
    for i, v in enumerate(batch):
        n = v.numel()
        x[i, :n] = v
        mask[i, :n] = 1.0
    return x, mask


# -------------------------
# Training
# -------------------------

def save_checkpoint(save_dir: str, step: int, model: DiffusionLRU, opt: torch.optim.Optimizer, cfg: DiffusionConfig, args):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'cfg': cfg.__dict__,
        'args': vars(args),
    }
    torch.save(ckpt, os.path.join(save_dir, f'ckpt_{step:07d}.pt'))


def sample_and_save(save_dir: str, model: DiffusionLRU, length: int, num: int = 4, steps: int = 50, eta: float = 0.0, device: Optional[torch.device] = None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        samples = model.sample_ddim(num_samples=num, length=length, steps=steps, eta=eta, device=device)
    np.save(os.path.join(save_dir, f'samples_len{length}_n{num}_steps{steps}.npy'), samples.cpu().numpy())


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='./runs/ddpm_lru')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation steps')

    ap.add_argument('--model_dim', type=int, default=256)
    ap.add_argument('--state_dim', type=int, default=256)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--bidirectional', action='store_true')
    ap.add_argument('--norm', type=str, default='layer', choices=['layer','batch'])
    ap.add_argument('--r_min', type=float, default=0.8)
    ap.add_argument('--r_max', type=float, default=0.99)
    ap.add_argument('--max_phase', type=float, default=math.pi/10)

    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--schedule', type=str, default='cosine', choices=['cosine','linear'])
    ap.add_argument('--param', type=str, default='v', choices=['v','eps','x0'])
    ap.add_argument('--ema', type=float, default=0.9999)

    ap.add_argument('--sample_every', type=int, default=1000)
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--num_samples', type=int, default=4)
    ap.add_argument('--sample_steps', type=int, default=50)
    ap.add_argument('--eta', type=float, default=0.0)

    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    ds = VectorFileDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pad, drop_last=True)

    # Infer a canonical sample length from the *max* length in the corpus (for sampling)
    max_len = 0
    for v, _ in dl:
        max_len = v.shape[1]
        break  # from first batch (already padded to batch max); safe fallback below if empty
    if max_len == 0:
        # fallback: scan files (very rare)
        max_len = max(x.numel() for x in ds)

    # Model & config
    cfg = DiffusionConfig(T=args.T, beta_schedule=args.schedule, param=args.param, use_ema=True, ema_decay=args.ema)
    model = DiffusionLRU(vector_dim=max_len, model_dim=args.model_dim, state_dim=args.state_dim,
                         depth=args.depth, norm_type=args.norm, r_min=args.r_min, r_max=args.r_max,
                         max_phase=args.max_phase, bidirectional=args.bidirectional, cfg=cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and torch.cuda.is_available()) else None

    os.makedirs(args.save_dir, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for x0, mask in dl:
            x0 = x0.to(device)
            mask = mask.to(device)
            # Optional: per-batch standardization (kept simple: center+scale then forget)
            # stdzr = VectorStandardizer(); x0, _, _ = stdzr.fit_transform(x0)

            loss_accum = 0.0
            for k in range(args.accum):
                loss = model.training_loss(x0, mask=mask)
                loss = loss / args.accum
                if scaler is None:
                    loss.backward()
                else:
                    with torch.cuda.amp.autocast():
                        pass  # already forward under autocast inside training_loss
                    scaler.scale(loss).backward()
                loss_accum += float(loss.detach())

            if args.grad_clip is not None:
                if scaler is None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                else:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler is None:
                opt.step()
            else:
                scaler.step(opt)
                scaler.update()
            opt.zero_grad(set_to_none=True)

            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss_accum:.6f}")

            if args.sample_every > 0 and step % args.sample_every == 0 and step > 0:
                sample_and_save(args.save_dir, model, length=max_len, num=args.num_samples, steps=args.sample_steps, eta=args.eta, device=device)

            if args.save_every > 0 and step % args.save_every == 0 and step > 0:
                save_checkpoint(args.save_dir, step, model, opt, cfg, args)

            step += 1

    # Final save + sample
    save_checkpoint(args.save_dir, step, model, opt, cfg, args)
    sample_and_save(args.save_dir, model, length=max_len, num=args.num_samples, steps=args.sample_steps, eta=args.eta, device=device)


if __name__ == '__main__':
    main()
