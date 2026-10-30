"""Entraînement ADM-T (L_simple) ou TDN (L_simple + λ L_top).

ADM-T :  python train.py --data data/shapes --out ckpt/admt --steps 15000
TDN   :  python train.py --data data/shapes --out ckpt/tdn --resume ckpt/admt_15k.pt --use_topo
(commandes exactes de tous les entraînements : train.ipynb et config.py)

Une itération :
  x0, c -> t ~ U{0..T-1}, ε ~ N(0, I) -> x_t = scheduler.add_noise(x0, ε, t)
  ε̂ = model(x_t, t, c) ; L_simple = ||ε̂ - ε||²
  si --use_topo : x̂₀ = (x_t - √(1-ᾱ_t) ε̂) / √ᾱ_t  (en fp32) ; L = L_simple + λ L_top(x̂₀, c)

Reprise : si --out contient déjà last.pt, l'entraînement reprend où il en était
(modèle, optimiseur, pas, état du RNG) ; sinon --resume initialise seulement les
poids (ex. TDN depuis ADM-T) et le compteur de pas part de 0.
TODO : logger le rapport des normes de gradient de L_simple et de λ L_top pour régler λ.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from data import MaskDataset
from model import build_unet
from topo_loss import topo_loss


def make_scheduler():
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )


def predict_x0(scheduler, x_t, eps_hat, t):
    """Eq. (2) de l'article."""
    alpha_bar = scheduler.alphas_cumprod.to(x_t.device)[t].view(-1, 1, 1, 1)
    return (x_t - torch.sqrt(1 - alpha_bar) * eps_hat) / torch.sqrt(alpha_bar)


def save_checkpoint(path, model, optimizer, step):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is None:
        return 0
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng_cpu"])
    if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
    return ckpt["step"]


def ckpt_step(path):
    """Pas d'entraînement atteint par un checkpoint."""
    return torch.load(path, map_location="cpu", weights_only=False)["step"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use_topo", action="store_true")
    p.add_argument("--lam", type=float, default=1e-5)
    p.add_argument(
        "--no_preserve", action="store_true", help="ablation : sans L_preserve"
    )
    p.add_argument(
        "--no_denoise", action="store_true", help="ablation : sans L_denoise"
    )
    p.add_argument(
        "--t_max_topo",
        type=int,
        default=1000,
        help="L_top seulement pour t < t_max_topo",
    )
    p.add_argument("--save_every", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    dataset = MaskDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    scheduler = make_scheduler()
    model = build_unet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    L_simple_crit = torch.nn.MSELoss()

    step = 0
    path = os.path.join(args.out, "last.pt")
    if os.path.exists(path):
        step = load_checkpoint(path, model, optimizer)
        print(f"reprise au pas {step}")
    elif args.resume:  # ex. TDN depuis ADM-T : poids seulement, pas 0
        load_checkpoint(args.resume, model)
        print(f"poids initialisés depuis {args.resume}")

    batches = infinite(loader)
    model.train()
    for step in range(step, args.steps):
        x0, c = next(batches)
        x0, c = x0.to(device), c.to(device)
        B = x0.shape[0]

        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
        epsilon = torch.randn_like(x0)
        xt = scheduler.add_noise(x0, epsilon, t)
        epsilon_hat = model(xt, t, class_labels=c).sample
        L_simple = L_simple_crit(epsilon_hat, epsilon)

        L_topo = torch.zeros((), device=device)
        topo_gap = 0.0
        mask = t < args.t_max_topo
        if args.use_topo and mask.any():
            x0hat = predict_x0(scheduler, xt[mask], epsilon_hat[mask].float(), t[mask])
            # x̂₀ ramené dans [-1, 1] comme les images : sinon L_preserve (-Σ persistance²)
            # n'est pas bornée et le modèle la fait diverger en gonflant x̂₀ (vu à λ = 1e-3)
            x0hat = x0hat.clamp(-1, 1)
            L_topo = topo_loss(
                x0hat, c[mask], not args.no_preserve, not args.no_denoise
            )
            # avec x̂₀ dans [-1, 1], L_top ≥ -4c par image (c composantes de persistance 2) :
            # L_top + 4·moyenne(c) ≥ 0 ne dépend plus des c tirés dans le batch, 0 = optimum
            if not args.no_preserve:
                topo_gap = L_topo.item() + 4 * c[mask].float().mean().item()

        loss = L_simple + args.lam * L_topo
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(
                f"pas {step:6d} | L_simple {L_simple.item():.4f} "
                f"| L_topo {L_topo.item():.4f} | écart à l'optimum {topo_gap:.3f}"
            )
        if (step + 1) % args.save_every == 0:
            # on sauvegarde step + 1 : c'est le prochain pas à exécuter
            save_checkpoint(path, model, optimizer, step + 1)

    save_checkpoint(path, model, optimizer, args.steps)


def infinite(loader):
    """Reparcourt le DataLoader indéfiniment (on compte en pas, pas en epochs)."""
    while True:
        for batch in loader:
            yield batch


if __name__ == "__main__":
    main()
