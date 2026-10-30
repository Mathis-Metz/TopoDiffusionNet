import argparse
import glob
import os

import numpy as np
import torch
from diffusers import DDIMScheduler

from config import EXTRA_SAMPLE_DIRS, NUM_STEPS, SEED, THRESHOLD
from data import C_VALUES
from evaluate import check_paired, predicted_counts
from model import build_unet
from train import ckpt_step, load_checkpoint, make_scheduler


def make_ddim_scheduler():
    """DDIM avec les mêmes β qu'à l'entraînement."""
    return DDIMScheduler.from_config(make_scheduler().config)


def load_model(ckpt, device):
    model = build_unet().to(device)
    load_checkpoint(ckpt, model)
    return model


@torch.no_grad()
def sample(model, scheduler, c, n, generator, num_steps=50, return_trajectory=False):
    """Renvoie n masques (n, 1, 64, 64) dans [-1, 1] conditionnés par c.

    Avec return_trajectory : renvoie aussi x̂₀ prédit à chaque pas,
    (n, num_steps, 1, 64, 64) sur CPU, et les pas de temps t correspondants.
    """
    model.eval()
    device = next(model.parameters()).device

    x = torch.randn((n, 1, 64, 64), generator=generator).to(device)
    labels = torch.full((n,), c, dtype=torch.long, device=device)
    scheduler.set_timesteps(num_steps)
    trajectory = []
    for t in scheduler.timesteps:
        epsilon = model(x, t, class_labels=labels).sample
        out = scheduler.step(epsilon, t, x)
        x = out.prev_sample
        if return_trajectory:
            trajectory.append(out.pred_original_sample.clamp(-1, 1).cpu())
    x = x.clamp(-1, 1)
    if return_trajectory:
        return x, torch.stack(trajectory, dim=1), scheduler.timesteps.cpu()
    return x


def generate(model, scheduler, n_per_c, seed=0, num_steps=50, verbose=True):
    images, labels = [], []
    for c in C_VALUES:
        generator = torch.Generator().manual_seed(seed + c)
        x = sample(model, scheduler, c, n_per_c, generator, num_steps=num_steps)
        images.append(x.cpu().numpy())
        labels.append(np.full(n_per_c, c))
        if verbose:
            print(f"c = {c} : {n_per_c} échantillons")
    return np.concatenate(images).astype(np.float32), np.concatenate(labels)


def first_per_c(data, n_per_c):
    """Indices des n_per_c premiers échantillons de chaque c (tous si n_per_c est None).

    Les bruits initiaux d'un c sont tirés d'un seul coup avec la graine seed + c :
    les n premiers d'un fichier plus grand sont donc ceux d'une génération de n.
    """
    c = data["c"]
    return np.concatenate([np.flatnonzero(c == v)[:n_per_c] for v in C_VALUES])


def cached_samples(
    name,
    ckpt,
    out,
    n_per_c,
    device,
    seed=0,
    num_steps=50,
    threshold=0.0,
    extra_dirs=(),
    force=False,
):
    """Échantillons d'un modèle : ceux déjà sur disque s'il y en a au moins n_per_c par c
    (on garde les n_per_c premiers), sinon générés et sauvegardés dans out/samples_n{n_per_c}.npz.

    Renvoie un dict : images (N, 1, 64, 64), c demandés, pred (comptages au seuil
    threshold), step (pas du checkpoint).
    """
    step = ckpt_step(ckpt)
    path = None
    if not force:
        found = [
            p
            for d in (out, *extra_dirs)
            for p in sorted(glob.glob(os.path.join(d, "samples*.npz")))
        ]
        sizes = {p: np.bincount(np.load(p)["c"]).max() for p in found}
        enough = [p for p in found if n_per_c is None or sizes[p] >= n_per_c]
        if enough:
            path = min(enough, key=sizes.get)  # le plus petit fichier suffisant
        elif found:
            print(
                f"{name} : échantillons existants trop peu nombreux ({max(sizes.values())} par c)"
            )
    if path is not None:
        data = np.load(path)
        print(f"{name} : échantillons existants ({path})")
        if "step" in data.files and int(data["step"]) != step:
            print(
                f"  attention : générés au pas {int(data['step'])}, le checkpoint est au pas {step} "
                f"(supprimer {path} pour régénérer)"
            )
    else:
        path = os.path.join(out, f"samples_n{n_per_c}.npz")
        print(f"{name} (pas {step}) : génération de {n_per_c} images par c...")
        os.makedirs(out, exist_ok=True)
        model = load_model(ckpt, device)
        images, c = generate(model, make_ddim_scheduler(), n_per_c, seed, num_steps)
        np.savez(path, images=images, c=c, step=step)
        print(f"{name} : sauvegardé dans {path}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        data = np.load(path)
    keep = first_per_c(data, n_per_c)
    images = data["images"][keep]
    return {
        "images": images,
        "c": data["c"][keep],
        "pred": predicted_counts(images, threshold),
        "step": step,
    }


def cached_samples_all(models, n_per_c, device, force=False):
    """cached_samples pour chaque modèle (nom -> (checkpoint, dossier)), réglages de config.py."""
    samples = {
        name: cached_samples(
            name,
            ckpt,
            out,
            n_per_c,
            device,
            SEED,
            NUM_STEPS,
            THRESHOLD,
            extra_dirs=EXTRA_SAMPLE_DIRS.get(out, []),
            force=force,
        )
        for name, (ckpt, out) in models.items()
    }
    check_paired(samples)
    return samples


def cached_trajectories(
    name,
    ckpt,
    out,
    n,
    device,
    seed=1000,
    num_steps=50,
    threshold=0.0,
    n_keep=4,
    force=False,
):
    """Trajectoires de x̂₀ (n par c), sauvegardées dans out/trajectory.npz.

    On ne garde que le nombre de composantes de x̂₀ à chaque pas, et les images
    des n_keep premières trajectoires (pour les figures).
    Renvoie (x0, counts, timesteps) : x0[c] (n_keep, steps, 1, 64, 64),
    counts[c] (n, steps), timesteps (steps,).
    """
    path = os.path.join(out, "trajectory.npz")
    if os.path.exists(path) and not force:
        print(f"{name} : trajectoires existantes ({path})")
    else:
        model = load_model(ckpt, device)
        scheduler = make_ddim_scheduler()
        to_save = {}
        for c in C_VALUES:
            generator = torch.Generator().manual_seed(seed + c)
            _, x0s, timesteps = sample(
                model,
                scheduler,
                c,
                n,
                generator,
                num_steps=num_steps,
                return_trajectory=True,
            )
            x0s = x0s.numpy()
            to_save[f"x0_c{c}"] = x0s[:n_keep].astype(np.float16)
            to_save[f"counts_c{c}"] = np.stack(
                [predicted_counts(x0s[:, s], threshold) for s in range(x0s.shape[1])],
                axis=1,
            )
            print(f"{name} : c = {c} fait")
        os.makedirs(out, exist_ok=True)
        np.savez(path, timesteps=timesteps.numpy(), threshold=threshold, **to_save)
        print(f"{name} : trajectoires sauvegardées dans {path}")
        del model
    data = np.load(path)
    x0 = {c: data[f"x0_c{c}"].astype(np.float32) for c in C_VALUES}
    counts = {c: data[f"counts_c{c}"] for c in C_VALUES}
    return x0, counts, data["timesteps"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_per_c", type=int, default=200)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_trajectory", action="store_true")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.ckpt, device)
    scheduler = make_ddim_scheduler()
    images, c = generate(model, scheduler, args.n_per_c, args.seed, args.steps)
    out = os.path.join(args.out, f"samples_n{args.n_per_c}.npz")
    np.savez(out, images=images, c=c, step=ckpt_step(args.ckpt))
    print(f"sauvegardé dans {out}")

    if args.save_trajectory:
        cached_trajectories(
            os.path.basename(args.out),
            args.ckpt,
            args.out,
            8,
            device,
            num_steps=args.steps,
            force=True,
        )


if __name__ == "__main__":
    main()
