"""Données synthétiques : masques binaires 64x64 contenant c objets.

- Shapes : cercles, carrés, triangles qui ne se touchent pas (jeu principal).
- Cellules jointives : ellipses qui peuvent se toucher (extension, pour montrer
  la limite du comptage par composantes connexes).

Les images sont sauvegardées en PNG sous le nom `{c}_{index:05d}.png`
(convention du dossier dataset-format/ des auteurs).
"""

import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import Dataset

SIZE = 64
C_VALUES = range(1, 6)
FORM = ["triangle", "circle", "rectangle"]

# --------------------------------------------------------------------------
# Génération
# --------------------------------------------------------------------------
MIN_SIZE, MAX_SIZE = 6, 24
MAX_TRIES = 100


def draw_shape(mask, kind, pos):
    """Dessine une forme pleine (kind ∈ {"circle", "square", "triangle"}) dans `mask`."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    if kind == "triangle":
        # Un pixel est dans le triangle s'il est du même côté des trois arêtes.
        (x0, y0), (x1, y1), (x2, y2) = pos
        d0 = (x1 - x0) * (yy - y0) - (y1 - y0) * (xx - x0)
        d1 = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
        d2 = (x0 - x2) * (yy - y2) - (y0 - y2) * (xx - x2)
        inside = ((d0 >= 0) & (d1 >= 0) & (d2 >= 0)) | (
            (d0 <= 0) & (d1 <= 0) & (d2 <= 0)
        )
    elif kind == "circle":
        cx, cy, r = pos
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    elif kind == "rectangle":
        (xa, ya), (xb, yb) = pos
        inside = (xx >= xa) & (xx <= xb) & (yy >= ya) & (yy <= yb)
    else:
        raise ValueError(kind)
    mask |= inside
    return mask


def draw_size(kind, rng):
    if kind == "triangle":
        while True:
            s = rng.integers(MIN_SIZE, MAX_SIZE + 1)
            corner = rng.integers(0, SIZE - s, 2)
            pos = corner + rng.integers(0, s + 1, (3, 2))
            if np.linalg.matrix_rank(np.c_[pos, np.ones(3)]) == 3:
                area = abs(np.linalg.det(np.c_[pos, np.ones(3)])) / 2
                if area >= MIN_SIZE**2:
                    return pos
    if kind == "circle":
        radius = rng.integers(MIN_SIZE // 2, MAX_SIZE // 2 + 1)
        center = rng.integers(radius, SIZE - radius, 2)
        return np.array([*center, radius])
    if kind == "rectangle":
        while True:
            w, h = rng.integers(MIN_SIZE, MAX_SIZE + 1, 2)
            corner = rng.integers(0, [SIZE - w, SIZE - h])
            pos = np.array([corner, corner + [w - 1, h - 1]])
            return pos


def test_overlap(shape, mask):
    """True si `shape` touche (ou est à 1 pixel de) une forme déjà dans `mask`."""
    grown = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool))
    return bool((shape & grown).any())


def make_shapes_image(c, rng):
    """Renvoie un masque bool (SIZE, SIZE) avec exactement c formes disjointes.

    Idée : tirer forme, taille, position ; refuser si la forme touche une forme
    déjà placée (dilater le masque existant d'un pixel pour garder un espace).
    """
    while True:  # si une forme ne trouve pas de place, on recommence l'image
        forms = rng.integers(0, len(FORM), c)
        mask = np.zeros([SIZE, SIZE], bool)
        for form in forms:
            for _ in range(MAX_TRIES):
                size = draw_size(FORM[form], rng)
                shape = draw_shape(np.zeros_like(mask), FORM[form], size)
                if count_components(shape) == 1 and not test_overlap(shape, mask):
                    mask |= shape
                    break
            else:
                break
        else:
            return mask


def make_cells_image(c, rng):
    """Renvoie un masque bool (SIZE, SIZE) avec c ellipses qui peuvent se toucher.

    Contrairement à Shapes, aucun espace n'est imposé : deux cellules jointives
    forment une seule composante connexe, donc le comptage peut donner moins que c.
    """
    yy, xx = np.mgrid[:SIZE, :SIZE]
    mask = np.zeros([SIZE, SIZE], bool)
    for _ in range(c):
        # demi-axes de 3 à 7 px : avec c = 5, environ 70 % des images ont des cellules jointives
        a, b = rng.uniform(3, 7, 2)
        theta = rng.uniform(0, np.pi)
        cy, cx = rng.uniform(8, SIZE - 8, 2)
        u = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
        v = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
        mask |= (u / a) ** 2 + (v / b) ** 2 <= 1
    return mask


def count_components(mask):
    """Nombre de composantes connexes d'un masque binaire (connexité 4 par défaut)."""
    _, n = ndimage.label(mask)
    return n


# --------------------------------------------------------------------------
# Dataset PyTorch
# --------------------------------------------------------------------------
class MaskDataset(Dataset):
    """Lit les PNG d'un dossier ; renvoie (image dans [-1, 1] de forme (1, 64, 64), c)."""

    def __init__(self, root):
        self.paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".png")
        )
        # c est lu dans le nom du fichier `{c}_{index:05d}.png`
        self.counts = [int(os.path.basename(p).split("_")[0]) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        mask = np.array(Image.open(self.paths[i]).convert("L")) > 127
        x = torch.from_numpy(mask).float().unsqueeze(0) * 2 - 1  # {0, 1} -> {-1, 1}
        return x, self.counts[i]


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/shapes")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--kind", choices=["shapes", "cells"], default="shapes")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    c_values = list(C_VALUES)
    for index in range(args.n):
        c = c_values[index % len(c_values)]  # équilibré entre les valeurs de c
        if args.kind == "shapes":
            mask = make_shapes_image(c, rng)
            assert count_components(mask) == c
        else:  # c = nombre d'ellipses, pas forcément de composantes
            mask = make_cells_image(c, rng)
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            os.path.join(args.out, f"{c}_{index:05d}.png")
        )
