"""Perte topologique L_top (dimension 0, filtration par sur-niveaux).

Étapes pour une image x̂₀ (tenseur (H, W) sur GPU, avec gradient) :
  1. f = -x̂₀ détaché sur CPU : cripser calcule une filtration par SOUS-niveaux,
     donc les sous-niveaux de -x̂₀ sont les sur-niveaux de x̂₀ ;
  2. cripser.computePH(f, maxdim=0) -> une ligne par paire ;
  3. pour chaque paire : m_p = pixel de naissance (maximum local de x̂₀),
     s_p = pixel de mort (col) ; composante essentielle : col = minimum global ;
  4. tri par persistance, P = c premières paires, Q = les autres ;
  5. L_top = -Σ_P (x̂₀[m_p] - x̂₀[s_p])² + Σ_Q (x̂₀[m_p] - x̂₀[s_p])²,
     valeurs lues en indexant le tenseur x̂₀ d'origine => différentiable.

Format de computePH et comportement de L_top vérifiés sur des exemples jouets
dans topologie.ipynb.
Attention : calculer x̂₀ et L_top en fp32 (√ᾱ_t est minuscule aux grands t).
"""

import numpy as np
import torch
import cripser


def persistence_pairs(img):
    """Paires (m_p, s_p, persistance) de dimension 0 pour les sur-niveaux de `img`.

    img : np.ndarray (H, W). Renvoie births (K, 2), deaths (K, 2), pers (K,),
    triées par persistance décroissante.
    """
    # une ligne par paire : [dim, birth, death, i_b, j_b, _, i_d, j_d, _]
    # (i, j) = (ligne, colonne) numpy ; valeurs de -img car filtration par sous-niveaux
    ph = cripser.computePH(-np.asarray(img, dtype=np.float64), maxdim=0)
    births = ph[:, 3:5].astype(np.int64)  # m_p : maximum local de img
    deaths = ph[:, 6:8].astype(np.int64)  # s_p : col de img
    essential = deaths[:, 0] < 0
    deaths[essential] = np.unravel_index(np.argmin(img), img.shape)
    pers = img[births[:, 0], births[:, 1]] - img[deaths[:, 0], deaths[:, 1]]
    order = np.argsort(-pers)
    return births[order], deaths[order], pers[order]


def topo_loss_single(x0_hat, c, use_preserve=True, use_denoise=True):
    """L_top pour une image x0_hat (H, W) et une contrainte c (int)."""
    births, deaths, _ = persistence_pairs(x0_hat.detach().cpu().numpy())
    diff = x0_hat[births[:, 0], births[:, 1]] - x0_hat[deaths[:, 0], deaths[:, 1]]
    S1 = (diff[:c] ** 2).sum()
    S2 = (diff[c:] ** 2).sum()
    loss = x0_hat.new_zeros(())
    if use_preserve:
        loss = loss - S1
    if use_denoise:
        loss = loss + S2
    return loss


def topo_loss(x0_hat, c, use_preserve=True, use_denoise=True):
    """Moyenne de L_top sur un batch : x0_hat (B, 1, H, W), c (B,)."""
    losses = [
        topo_loss_single(x0_hat[i, 0], int(c[i]), use_preserve, use_denoise)
        for i in range(x0_hat.shape[0])
    ]
    return torch.stack(losses).mean()
