"""Évaluation du respect de la contrainte c.

python evaluate.py --samples samples/tdn

Pour chaque échantillon : seuiller à 0, compter les composantes connexes,
comparer au c demandé. Métriques par c et moyennes :
  - accuracy (comptage exact) ;
  - précision et F1, en traitant chaque valeur de c comme une classe
    (à confronter à analysis-scripts/eval-metrics.py des auteurs) ;
  - erreur absolue moyenne |ĉ - c| et répartition sur-comptage / sous-comptage.
Le contrôle de qualité des formes (aires, petits fragments) est dans
ablations_limites.ipynb.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from data import count_components


def find_cached(n_per_c, *dirs):
    """Fichier d'échantillons présent dans l'un des dossiers (de préférence avec n_per_c), ou None.

    Accepte samples.npz (ancien format) comme samples_n{N}.npz.
    """
    found = [
        p for d in dirs for p in sorted(glob.glob(os.path.join(d, "samples*.npz")))
    ]
    preferred = [p for p in found if p.endswith(f"_n{n_per_c}.npz")]
    return (preferred or found or [None])[0]


def predicted_counts(samples, threshold=0.0):
    """samples : (N, 1, H, W) dans [-1, 1] -> nombre de composantes par image."""
    binary_samples = np.asarray(samples) > threshold
    return np.array([count_components(img[0]) for img in binary_samples])


def metrics(c_true, c_pred):
    """Renvoie un dict : accuracy, précision, F1, MAE, globalement et par c."""
    c_true, c_pred = np.asarray(c_true), np.asarray(c_pred)
    per_c = {}
    for c in np.unique(c_true):
        is_true, is_pred = c_true == c, c_pred == c
        tp = np.sum(is_true & is_pred)
        precision = tp / is_pred.sum() if is_pred.any() else 0.0
        recall = tp / is_true.sum()  # = accuracy restreinte aux images demandées avec c
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        err = c_pred[is_true] - c
        per_c[int(c)] = {
            "n": int(is_true.sum()),
            "accuracy": float(recall),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mae": float(np.abs(err).mean()),
            "over": float((err > 0).mean()),  # sur-comptage
            "under": float((err < 0).mean()),  # sous-comptage
        }

    err = c_pred - c_true
    return {
        "accuracy": float((err == 0).mean()),
        "precision": float(np.mean([m["precision"] for m in per_c.values()])),
        "recall": float(np.mean([m["recall"] for m in per_c.values()])),
        "f1": float(np.mean([m["f1"] for m in per_c.values()])),
        "mae": float(np.abs(err).mean()),
        "over": float((err > 0).mean()),
        "under": float((err < 0).mean()),
        "per_c": per_c,
    }


COLUMNS = {
    "accuracy": "accuracy",
    "precision": "précision",
    "f1": "F1",
    "mae": "MAE",
    "over": "sur-comptage",
    "under": "sous-comptage",
}


def tables(samples):
    """samples : nom -> dict(c, pred, step).

    Renvoie le tableau des moyennes (une ligne par modèle) et l'accuracy par c
    (une colonne par modèle).
    """
    results = {name: metrics(s["c"], s["pred"]) for name, s in samples.items()}
    summary = pd.DataFrame(
        {
            name: {"pas": samples[name]["step"], **{COLUMNS[k]: r[k] for k in COLUMNS}}
            for name, r in results.items()
        }
    ).T
    summary["pas"] = summary["pas"].astype(int)
    per_c = pd.DataFrame(
        {
            name: {c: m["accuracy"] for c, m in r["per_c"].items()}
            for name, r in results.items()
        }
    )
    per_c.index.name = "c"
    return summary, per_c


def check_paired(samples):
    """La comparaison appariée suppose le même nombre d'échantillons pour tous les modèles."""
    sizes = {name: len(s["c"]) for name, s in samples.items()}
    if len(set(sizes.values())) > 1:
        print(
            "attention : nombres d'échantillons différents",
            sizes,
            "-> comparaison appariée non valable (supprimer les fichiers concernés pour régénérer)",
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()

    # format écrit par sample.py : images (N, 1, H, W) dans [-1, 1], c (N,) demandés
    data = np.load(find_cached(None, args.samples))
    c_true = data["c"]
    c_pred = predicted_counts(data["images"], args.threshold)
    res = metrics(c_true, c_pred)

    header = f"{'c':>7} {'n':>5} {'acc':>6} {'prec':>6} {'f1':>6} {'mae':>6} {'sur':>6} {'sous':>6}"
    print(header)
    print("-" * len(header))
    for c, m in res["per_c"].items():
        print(
            f"{c:>7} {m['n']:>5} {m['accuracy']:6.3f} {m['precision']:6.3f} "
            f"{m['f1']:6.3f} {m['mae']:6.3f} {m['over']:6.3f} {m['under']:6.3f}"
        )
    print("-" * len(header))
    print(
        f"{'moyenne':>7} {len(c_true):>5} {res['accuracy']:6.3f} {res['precision']:6.3f} "
        f"{res['f1']:6.3f} {res['mae']:6.3f} {res['over']:6.3f} {res['under']:6.3f}"
    )

    out = os.path.join(args.samples, "metrics.json")
    with open(out, "w") as f:
        json.dump({"threshold": args.threshold, **res}, f, indent=2)
    print(f"\nmétriques sauvegardées dans {out}")


if __name__ == "__main__":
    main()
