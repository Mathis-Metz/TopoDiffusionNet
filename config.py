"""Chemins et réglages partagés par les notebooks.

Un seul endroit pour les checkpoints comparés, les dossiers d'échantillons et
les options d'entraînement des ablations : train.ipynb entraîne, demo.ipynb et
ablations_limites.ipynb évaluent avec les mêmes noms.
"""

import os

CKPT_DIR = "ckpt"
SAMPLES_DIR = "samples"

# --------------------------------------------------------------------------
# Évaluation
# --------------------------------------------------------------------------
SEED = 0  # même graine pour tous les modèles -> mêmes bruits initiaux (comparaison appariée)
NUM_STEPS = 50  # pas DDIM
THRESHOLD = 0.0  # seuil de binarisation avant comptage
REF = "ADM-T 24k"  # référence de la comparaison appariée

# dossier d'échantillons d'un modèle -> autres dossiers où en chercher (ancien demo_eval.ipynb) ;
# indexé par dossier et non par nom : "TDN" désigne maintenant λ = 1e-3, dont ce ne sont pas les échantillons
EXTRA_SAMPLE_DIRS = {
    f"{SAMPLES_DIR}/tdn": ["resultats_eval/TDN"],
}

# une couleur par modèle, la même dans toutes les figures
COLORS = {"ADM-T 24k": "#2a78d6", "TDN λ=1e-3": "#eb6834", "TDN λ=1e-5": "#f3b590"}


def n_per_c(device):
    """Échantillons par valeur de c : moins sur CPU, où la génération est lente."""
    return 200 if device == "cuda" else 50


def first_existing(*paths):
    """Premier chemin qui existe (le dernier sinon, pour que le message d'erreur le nomme)."""
    return next((p for p in paths if os.path.exists(p)), paths[-1])


# --------------------------------------------------------------------------
# Entraînement (train.ipynb)
# --------------------------------------------------------------------------
# TDN : fine-tuning d'ADM-T 15k ; les ablations ajoutent leurs options à la fin
# (argparse garde la dernière valeur d'une option répétée)
TDN_ARGS = [
    "--use_topo",
    "--lam",
    1e-5,
    "--t_max_topo",
    500,
    "--steps",
    9_000,
    "--save_every",
    500,
]

# clé (dossier ckpt/abl/<clé>) -> (nom affiché, options ajoutées après TDN_ARGS)
ABLATIONS = {
    "preserve": ("L_preserve seul", ["--no_denoise"]),
    "denoise": ("L_denoise seul", ["--no_preserve"]),
    "lam1e-6": ("λ = 1e-6", ["--lam", 1e-6]),
    "lam1e-4": ("λ = 1e-4", ["--lam", 1e-4]),
    # à λ = 1e-5, |∇(λ L_top)| ≈ 0,2 % de |∇L_simple| (mesuré sur tdn_9k, t < 500) ; à 1e-3 : ≈ 20 %
    "lam1e-3": ("λ = 1e-3", ["--lam", 1e-3]),
    "tall": ("L_top pour tout t", ["--t_max_topo", 1000]),
}


# --------------------------------------------------------------------------
# Modèles évalués : nom affiché -> (checkpoint, dossier des échantillons)
# --------------------------------------------------------------------------
TDN_LAM1E3 = (f"{CKPT_DIR}/tdn_lam1e-3_9k.pt", f"{SAMPLES_DIR}/tdn_lam1e-3")
TDN_LAM1E5 = (
    first_existing(f"{CKPT_DIR}/tdn_9k.pt", f"{CKPT_DIR}/tdn/last.pt"),
    f"{SAMPLES_DIR}/tdn",
)


def main_models():
    """Les modèles comparés dans demo.ipynb, tous au même budget de 24 000 pas."""
    return {
        # ADM-T 15k continué sans L_top jusqu'à 24 000 pas (copie figée, sinon last.pt)
        "ADM-T 24k": (
            first_existing(
                f"{CKPT_DIR}/admt_suite_24k.pt", f"{CKPT_DIR}/admt_suite/last.pt"
            ),
            f"{SAMPLES_DIR}/admt_24k",
        ),
        "TDN λ=1e-3": TDN_LAM1E3,
        # TDN d'origine : L_top n'y pesait que ≈ 0,2 % du gradient de L_simple
        "TDN λ=1e-5": TDN_LAM1E5,
    }


def ablation_models():
    """Les variantes comparées dans ablations_limites.ipynb (avec ADM-T 24k et TDN en référence)."""
    models = main_models()
    out = {
        "ADM-T 24k": models["ADM-T 24k"],
        "TDN (λ=1e-5, t<500)": models["TDN λ=1e-5"],
    }
    for key, (label, _) in ABLATIONS.items():
        out[label] = (f"{CKPT_DIR}/abl/{key}/last.pt", f"{SAMPLES_DIR}/abl_{key}")
    return out


def keep_existing(models):
    """Retire (en le signalant) les modèles dont le checkpoint n'existe pas."""
    from train import ckpt_step

    kept = {}
    for name, (ckpt, out) in models.items():
        if os.path.exists(ckpt):
            kept[name] = (ckpt, out)
            print(f"{name:22s} : {ckpt} (pas {ckpt_step(ckpt)})")
        else:
            print(f"{name:22s} : checkpoint introuvable ({ckpt}), ignoré")
    return kept
