# topodiffusion-mini

Réimplémentation à petite échelle de **TopoDiffusionNet** (Gupta, Samaras, Chen, ICLR 2025, [arXiv 2410.16646](https://arxiv.org/abs/2410.16646)), dans le cadre du cours de topologie du master MVA.

Un modèle de diffusion génère des masques binaires 64×64 contenant un nombre d'objets *c* donné (1 à 5). Conditionné seulement par *c* (**ADM-T**), il se trompe souvent sur le nombre d'objets. **TDN** ajoute une perte topologique **L_top** fondée sur l'homologie persistante de dimension 0 de l'image débruitée prédite x̂₀ : elle renforce les *c* composantes les plus persistantes et efface les autres.

On compare, au même budget d'entraînement (24 000 pas), ADM-T et TDN pour deux poids de la perte (λ = 1e-3 et 1e-5), en comptant les composantes connexes des échantillons générés.

## Fichiers

| | |
|---|---|
| `data.py` | génération du jeu de données (formes simples) |
| `model.py`, `train.py` | UNet conditionné par *c*, entraînement ADM-T / TDN |
| `topo_loss.py` | L_top (diagramme de persistance avec cripser) |
| `sample.py`, `evaluate.py` | échantillonnage DDIM, comptage et métriques |
| `config.py` | chemins, modèles comparés, réglages communs |
| `train.ipynb` | entraînement de tous les modèles (GPU, Colab) |
| `demo.ipynb` | évaluation : ADM-T contre TDN, exemples, x̂₀ au fil du débruitage |
| `topologie.ipynb` | homologie persistante et L_top sur des exemples jouets |
| `ablations_limites.ipynb` | ablations et limites de l'approche |

## Utilisation

`pip install -r requirements.txt`, puis `train.ipynb` pour entraîner et `demo.ipynb` pour évaluer (sur Colab : *Tout exécuter*).

## Référence

Gupta, Samaras, Chen. *TopoDiffusionNet: A Topology-aware Diffusion Model*. ICLR 2025. Code : https://github.com/Saumya-Gupta-26/TopoDiffusionNet
