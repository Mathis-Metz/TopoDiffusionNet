"""UNet de débruitage conditionné par le nombre d'objets c.

Le plongement de c est ajouté au plongement du pas de temps t, comme dans
l'article (Nichol & Dhariwal 2021) : `num_class_embeds` de diffusers, c traité
comme une classe discrète (table d'embedding). L'indice 0 est réservé à une
classe « nulle », pour une éventuelle classifier-free guidance.
"""

import torch
from diffusers import UNet2DModel


def build_unet(num_classes=6, image_size=64):
    """UNet2DModel d'environ 10-20 M de paramètres, 1 canal en entrée et en sortie.

    num_classes = 5 valeurs de c + 1 classe nulle (pour la CFG éventuelle).
    """
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        # résolutions 64 -> 32 -> 16 -> 8 ; attention seulement en 16x16 (coûteuse sinon)
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        # c ∈ {1, ..., 5} sert directement d'indice ; l'indice 0 est la classe nulle
        num_class_embeds=num_classes,
    )
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test rapide : une passe avant sur un batch aléatoire.
    model = build_unet()
    print(f"{count_parameters(model) / 1e6:.1f} M paramètres")
    x = torch.randn(4, 1, 64, 64)
    t = torch.randint(0, 1000, (4,))
    c = torch.randint(1, 6, (4,))
    print(model(x, t, class_labels=c).sample.shape)  # attendu : (4, 1, 64, 64)
