# Watermark Localization

Sujet complet : [../Sujets/Watermark Localization.pdf](../Sujets/Watermark%20Localization.pdf)

## Objectif

Chaque image (512×512) a été watermarkée sur un **sous-rectangle** par l'une de ces trois méthodes post-hoc :
**PixelSeal**, **TrustMark** ou **StegaStamp**. Elle a ensuite été éditée. Pour chaque image de test, il faut prédire
un masque binaire : `1` pour la zone watermarkée, `0` pour la zone propre.

- Ratio d'aspect du rectangle : ≤ 16:1 pour PixelSeal et StegaStamp, ≤ 2:1 pour TrustMark.
- Types d'édition :
  - **outpainting** : l'image est agrandie et le watermark ne couvre plus qu'une partie ;
  - **retrait ou modification d'objet** : trou dans le watermark ;
  - **ajout d'objet** : trou dans le watermark ;
  - **transplantation** : un objet watermarké est collé dans une image propre.
- Ces trois méthodes décodent un message pour l'image entière et ne sont pas conçues pour la localisation, contrairement à
  Watermark Anything. Il n'y a probablement pas d'approche unique qui marche pour les trois.

## Score

mIoU sur 2 classes (watermark / propre), calculé **par image** puis moyenné. Toutes les images comptent autant, quelle
que soit la taille de l'édition.

- Évaluation sur une **grille 128×128** (une cellule = 4×4 pixels). Une cellule vaut 1 si au moins 50 % de ses pixels
  sont watermarkés. La fonction de sous-échantillonnage exacte est dans le starter kit.
- `IoU_c = |P_c ∩ G_c| / |P_c ∪ G_c|` (vaut 1 si l'union est vide), puis `mIoU_image = (IoU_0 + IoU_1) / 2`.
- Leaderboard public : 30 % des images de test. Classement final : les 70 % restants.

## Données

1 530 PNG, répartis à parts égales entre les trois méthodes.

- **train** : 510 images (170 par méthode), avec les masques vérité (512² et 128²), la méthode et le message embarqué.
- **test** : 1 020 images (340 par méthode), avec la méthode et le message embarqué.

Le starter kit contient le chargement en datasets PyTorch, les wrappers PixelSeal, TrustMark et StegaStamp
(`decode(images) -> bits` et accès aux feature maps intermédiaires), la bit accuracy, le downsampling 128×128,
l'évaluation mIoU locale sur train, et `submission_template.py`.

## Format de soumission

Un fichier `.pt` contenant `{"image_names": names, "masks": masks}` :

- `masks` : tenseur `[num_images, 128, 128]` à valeurs 0 ou 1, et `names[i]` correspond à `masks[i]` ;
- chaque image de test apparaît **exactement une fois**, sans doublon ;
- **≤ 20 Mo**.

## Démarrage

Le dataset HuggingFace est **à accès restreint** : il faut d'abord demander l'accès sur
<https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task> avec son compte HF, puis créer un token.

```bash
cd watermark_localization
export HF_TOKEN="hf_..."
# Tout le dataset (~570 Mo : starter_kit.zip, train.zip, test.zip) dans data/, ignoré par git
uvx --from huggingface_hub hf download SprintML/Hackathon_Watermark_Localisation_Task --repo-type dataset --local-dir data
unzip data/starter_kit.zip -d .          # code du starter kit, à commiter (petit)
```

Ensuite, suivre `data/README.txt` et le README du starter kit pour l'environnement et le chargement des données. Sur
JURECA, le script organisateur télécharge déjà ce dataset dans le dossier d'équipe.

## Références

- Souček et al., *Pixel Seal: Adversarial-only training for invisible image and video watermarking* — <https://arxiv.org/abs/2512.16874>
- Bui, Agarwal, Collomosse, *TrustMark*, ICCV 2025 — <https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html>
- Tancik, Mildenhall, Ng, *StegaStamp*, CVPR 2020 — <https://www.computer.org/csdl/proceedings-article/cvpr/2020/716800c114/1m3nYiyKXK0>
- Sander et al., *Watermark Anything with Localized Messages*, ICLR 2025 — <https://openreview.net/forum?id=IkZVDzdC8M>
