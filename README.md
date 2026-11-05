# CURE: Visual Reprogramming of Vision-Language Models under Limited Supervision
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2.0) 
[![Static Badge](https://img.shields.io/badge/View-Poster-purple)](https://neurips.cc/virtual/2026/loc/sydney/poster/155870)
[![Static Badge](https://img.shields.io/badge/Pub-NeurIPS'2026-blue)](https://openreview.net/forum?id=2H603JxvpD#discussion)

Official PyTorch code for CURE, visual reprogramming baselines, and semi-supervised
learning (SSL) comparisons with frozen vision-language models.

## Authors

**Lingzhi Wang**<sup>1</sup>, **Wei Wang**<sup>2,3*</sup>, **Xinyang Chen**<sup>1*</sup>

<sup>1</sup> `School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen)`  
<sup>2</sup> `The Qingdao Research Institute, Harbin Institute of Technology (Weihai)`  
<sup>3</sup> `Shandong Key Laboratory of Industrial Network Security`  
\* Corresponding author


## Environment and installation

The archived CURE runs used the following environment:

| Component | Version |
| --- | --- |
| Python | 3.10.0 |
| PyTorch | 2.9.0+cu128 |
| TorchVision | 0.24.0+cu128 |
| CUDA runtime | 12.8 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Pillow | 11.3.0 |


Run these commands from the repository root:

```bash
conda create -n cure python=3.10 -y
conda activate cure
python -m pip install torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

`environment.yml` is also provided for Conda. Install the CUDA-specific
PyTorch wheels above after creating that environment if necessary.
CLIP is the default backbone and does not require `transformers` or an API key.
For the optional SigLIP/SigLIP2 branches, install
`requirements-siglip.txt` and supply `--model-name` with a compatible model ID
or local model directory. The main reproduction protocol uses CLIP ViT-B/16.

## Repository structure

```text
.
├── README.md                  # Installation, datasets, commands, protocols
├── environment.yml            # Conda environment
├── requirements.txt           # Pinned CLIP training dependencies
├── requirements-siglip.txt     # Optional SigLIP dependencies
├── cfg.py                     # Dataset root from CURE_DATA_ROOT
├── tools.py                   # Text features, logits, seeds, data loaders
├── experiments/               # Training and evaluation entry points
│   ├── fs_cure.py             # Final CURE training and checkpoint evaluation
│   ├── fs_vp.py               # VR: watermarking visual prompt
│   ├── fs_ar.py               # AR: padding visual reprogramming
│   ├── fs_attrvr.py           # Supervised AttrVR
│   ├── fs_zs.py               # Zero-shot baseline
│   ├── fs_ssl.py              # AttrVR + five SSL objectives
│   ├── fs_ar_fixmatch.py      # Local AR + FixMatch implementation
│   ├── fs_ar_flexmatch.py     # Local AR + FlexMatch implementation
│   ├── fs_ar_attrfixmatch.py  # Local attribute-assisted AR + FixMatch
│   ├── fs_attrvr_mixmatch.py  # Local AttrVR + MixMatch implementation
│   └── fs_attrvr_refixmatch.py # Local AttrVR + ReFixMatch implementation
├── methods/                   # Model components and SSL objectives
│   ├── vp.py                  # PaddingVR and WatermarkingVR
│   ├── ssl_core.py            # FixMatch/SoftMatch/CGMatch/FlexMatch/FreeMatch
│   ├── backbones.py           # Portable frozen backbone loading
│   └── runtime.py             # Baseline output configuration
├── datasets/                  # Dataset readers, split loading, and SSL wrappers
├── attributes/gpt3/           # Descriptive/distinctive attributes (24 JSONs)
├── splits/                    # Fixed data partitions for reproducible experiments
└── scripts/                   # Installation checks and benchmark launching
    ├── run_benchmark.py       # Reproduction commands and sequential launcher
    └── check_install.py       # Environment, metadata, loss/gradient checks

```

`splits/` provides the predefined 2-shot data partitions for 12 datasets and
three random seeds (0, 42, and 2025). It contains relative image paths and
labels for the labeled, unlabeled, and test sets, organized by dataset,
with an `index.json` summary covering all 36 splits. Dataset images are
downloaded separately.

## Dataset preparation

Download the datasets following the
[CoOp dataset preparation guide](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).
RESISC45 preparation is also described in
[BlackVIP](https://github.com/changdaeoh/BlackVIP).
Set the dataset root without editing source files:

```bash
export CURE_DATA_ROOT=/path/to/datasets
```

The default is `data/` under this repository. Expected image directories are:

| `--dataset` | Directory under the data root |
| --- | --- |
| `caltech101` | `caltech101/101_ObjectCategories/` |
| `dtd` | `dtd/dtd/images/` |
| `eurosat` | `eurosat/2750/` |
| `fgvc` | `fgvc-aircraft-2013b/data/images/` |
| `food101` | `food-101/images/` |
| `oxford_flowers` | `oxford_flowers/jpg/` |
| `oxford_pets` | `oxford-iiit-pet/images/` |
| `stanford_cars` | `stanford_cars/cars_train/`, `stanford_cars/cars_test/` |
| `sun397` | `sun/images/` |
| `ucf101` | `ucf101/UCF-101-midframes/` |
| `resisc45` | `resisc45/data/` |
| `imagenet` | `ImageNet/data/train/`, `ImageNet/data/val/` |

For 2-shot runs with seeds **0, 42, 2025**, the loader automatically uses the
bundled `splits/<dataset>/shot_2-seed_<seed>.json.gz` membership. These files
contain relative image paths and labels, retaining the recorded labeled,
unlabeled, and test order. They remove the need for local pickle caches or
randomly regenerated dataset partitions. Verify an installed dataset with:

```bash
python scripts/check_install.py --dataset dtd
```

Other shot counts or seeds use the original dataset readers, which require
the annotations and split files described by CoOp, including class-name
metadata for Flowers and ImageNet. They create their own few-shot caches.
Do not reuse pickle caches containing paths from another machine.

**ImageNet protocol:** the archived experiment uses a local **998-class
subset**, excluding `n04008634` and `n04355933` (See this [issue](https://github.com/caichengyi/AttrVR/issues/1)). Its training pool is also a
subset of the full ImageNet training set. The bundled membership specifies
the exact images; this result is not a standard full ImageNet-1K result.

## Reproduce CURE

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/fs_cure.py \
  --dataset dtd --seed 0 \
  --output results/cure/dtd/seed_0
```

The final protocol is the default:

| Setting | Value |
| --- | --- |
| Backbone | Frozen CLIP ViT-B/16 |
| Labeled samples | 2 per class |
| Attribute fusion | `alpha=0.5` |
| Input / padded output | 192 / 224 pixels |
| Labeled batch / unlabeled ratio | 16 / 7 |
| Epochs | 200 |
| Visual prompt / text adapter LR | 40 / `3e-4` |
| Optimizer | SGD, momentum 0.9, weight decay 0 |
| Scheduler | Cosine, updated each optimization step |
| Unlabeled loss weight | 1 |
| Probability/margin statistics EMA | 0.999 |
| Strong augmentation | Random erasing, probability 1 |
| Alignment, Gaussian, margin, attribute consistency | Enabled |

`ema_p` controls running statistics; it is not a separate EMA teacher model.
The margin is the raw top-two probability gap, without batch-sum or
labeled/unlabeled ratio normalization. `TextMLP` is an identity-initialized
linear feature adapter.

## Acknowledgements

We thank the authors of the following open-source projects for making their
code and resources publicly available:

- [AttrVR](https://github.com/tmlr-group/AttrVR): attribute-based visual reprogramming and attribute resources.
- [CoOp](https://github.com/KaiyangZhou/CoOp): dataset readers and dataset preparation guidelines.
- [OpenAI CLIP](https://github.com/openai/CLIP): pretrained vision-language models and text tokenization.
- [USB](https://github.com/microsoft/Semi-supervised-learning): semi-supervised learning objectives and reference implementations.
- [CGMatch](https://github.com/BoCheng-96/CGMatch): count-gap-based semi-supervised learning.

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{wang2026cure,
  title     = {{CURE}: Visual Reprogramming of Vision-Language Models under Limited Supervision},
  author    = {Lingzhi Wang and Wei Wang and Xinyang Chen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=2H603JxvpD}
}
```
