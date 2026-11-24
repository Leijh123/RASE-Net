# RASE-Net

**Rare-organ compensation and anatomical-prior refinement for interactive abdominal multi-organ segmentation.**

RASE-Net combines training-stage rare-organ compensation (TRC), semantic anatomical contradiction refinement (SCAR), and exemplar-guided residual correction (EGRC). The stages address supervision imbalance, local prediction errors, and residual shape discrepancies, respectively.

**[中文说明](docs/README_zh.md) · [Data preparation](docs/data.md) · [Training](docs/training.md) · [Evaluation](docs/evaluation.md) · [Code versions](docs/versions.md)**

## Method

```text
Training images + masks ── TRC (AAECA + ISRFOS) ── IMIS-Net
                                                    │
Image + box + organ text ────────────────────────── P₀
                                                    │
                   SCAR: SACF → field-guided refiner ─┤
                                                    │
Training-only exemplar bank → retrieve → align → EGRC → prediction
```

The exemplar bank uses training annotations only. Query inference uses the image, supported prompts, and model predictions; query segmentation labels are used for evaluation, not as refinement inputs. GT-box experiments explicitly derive the box from the target annotation.

## Repository

```text
rase/
  models/              # SACF, field-guided refiner, state-aware EGRC
  losses/              # SACF supervision
  trc/                 # Anatomy-anchored augmentation and frequency weights
  retrieval/           # Joint retrieval and centroid translation
  evaluation/          # Archived metric implementation
  _historical/         # Self-contained historical IMIS/TRC/SCAR/EGRC runtime
  cli.py               # Unified stage launcher
configs/               # Explicit, editable stage arguments
docs/                  # Data, commands, evaluation and version notes
experiments/           # Internal-ablation source reference, kept separate
tests/                 # Source integrity, CLI and model checks
provenance/            # File-level source mapping and hashes
```

## Installation

Use Python 3.10 or 3.11. Install a compatible PyTorch/torchvision pair for your CUDA driver, then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m rase list
python -m unittest discover -s tests -v
```

The original IMIS text encoder loads tokenizer files from `pretrained/clip-vit-base-patch32` relative to the working directory. Obtain model/tokenizer assets from their upstream distributors; no pretrained files are bundled.

## Training and evaluation

The launcher exposes the existing stage scripts without flattening their dependencies or changing their optimization logic:

```bash
python -m rase sacf -- --help
python -m rase sacf --config configs/btcv.example.json --dry-run
python -m rase sacf --config configs/btcv.example.json
python -m rase refiner --config configs/btcv.example.json
python -m rase bank --config configs/btcv.example.json
python -m rase cache --config configs/btcv.example.json
python -m rase egrc --config configs/btcv.example.json
```

Set data and checkpoint paths before executing. A full stage dependency graph and standalone commands are in [Training](docs/training.md). Configuration files are examples, not a claim that one newly assembled recipe reproduces every historical paper result.

## Implementation versions

`rase.models.egrc` provides the state-aware EGRC implementation used in the recent matched component experiments. The historical stage launcher retains the earlier EGRC implementation required by historical checkpoints. Their gate and positional parameters differ; **do not interchange checkpoints**. See [Code versions](docs/versions.md) before choosing a runtime.

## Data and release scope

BTCV, AMOS MRI, and WORD data must be obtained under their respective terms. This repository includes neither patient data, experiment caches, nor trained weights. Dataset examples document the existing splits without regenerating them. Reorganized source and CPU checks do not constitute a new end-to-end training reproduction.

## Acknowledgements and licensing

The baseline builds on [IMIS-Bench](https://github.com/uni-medical/IMIS-Bench) and its Segment Anything components. Preserve upstream notices and follow their licenses. See [Third-party notices](THIRD_PARTY_NOTICES.md). A project-wide distribution license has not been assigned automatically.
