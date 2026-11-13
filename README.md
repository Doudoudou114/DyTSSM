# DyTSSM: Dynamic Topology-guided State Space Modeling for Hyperspectral Change Detection

DyTSSM is a dual-temporal hyperspectral change detection framework. This
repository contains the public training, evaluation, data processing, and
selective scan components. The core model implementation is not included in
this public release.

## Files

- `main.py`: training and evaluation entry point
- `selective_scan.py`: self-contained selective scan implementation
- `load_data.py`: dataset loading, PCA, patch generation, and visualization
- `train_test.py`: training, evaluation, and prediction utilities

## Data Layout

Place the data under the repository-level `dataset` directory:

```text
dataset/
  Hermiston/
  farm/
  river/
  bayArea/
  santaBarbara/
```

The default data root is `../dataset` relative to `DyTSSM`. A different path can
be supplied with `--data-root`.

## Install

```bash
pip install -r requirements.txt
```

## Train

Add the DyTSSM model implementation as `model.py` before running training.

```bash
cd DyTSSM
python main.py --dataset hermiston
```

Available dataset names are `hermiston`, `farmland`, `river`, `bayArea`, and
`santaBarbara`.

Common options:

```bash
python main.py \
  --dataset hermiston \
  --epochs 25 \
  --batch-size 64 \
  --pca-channels 32 \
  --patch-size 5 \
  --d-model 64 \
  --label-ratio 0.01
```

Results are written to `DyTSSM/output/<dataset>/`.
