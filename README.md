# Plant-Leaf-Health-Classification

Binary classification of plant leaf images (healthy vs unhealthy) built with TensorFlow/Keras. This repo is a cleaned-up, modular split of an earlier notebook project, with a command-line trainer, model registry, and optional notebook/report for reference.

## Install

```bash
pip install -r requirements.txt
```

## Data
Put your dataset at `data/plant_dataset.npz` (default), or edit `DATA_PATH` in `src/config.py`. Images should be RGB of size 96×96; labels are mapped internally to {0,1}.

## Train (CLI)

# Single model
python -m src.cli --mode single --name efficientnetb0 --plot-history --eval

# Subset of models
python -m src.cli --mode subset --names resnet50 inceptionv3 mobilenetv2 --eval

# All registered models (as defined in config)
python -m src.cli --mode all

## What it does
- loads/inspects data, runs outlier cleaning, builds splits + augmentation
- trains the chosen model(s) using per-model defaults from `src/config.py`
- optional: plots history (`--plot-history`), prints metrics (`--eval`)

Tip: The custom CNN can be hyper-parameter tuned. Toggle `MODELS["cnn"].tuner.enabled = True` in `src/config.py` to enable Keras-Tuner search before the final fit.

## Models
Transfer learning: resnet50, efficientnetb0, inceptionv3, mobilenetv2

Custom: cnn (tuner-ready)

## Project layout
- `src/cli.py` – command-line entry point (training modes & flags)
- `src/training.py` – small training API + model registry
- `src/models/*` – model builders & callbacks
- `src/data.py` – NPZ loading, label encoding, splits, augmentation
- `src/utils/*` – cleaning, metrics, viz, tuner helpers
- `src/config.py` – seeds, augmentation defaults, per-model training/tuning knobs
- `report.pdf` – short report summarizing the original notebook findings

## Optional notebook
`notebook/compact_all.ipynb` (legacy, for exploration/reading). The recommended path is the CLI above.

