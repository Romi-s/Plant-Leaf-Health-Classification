# Plant-Leaf-Health-Classification

Binary classification of plant leaf images (healthy vs unhealthy) built with TensorFlow/Keras. This repo is a cleaned-up, modular split of an earlier notebook project, with a command-line trainer, model registry, and optional notebook/report for reference.

## Install
```bash
pip install -r requirements.txt
```

## Data
Put your dataset at `data/plant_dataset.npz` (default), or edit `DATA_PATH` in `src/config.py`. Images should be RGB of size 96×96; labels are mapped internally to {0,1}.

## Train (CLI)
```bash
# See options
python -m cli --help

# Train one model
python -m cli --mode single --name efficientnetb0 --eval --plot-history

# Train a subset
python -m cli --mode subset --names resnet50 inceptionv3 mobilenetv2 --eval

# Train all registered models
python -m cli --mode all
```
What happens: data is loaded/cleaned, splits+augmentation are built, the selected model(s) are trained with per‑model defaults from `config.py`.  
Optional flags print metrics and save simple plots.

> Tip: The custom CNN can be hyper-parameter tuned. Toggle `MODELS["cnn"].tuner.enabled = True` in `src/config.py` to enable Keras-Tuner search before the final fit.

## Models
- Transfer learning: `resnet50`, `efficientnetb0`, `inceptionv3`, `mobilenetv2`
- Custom: `cnn` (tuner‑ready)

## Project layout
- `src/cli.py` – command-line entry point (training modes & flags)
- `src/training.py` – small training API + model registry
- `src/models/*` – model builders & callbacks
- `src/data.py` – NPZ loading, label encoding, splits, augmentation
- `src/utils/*` – cleaning, metrics, viz, tuner helpers
- `src/config.py` – seeds, augmentation defaults, per-model training/tuning knobs
- `report.pdf` – short report summarizing the original notebook findings

- `notebook/compact_all.ipynb` (legacy notebook). The recommended path is the CLI above.

## Notes
- You supply the generators/datasets; the CLI builds standard pipelines from `config.py`.
- Override epochs/params via CLI or by editing each model’s block in `config.py`.

