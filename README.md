(Update: All files required to reproduce the project are available including .h5 files. Extra files have been redacted for privacy reasons)
# Chest X-Ray Deep Learning

A collection of scripts, models and notebooks for chest X-ray classification tasks (cardiomegaly, pneumonia, lung cancer, mass detection). This repository bundles training scripts, pretrained model files, example prediction code, and exploratory notebooks used during development.

## What this project does

- Provides model training and inference scripts for several chest X-ray tasks:
  - Cardiomegaly classification (see `mainC.py`, `cardiomegaly.h5`, `cardiomegaly/`)
  - Pneumonia detection (see `mainP.py`)
  - Lung cancer classification (see `mainL.py`)
- Includes example notebooks used for data analysis and model experiments

## Why this is useful

- Useful as a research / teaching collection showing end-to-end image classification pipelines on chest X-ray datasets.
- Contains both training scripts and saved model weights so you can reproduce experiments or run inference quickly.

## Repository layout (high level)

- `Images/data/` — dataset split by `train/`, `val/`, `test/` and class subfolders (used by Keras ImageDataGenerator).
- `cardiomegaly/` — additional models, notebooks and a `ChestXRay_Medical_Diagnosis_Deep_Learning.ipynb` notebook.
- `lung_cancer/` — lung cancer model and data layout.
- `mass/` — scripts and CSVs for mass detection experiments.
- `pneumonia/` — pneumonia model and scripts.
- `extra/` — earlier experiments and notebooks.

## Quick start

Prerequisites

- Python 3.8+ (tested with 3.8–3.11)
- Recommended packages: tensorflow (or tensorflow-cpu), keras, opencv-python, numpy, matplotlib, scikit-learn, pandas, pillow

Create a virtual environment and install typical dependencies (adjust as needed):

```powershell
# Windows PowerShell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow opencv-python matplotlib scikit-learn pandas pillow
```

If this repo contains a `requirements.txt` file later, prefer:

```powershell
pip install -r requirements.txt
```

Running an example inference (cardiomegaly)

1. Make sure you have the model `cardiomegaly.h5` at the repository root (or produce one by running training scripts).
2. Run the example prediction script in `mainC.py`:

```powershell
python mainC.py
```

`mainC.py` contains training code and an example inference block which loads `cardiomegaly.h5` and runs a prediction on an example image (path: `Images/data/test/pcardiomegaly/00000032_027.png` in the repo). Adjust the image path or script to point to your image for inference.

Other scripts / entry points

- `pneumonia/mainP.py` — (if present) pneumonia training/inference scripts and saved weights under `pneumonia/`.
- `lung_cancer/mainL.py` — lung cancer training and saved models in `lung_cancer/`.
- `mass/mainM.py` — mass detection dataset and scripts in `mass/`.

Check those folders for notebooks (`.ipynb`) and saved model files (`*.h5`, `*.weights.h5`).

## How to reproduce training

- Each folder contains code used to train a model. Typical steps:
  1. Prepare data under `Images/data/` or the folder-specific `data/` subfolders.
  2. Create and activate a Python virtual environment.
  3. Install dependencies.
  4. Run the training script (for example `python mainC.py` for cardiomegaly). Training hyperparameters are in the scripts.

Note: scripts in the repo are a mix of experiments. Inspect the scripts for hard-coded paths, hyperparameters, and dataset locations before running on your data.

## Maintainers & contributing

If you'd like to contribute:

- Submit pull requests with focused changes (bugfixes, documentation, small experiments).
- If you add or change dependencies, include or update `requirements.txt`.
- Add step-by-step run instructions or small unit tests for new scripts.

For internal prompts and repo metadata see `.github/prompts/`.
