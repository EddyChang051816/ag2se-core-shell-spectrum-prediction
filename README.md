# Ag2Se Core and Core-Shell Spectrum Prediction

This project predicts complete Ag2Se core and core-shell spectra at specified concentration ratios or pH values using spectra measured under known conditions.

Python 3.10 or later is required.

## Model Design

- Leave-one-spectrum-out cross-validation selects the amplitude, peak-position, and full width at half maximum (FWHM) strategies, along with the XGBoost hyperparameters.
- Spectral preprocessing and a physics-informed prior are combined with XGBoost residual learning.
- The pipeline supports both normalized and raw-scale spectral outputs, as well as a separate model-evaluation workflow.

## Method

1. Apply median filtering, Savitzky-Golay smoothing, and interpolation onto a common wavelength grid to the training spectra.
2. Estimate the peak position, amplitude, baseline, and FWHM using only the training data.
3. Build a quality-weighted, peak-aligned physics-informed prior.
4. Use XGBoost to learn the leave-one-out prior residuals. The residual model is enabled only when it outperforms the prior during training-side cross-validation.
5. Apply the FWHM strategy selected by training-side cross-validation and generate both normalized and raw-scale predictions.

Latin hypercube sampling (LHS) is used for cross-validated hyperparameter search. For a faster run, set `--lhs-samples 0` to use fixed parameters.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Directory Structure

The data is not included in this repository. `--data-root` must point to the root directory of the original training dataset. The program expects the following four subdirectories. These directory names are part of the dataset layout and must not be translated or renamed:

```text
<data-root>/
├─ 只有核/
│  ├─ Ag2Se_濃度比(PL)/
│  └─ Ag2Se_pH值(6~11pH)/
└─ 有核也有殼/
   ├─ CS_不同濃度比(PL)/
   └─ CS_pH值(7-11 pH)/
```

Each text file must contain two numeric columns: wavelength and intensity.

## Usage

Run all four datasets:

```bash
python train.py --data-root "C:\path\to\training-dataset" --dataset all
```

Run one dataset with a GPU:

```bash
python train.py --data-root "C:\path\to\training-dataset" --dataset PH-SHELL --device cuda
```

Generate predictions without running evaluation:

```bash
python train.py --data-root "C:\path\to\training-dataset" --dataset PH-SHELL --no-evaluate
```

Generated files are written to `outputs/<dataset>/`, which is excluded by `.gitignore`. Only the source code and documentation in this repository need to be uploaded to GitHub; do not upload the data, images, or generated outputs.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests cover the main training and prediction interfaces.
