"""Physics-informed XGBoost spectrum prediction without target leakage.

The public prediction API accepts training spectra and a numeric target label.
It does not accept the target spectrum.  Model selection, physical-property
extrapolation, residual gating, and post-processing decisions are made using
leave-one-training-spectrum-out validation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import xgboost as xgb
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import curve_fit
from scipy.signal import medfilt, savgol_filter
from scipy.stats import qmc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm


X_MIN, X_MAX, N_GRID = 550.0, 1000.0, 2000
WAVELENGTHS = np.linspace(X_MIN, X_MAX, N_GRID)

SNR_THRESHOLD = 8.0
FWHM_SPLIT = 95.0
SHARP_AMPLITUDE_FRACTION = 0.30
MASK_PADDING = 1.3
DEFAULT_MASK_HALF_WIDTH = 45.0
SMOOTH_WINDOW = 81
STRETCH_CLIP = (0.4, 1.8)
GATE_MARGIN = 0.98

PREPROCESS_PROFILES = {
    "PL": {
        "use_mask": False,
        "weight_power": 0.0,
        "median_window": 101,
        "savgol_window": 1001,
        "savgol_order": 3,
    },
    "PH": {
        "use_mask": True,
        "weight_power": 0.5,
        "median_window": 101,
        "savgol_window": 1001,
        "savgol_order": 3,
    },
}

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "gamma": 0.5,
}


@dataclass(frozen=True)
class ProcessedSpectrum:
    raw: np.ndarray
    normalized: np.ndarray
    amplitude: float
    baseline: float
    peak_x: float
    fwhm: float
    snr: float
    is_real_peak: bool


@dataclass(frozen=True)
class TrainingBundle:
    spectra: dict[float, ProcessedSpectrum]
    quality: dict[float, float]
    sample_weights: dict[float, np.ndarray]
    masks: dict[float, np.ndarray]

    @property
    def labels(self) -> list[float]:
        return sorted(self.spectra)


@dataclass(frozen=True)
class Prediction:
    wavelength_nm: np.ndarray
    normalized_intensity: np.ndarray
    raw_intensity: np.ndarray
    metadata: dict


def _odd_window(requested: int, size: int, minimum: int = 3) -> int:
    window = min(requested, size if size % 2 else size - 1)
    if window < minimum:
        raise ValueError(f"Spectrum has too few points ({size}) for preprocessing")
    return window


def _load_raw(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Expected two numeric columns in {path}")
    x, y = values[:, 0], values[:, 1]
    keep = np.isfinite(x) & np.isfinite(y) & (x >= X_MIN) & (x <= X_MAX)
    x, y = x[keep], y[keep]
    if len(x) < 5:
        raise ValueError(f"Insufficient data in {path} between {X_MIN:g}-{X_MAX:g} nm")
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, unique_index = np.unique(x, return_index=True)
    return unique_x, y[unique_index]


def _fwhm(y: np.ndarray) -> float:
    shifted = y - np.min(y)
    peak = int(np.argmax(shifted))
    half = shifted[peak] / 2.0
    left = right = None
    for i in range(peak - 1, -1, -1):
        if shifted[i] <= half < shifted[i + 1]:
            left = WAVELENGTHS[i] + (
                (half - shifted[i]) / (shifted[i + 1] - shifted[i])
            ) * (WAVELENGTHS[i + 1] - WAVELENGTHS[i])
            break
    for i in range(peak + 1, len(shifted)):
        if shifted[i] <= half < shifted[i - 1]:
            right = WAVELENGTHS[i] + (
                (half - shifted[i]) / (shifted[i - 1] - shifted[i])
            ) * (WAVELENGTHS[i - 1] - WAVELENGTHS[i])
            break
    return float(right - left) if left is not None and right is not None else np.nan


def _preprocess_spectrum(path: str | Path, kind: str) -> ProcessedSpectrum:
    if kind not in PREPROCESS_PROFILES:
        raise ValueError(f"Unknown spectrum kind {kind!r}; expected 'PL' or 'PH'")
    profile = PREPROCESS_PROFILES[kind]
    x, y = _load_raw(path)
    median_window = _odd_window(profile["median_window"], len(y))
    savgol_window = _odd_window(profile["savgol_window"], len(y), profile["savgol_order"] + 2)
    filtered = medfilt(y, median_window)
    smoothed = savgol_filter(filtered, savgol_window, profile["savgol_order"])
    curve = interp1d(
        x,
        smoothed,
        bounds_error=False,
        fill_value=(float(smoothed[0]), float(smoothed[-1])),
    )(WAVELENGTHS)

    noise_window = _odd_window(51, len(y))
    residual = y - medfilt(y, noise_window)
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
    baseline = float(np.min(curve))
    amplitude = float(np.max(curve) - baseline)
    normalized = (curve - baseline) / max(amplitude, 1e-9)
    width = _fwhm(curve)
    snr = float(amplitude / noise)
    return ProcessedSpectrum(
        raw=curve,
        normalized=normalized,
        amplitude=amplitude,
        baseline=baseline,
        peak_x=float(WAVELENGTHS[np.argmax(curve)]),
        fwhm=width,
        snr=snr,
        is_real_peak=bool(snr >= SNR_THRESHOLD and np.isfinite(width)),
    )


def prepare_training_bundle(
    training_files: Mapping[float, str | Path], kind: str
) -> TrainingBundle:
    """Load training files only; a target/ground-truth path is not accepted."""
    if len(training_files) < 3:
        raise ValueError("At least three training spectra are required")
    if kind not in PREPROCESS_PROFILES:
        raise ValueError(f"Unknown spectrum kind {kind!r}; expected 'PL' or 'PH'")
    profile = PREPROCESS_PROFILES[kind]
    spectra = {float(label): _preprocess_spectrum(path, kind) for label, path in training_files.items()}
    quality = {label: max(spectrum.snr, 1e-9) for label, spectrum in spectra.items()}
    sample_weights = {
        label: np.full(N_GRID, max(spectrum.snr, 1.0) ** profile["weight_power"])
        for label, spectrum in spectra.items()
    }
    masks = {}
    for label, spectrum in spectra.items():
        if not profile["use_mask"] or spectrum.is_real_peak:
            masks[label] = np.ones(N_GRID, dtype=bool)
        else:
            half_width = (
                spectrum.fwhm / 2.0 * MASK_PADDING
                if np.isfinite(spectrum.fwhm)
                else DEFAULT_MASK_HALF_WIDTH
            )
            masks[label] = np.abs(WAVELENGTHS - spectrum.peak_x) > half_width
    return TrainingBundle(spectra, quality, sample_weights, masks)


def _weighted_polyfit(x, y, weights, degree: int) -> np.ndarray:
    x, y, weights = map(lambda v: np.asarray(v, dtype=float), (x, y, weights))
    degree = max(0, min(degree, len(x) - 1))
    return np.polyfit(x, y, degree, w=np.sqrt(np.maximum(weights, 1e-12)))


def _gaussian(x, amplitude, center, sigma, baseline):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2)) + baseline


def _gaussian_amplitude(labels, amplitudes, target):
    labels = np.asarray(labels, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    baseline = float(np.min(amplitudes))
    try:
        params, _ = curve_fit(
            _gaussian,
            labels,
            amplitudes,
            p0=[amplitudes.max(), labels[np.argmax(amplitudes)], 6.0, baseline],
            bounds=(
                [0, labels.min() - 10, 1.0, 0],
                [amplitudes.max() * 3, labels.max() + 10, 40.0, max(baseline * 2, 1.0)],
            ),
            maxfev=20_000,
        )
        return float(_gaussian(target, *params))
    except (RuntimeError, ValueError):
        order = np.argsort(labels)
        return float(np.interp(target, labels[order], amplitudes[order]))


def _pchip_or_linear(labels, values, quality, target):
    labels, values = np.asarray(labels, float), np.asarray(values, float)
    order = np.argsort(labels)
    labels, values = labels[order], values[order]
    if labels.min() <= target <= labels.max():
        if len(labels) >= 3:
            return float(PchipInterpolator(labels, values)(target))
        return float(np.interp(target, labels, values))
    return float(np.polyval(_weighted_polyfit(labels, values, quality, 1), target))


AMPLITUDE_STRATEGIES = {
    "pchip": _pchip_or_linear,
    "gaussian": lambda x, y, w, target: _gaussian_amplitude(x, y, target),
    "quadratic": lambda x, y, w, target: float(np.polyval(_weighted_polyfit(x, y, w, 2), target)),
    "max_gaussian_quadratic": lambda x, y, w, target: max(
        _gaussian_amplitude(x, y, target),
        float(np.polyval(_weighted_polyfit(x, y, w, 2), target)),
    ),
}

FWHM_STRATEGIES = {
    "pchip": _pchip_or_linear,
    "linear": lambda x, y, w, target: float(np.polyval(_weighted_polyfit(x, y, w, 1), target)),
}


def _select_amplitude_strategy(bundle: TrainingBundle) -> tuple[str, dict[str, float]]:
    labels = np.asarray(bundle.labels, dtype=float)
    amplitudes = np.asarray([bundle.spectra[label].amplitude for label in labels])
    quality = np.asarray([bundle.quality[label] for label in labels])
    scores = {}
    for name, strategy in AMPLITUDE_STRATEGIES.items():
        errors = []
        for index in range(len(labels)):
            keep = np.arange(len(labels)) != index
            try:
                predicted = strategy(labels[keep], amplitudes[keep], quality[keep], labels[index])
                relative_error = abs(predicted - amplitudes[index]) / max(amplitudes[index], 1e-9)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                relative_error = 1.0
            errors.append(quality[index] * min(relative_error, 1.0))
        scores[name] = float(np.sum(errors) / np.sum(quality))
    return min(scores, key=scores.get), scores


def _predict_peak_x(group, bundle: TrainingBundle, target_label: float) -> float:
    labels = np.asarray(sorted(group), dtype=float)
    peaks = np.asarray([bundle.spectra[label].peak_x for label in labels])
    quality = np.asarray([bundle.quality[label] for label in labels])
    if len(labels) == 1:
        return float(peaks[0])
    if labels.min() <= target_label <= labels.max():
        return float(np.interp(target_label, labels, peaks))
    return float(np.polyval(_weighted_polyfit(labels, peaks, quality, 1), target_label))


def _shape_prior(
    bundle: TrainingBundle,
    target_label: float,
    target_peak_x: float,
    *,
    exclude: float | None = None,
    allowed: list[float] | None = None,
) -> np.ndarray:
    labels = [label for label in bundle.labels if label != exclude]
    if allowed is not None:
        selected = [label for label in labels if label in allowed]
        if selected:
            labels = selected
    if not labels:
        raise ValueError("No training spectra remain for prior generation")
    weights = np.asarray(
        [bundle.quality[label] / (abs(label - target_label) + 0.5) for label in labels]
    )
    weights /= weights.sum()
    prior = np.zeros(N_GRID)
    for weight, label in zip(weights, labels):
        spectrum = bundle.spectra[label]
        aligned = interp1d(
            WAVELENGTHS - spectrum.peak_x,
            spectrum.normalized,
            bounds_error=False,
            fill_value=0.0,
        )
        prior += weight * aligned(WAVELENGTHS - target_peak_x)
    return np.clip(prior, 0.0, 1.0)


def _stretch_fwhm(curve: np.ndarray, target_fwhm: float) -> np.ndarray:
    current_fwhm = _fwhm(curve)
    if not (np.isfinite(current_fwhm) and np.isfinite(target_fwhm) and current_fwhm > 0):
        return curve
    scale = float(np.clip(target_fwhm / current_fwhm, *STRETCH_CLIP))
    peak_x = WAVELENGTHS[int(np.argmax(curve))]
    stretched = interp1d(
        peak_x + (WAVELENGTHS - peak_x) * scale,
        curve,
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
    )
    return np.clip(stretched(WAVELENGTHS), 0.0, 1.2)


def _select_fwhm_strategy(bundle: TrainingBundle, group: list[float]):
    candidates = [
        label
        for label in group
        if bundle.spectra[label].is_real_peak and np.isfinite(bundle.spectra[label].fwhm)
    ]
    weighted_errors = {"none": []} | {name: [] for name in FWHM_STRATEGIES}
    total_quality = 0.0
    for held_out in candidates:
        others = [
            label for label in group
            if label != held_out and np.isfinite(bundle.spectra[label].fwhm)
        ]
        if len(others) < 2:
            continue
        true_spectrum = bundle.spectra[held_out]
        prior = _shape_prior(
            bundle, held_out, true_spectrum.peak_x, exclude=held_out, allowed=others
        )
        prior_fwhm = _fwhm(prior)
        if not (np.isfinite(prior_fwhm) and np.isfinite(true_spectrum.fwhm)):
            continue
        weight = bundle.quality[held_out]
        total_quality += weight
        weighted_errors["none"].append(weight * abs(prior_fwhm - true_spectrum.fwhm))
        labels = np.asarray(others, float)
        widths = np.asarray([bundle.spectra[label].fwhm for label in others])
        quality = np.asarray([bundle.quality[label] for label in others])
        for name, strategy in FWHM_STRATEGIES.items():
            try:
                target_width = strategy(labels, widths, quality, held_out)
                predicted_width = _fwhm(_stretch_fwhm(prior, target_width))
                error = abs(predicted_width - true_spectrum.fwhm)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                error = abs(prior_fwhm - true_spectrum.fwhm)
            weighted_errors[name].append(weight * error)
    if len(weighted_errors["none"]) < 2 or total_quality <= 0:
        return "none", {}
    scores = {name: float(np.sum(values) / total_quality) for name, values in weighted_errors.items()}
    best = min(scores, key=scores.get)
    if best != "none" and scores[best] >= scores["none"] * GATE_MARGIN:
        best = "none"
    return best, scores


def _predict_physics(bundle: TrainingBundle, target_label: float) -> dict:
    labels = np.asarray(bundle.labels, dtype=float)
    quality = np.asarray([bundle.quality[label] for label in labels])
    amplitudes = np.asarray([bundle.spectra[label].amplitude for label in labels])
    baselines = np.asarray([bundle.spectra[label].baseline for label in labels])

    amplitude_strategy, amplitude_scores = _select_amplitude_strategy(bundle)
    amplitude = AMPLITUDE_STRATEGIES[amplitude_strategy](
        labels, amplitudes, quality, target_label
    )
    amplitude = max(float(amplitude), float(np.median(amplitudes)) * 0.2)
    order = np.argsort(labels)
    baseline = float(interp1d(
        labels[order], baselines[order], fill_value="extrapolate", bounds_error=False
    )(target_label))

    amplitude_floor = amplitudes.max() * 0.15
    sharp = [
        label for label in bundle.labels
        if bundle.spectra[label].is_real_peak
        and np.isfinite(bundle.spectra[label].fwhm)
        and bundle.spectra[label].fwhm < FWHM_SPLIT
        and bundle.spectra[label].amplitude >= amplitude_floor
    ]
    broad = [
        label for label in bundle.labels
        if bundle.spectra[label].is_real_peak and label not in sharp
    ]
    usable = [label for label in bundle.labels if bundle.spectra[label].is_real_peak]
    usable = usable or bundle.labels
    sharp_max = max((bundle.spectra[label].amplitude for label in sharp), default=0.0)
    is_sharp = bool(sharp and amplitude >= sharp_max * SHARP_AMPLITUDE_FRACTION)
    group = sharp if is_sharp else broad if len(broad) >= 2 else usable
    if is_sharp:
        amplitude = max(amplitude, sharp_max)

    peak_x = _predict_peak_x(group, bundle, target_label)
    fwhm_strategy, fwhm_scores = _select_fwhm_strategy(bundle, group)
    widths_from = [label for label in group if np.isfinite(bundle.spectra[label].fwhm)]
    if len(widths_from) >= 2:
        width_labels = np.asarray(widths_from, float)
        widths = np.asarray([bundle.spectra[label].fwhm for label in widths_from])
        width_quality = np.asarray([bundle.quality[label] for label in widths_from])
        strategy = FWHM_STRATEGIES.get(fwhm_strategy, FWHM_STRATEGIES["linear"])
        target_fwhm = float(strategy(width_labels, widths, width_quality, target_label))
    else:
        finite_widths = [bundle.spectra[label].fwhm for label in usable if np.isfinite(bundle.spectra[label].fwhm)]
        target_fwhm = float(np.mean(finite_widths)) if finite_widths else np.nan

    return {
        "peak_x": peak_x,
        "amplitude": amplitude,
        "baseline": baseline,
        "target_fwhm": target_fwhm,
        "group": group,
        "spectrum_type": "sharp" if is_sharp else "broad",
        "amplitude_strategy": amplitude_strategy,
        "amplitude_cv_scores": amplitude_scores,
        "fwhm_strategy": fwhm_strategy,
        "fwhm_cv_scores": fwhm_scores,
        "stretch_enabled": fwhm_strategy != "none",
    }


def _features(prior: np.ndarray, label: float, mean_peak_x: float) -> np.ndarray:
    distance = WAVELENGTHS - mean_peak_x
    gradient = np.gradient(prior, WAVELENGTHS)
    second_gradient = np.gradient(gradient, WAVELENGTHS)
    return np.column_stack([
        np.full(N_GRID, label),
        WAVELENGTHS,
        np.full(N_GRID, label**2),
        distance,
        distance**2,
        np.abs(distance),
        prior,
        gradient,
        second_gradient,
    ])


def _build_records(bundle: TrainingBundle, physics: dict, target_label: float):
    mean_peak_x = float(np.mean([spectrum.peak_x for spectrum in bundle.spectra.values()]))
    records = {}
    for held_out in bundle.labels:
        spectrum = bundle.spectra[held_out]
        prior = _shape_prior(
            bundle,
            held_out,
            spectrum.peak_x,
            exclude=held_out,
            allowed=physics["group"],
        )
        mask = bundle.masks[held_out]
        records[held_out] = (
            _features(prior, held_out, mean_peak_x)[mask],
            (spectrum.normalized - prior)[mask],
            bundle.sample_weights[held_out][mask],
        )
    target_prior = _shape_prior(
        bundle, target_label, physics["peak_x"], allowed=physics["group"]
    )
    return records, target_prior, _features(target_prior, target_label, mean_peak_x)


def _xgb_options(device: str) -> dict:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "random_state": 42,
        "verbosity": 0,
    }


def _loro_rmse(records, params: dict, device: str) -> float:
    errors = []
    for held_out in records:
        train_labels = [label for label in records if label != held_out]
        x_train = np.vstack([records[label][0] for label in train_labels])
        y_train = np.concatenate([records[label][1] for label in train_labels])
        weights = np.concatenate([records[label][2] for label in train_labels])
        x_validation, y_validation, validation_weights = records[held_out]
        model = xgb.XGBRegressor(**params, **_xgb_options(device))
        model.fit(x_train, y_train, sample_weight=weights)
        errors.append(np.sqrt(mean_squared_error(
            y_validation,
            model.predict(x_validation),
            sample_weight=validation_weights,
        )))
    return float(np.mean(errors))


def _optimize_lhs(records, samples: int, device: str):
    if samples <= 0:
        return dict(DEFAULT_MODEL_PARAMS), _loro_rmse(records, DEFAULT_MODEL_PARAMS, device)
    lower = [0.01, 2, 3, 0.6, 0.6, 0.0, 0.5]
    upper = [0.08, 5, 20, 1.0, 1.0, 1.0, 5.0]
    candidates = qmc.scale(
        qmc.LatinHypercube(d=7, seed=1).random(samples), lower, upper
    )
    best_params, best_score = None, np.inf
    for point in tqdm(candidates, desc="training-only LHS", leave=False):
        params = {
            "n_estimators": DEFAULT_MODEL_PARAMS["n_estimators"],
            "learning_rate": float(point[0]),
            "max_depth": int(point[1]),
            "min_child_weight": int(point[2]),
            "subsample": float(point[3]),
            "colsample_bytree": float(point[4]),
            "reg_alpha": float(point[5]),
            "reg_lambda": float(point[6]),
            "gamma": DEFAULT_MODEL_PARAMS["gamma"],
        }
        score = _loro_rmse(records, params, device)
        if score < best_score:
            best_params, best_score = params, score
    return best_params, float(best_score)


def _clean_prediction(curve: np.ndarray, peak_x: float, target_fwhm: float) -> np.ndarray:
    curve = savgol_filter(np.clip(curve, 0.0, 1.3), SMOOTH_WINDOW, 3)
    width = target_fwhm if np.isfinite(target_fwhm) else DEFAULT_MASK_HALF_WIDTH * 2
    far = np.abs(WAVELENGTHS - peak_x) > 2.5 * max(width, 20.0)
    floor = float(np.percentile(curve[far], 40)) if np.any(far) else 0.0
    curve = np.clip(curve - floor, 0.0, None)
    return curve / (curve.max() + 1e-9)


def predict_spectrum(
    training_files: Mapping[float, str | Path],
    kind: str,
    target_label: float,
    *,
    lhs_samples: int = 24,
    device: str = "cpu",
) -> Prediction:
    """Train and predict one held-out spectrum without reading its true curve."""
    bundle = prepare_training_bundle(training_files, kind)
    target_label = float(target_label)
    if target_label in bundle.spectra:
        raise ValueError("target_label must not be present in training_files")

    physics = _predict_physics(bundle, target_label)
    records, prior, target_features = _build_records(bundle, physics, target_label)
    params, residual_cv_rmse = _optimize_lhs(records, lhs_samples, device)
    prior_cv_rmse = float(np.mean([
        np.sqrt(np.average(residual**2, weights=weights))
        for _, residual, weights in records.values()
    ]))
    residual_enabled = residual_cv_rmse < prior_cv_rmse * GATE_MARGIN

    if residual_enabled:
        x_train = np.vstack([record[0] for record in records.values()])
        y_train = np.concatenate([record[1] for record in records.values()])
        weights = np.concatenate([record[2] for record in records.values()])
        model = xgb.XGBRegressor(**params, **_xgb_options(device))
        model.fit(x_train, y_train, sample_weight=weights)
        predicted = prior + model.predict(target_features)
    else:
        predicted = prior.copy()
    if physics["stretch_enabled"]:
        predicted = _stretch_fwhm(predicted, physics["target_fwhm"])
    normalized = _clean_prediction(predicted, physics["peak_x"], physics["target_fwhm"])
    raw = normalized * physics["amplitude"] + physics["baseline"]

    metadata = {
        key: value for key, value in physics.items()
        if key not in {"group"}
    }
    metadata.update({
        "training_labels": bundle.labels,
        "target_label": target_label,
        "selected_group": physics["group"],
        "residual_enabled": residual_enabled,
        "prior_loro_rmse": prior_cv_rmse,
        "residual_loro_rmse": residual_cv_rmse,
        "model_params": params,
    })
    return Prediction(WAVELENGTHS.copy(), normalized, raw, metadata)


def evaluate_prediction(
    prediction: Prediction,
    target_file: str | Path,
    kind: str,
) -> dict[str, float]:
    """Evaluate only after prediction; target data never enters model selection."""
    target = _preprocess_spectrum(target_file, kind)
    true_normalized = target.normalized
    predicted_normalized = np.clip(prediction.normalized_intensity, 0.0, 1.2)
    true_peak = float(WAVELENGTHS[np.argmax(target.raw)])
    predicted_peak = float(WAVELENGTHS[np.argmax(prediction.raw_intensity)])
    true_fwhm = _fwhm(true_normalized)
    predicted_fwhm = _fwhm(predicted_normalized)
    return {
        "r2_raw": float(r2_score(target.raw, prediction.raw_intensity)),
        "r2_normalized": float(r2_score(true_normalized, predicted_normalized)),
        "rmse_normalized": float(np.sqrt(mean_squared_error(true_normalized, predicted_normalized))),
        "mae_normalized": float(mean_absolute_error(true_normalized, predicted_normalized)),
        "peak_error_nm": abs(true_peak - predicted_peak),
        "fwhm_error_nm": (
            abs(true_fwhm - predicted_fwhm)
            if np.isfinite(true_fwhm) and np.isfinite(predicted_fwhm)
            else np.nan
        ),
    }

