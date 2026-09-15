"""Command-line entry point for the four Ag2Se CORE/SHELL experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ag2se_prediction.datasets import DATASET_SPECS, resolve_dataset
from ag2se_prediction.pipeline import evaluate_prediction, predict_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Ag2Se CORE and CORE-SHELL PL/pH spectra."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Local measurement-data root")
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_SPECS],
        default="all",
        help="Experiment to run (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--lhs-samples", type=int, default=24)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Save the prediction without calculating evaluation metrics",
    )
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_dataset(name: str, args: argparse.Namespace) -> None:
    dataset = resolve_dataset(name, args.data_root)
    missing_training = [
        str(path) for path in dataset["training_files"].values() if not path.is_file()
    ]
    if missing_training:
        raise FileNotFoundError("Missing training files:\n" + "\n".join(missing_training))

    prediction = predict_spectrum(
        dataset["training_files"],
        dataset["kind"],
        dataset["target_label"],
        lhs_samples=args.lhs_samples,
        device=args.device,
    )

    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_dir / "prediction.csv",
        np.column_stack([
            prediction.wavelength_nm,
            prediction.normalized_intensity,
            prediction.raw_intensity,
        ]),
        delimiter=",",
        header="wavelength_nm,predicted_normalized,predicted_raw",
        comments="",
    )
    report = {"dataset": name, "prediction": prediction.metadata}
    if not args.no_evaluate:
        if not dataset["target_file"].is_file():
            raise FileNotFoundError(f"Missing held-out file: {dataset['target_file']}")
        report["evaluation"] = evaluate_prediction(
            prediction, dataset["target_file"], dataset["kind"]
        )
    (output_dir / "report.json").write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{name}: saved {output_dir / 'prediction.csv'}")
    if "evaluation" in report:
        metrics = report["evaluation"]
        print(
            f"  R2(normalized)={metrics['r2_normalized']:.4f}, "
            f"RMSE(normalized)={metrics['rmse_normalized']:.4f}, "
            f"peak error={metrics['peak_error_nm']:.2f} nm"
        )


def main() -> None:
    args = parse_args()
    names = DATASET_SPECS if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, args)


if __name__ == "__main__":
    main()
