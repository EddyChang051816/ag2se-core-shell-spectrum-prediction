"""Regression checks for the strict train/predict boundary."""

import inspect

from ag2se_prediction.pipeline import predict_spectrum, prepare_training_bundle


def test_prediction_api_cannot_accept_target_spectrum():
    parameters = inspect.signature(predict_spectrum).parameters
    assert "target_file" not in parameters
    assert "ground_truth" not in parameters
    assert set(parameters).issuperset({"training_files", "kind", "target_label"})


def test_training_preprocessor_cannot_accept_target_spectrum():
    parameters = inspect.signature(prepare_training_bundle).parameters
    assert set(parameters) == {"training_files", "kind"}

