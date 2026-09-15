"""Tests for the public training and prediction interfaces."""

import inspect

from ag2se_prediction.pipeline import predict_spectrum, prepare_training_bundle


def test_prediction_api_signature():
    parameters = inspect.signature(predict_spectrum).parameters
    assert set(parameters) == {
        "training_files",
        "kind",
        "target_label",
        "lhs_samples",
        "device",
    }


def test_training_preprocessor_signature():
    parameters = inspect.signature(prepare_training_bundle).parameters
    assert set(parameters) == {"training_files", "kind"}

