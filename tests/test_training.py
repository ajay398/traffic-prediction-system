import pandas as pd

from ml.training.train import (
    create_preprocessor,
    create_models,
)


def test_preprocessor_creation():

    preprocessor = create_preprocessor()

    assert preprocessor is not None


def test_models_are_created():

    models = create_models()

    assert "linear_regression" in models
    assert "random_forest" in models
    assert "xgboost" in models


def test_model_count():

    models = create_models()

    assert len(models) == 3