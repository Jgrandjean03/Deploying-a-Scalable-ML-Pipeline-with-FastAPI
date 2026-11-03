import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

# Same categorical columns the training script uses.
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "census.csv"
)
SAMPLE_SIZE = 2000
TEST_SIZE = 0.2
SEED = 42  # used for both the sample and the split


@pytest.fixture(scope="module")
def pipeline():
    """
    Build a small, reproducible train/test setup and train the model once.
    """
    data = pd.read_csv(DATA_PATH).sample(n=SAMPLE_SIZE, random_state=SEED)
    train, test = train_test_split(
        data, test_size=TEST_SIZE, random_state=SEED, stratify=data[LABEL]
    )
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    model = train_model(X_train, y_train)
    return {
        "train": train,
        "test": test,
        "X_train": X_train,
        "X_test": X_test,
        "encoder": encoder,
        "lb": lb,
        "model": model,
    }


def test_train_model_and_inference_output(pipeline):
    """
    train_model returns a fitted GradientBoostingClassifier, and inference
    returns one 0/1 prediction per test row as a numpy array.
    """
    model = pipeline["model"]
    X_test = pipeline["X_test"]

    assert isinstance(model, GradientBoostingClassifier)
    check_is_fitted(model)  # raises NotFittedError if the model is not fitted

    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_hand_made_case():
    """
    On a hand-made case, TP = 1, FP = 0, FN = 1, so precision = 1/1 and
    recall = 1/2.
    """
    y = [1, 0, 1, 0]
    preds = [1, 0, 0, 0]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 * 1.0 * 0.5 / 1.5)


def test_process_data_shapes_and_encoder_reuse(pipeline):
    """
    Processing keeps every row, train and test get the same columns, and
    training=False reuses the passed-in encoder instead of refitting it
    (refitting on test data would be data leakage).
    """
    train, test = pipeline["train"], pipeline["test"]
    X_train, X_test = pipeline["X_train"], pipeline["X_test"]
    encoder, lb = pipeline["encoder"], pipeline["lb"]

    assert X_train.shape[0] == len(train)
    assert X_test.shape[0] == len(test)
    assert X_train.shape[1] == X_test.shape[1]

    categories_before = [c.copy() for c in encoder.categories_]
    _, _, encoder_after, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert encoder_after is encoder
    assert len(encoder.categories_) == len(categories_before)
    for before, after in zip(categories_before, encoder.categories_):
        assert np.array_equal(before, after)
