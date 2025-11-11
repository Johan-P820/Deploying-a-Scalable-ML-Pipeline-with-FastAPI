import pytest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Use a small sample of census data to make tests fast and stable
def load_data():
    df = pd.read_csv("data/census.csv")
    return df.sample(1000, random_state=0)

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_output_shapes_and_types():
    """
    Test that process_data returns arrays with compatible shapes
    """
    data = load_data()
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    train, _ = train_test_split(data, test_size=0.2, random_state=0)
    X, y, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == len(y)
    assert encoder is not None and hasattr(encoder, "categories_")
    assert lb is not None and hasattr(lb, "classes_")


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_and_inference_output():
    """
    Test that train_model returns a RandomForestClassifier
    and inference returns predictions of correct length.
    """
    data = load_data()
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    train, test = train_test_split(data, test_size=0.2, random_state=0)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_known_values():
    """
    Test compute_model_metrics on a known case.
    y = [0,1,1,0], preds = [0,1,0,0]
    Precision = 1.0, Recall = 0.5, F1 = 0.6667
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])

    p, r, f1 = compute_model_metrics(y, preds)

    assert p == pytest.approx(1.0)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(2/3, rel=1e-6)
