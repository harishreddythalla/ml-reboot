import joblib
import numpy as np
import os


def test_logistic_regression():
    model_path = os.path.join("classification", "model", "logistic_model.pkl")
    model = joblib.load(model_path)
    X_test = np.array([[0.5, 1.2], [1.5, 0.1], [4,2.5]])
    predictions = model.predict(X_test)
    assert len(predictions) == 3
    assert all(p in [0, 1] for p in predictions)