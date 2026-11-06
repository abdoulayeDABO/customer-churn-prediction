from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_pred.app import build_demo, ordered_fields
from churn_pred.core import (
    DEFAULT_BUSINESS,
    build_features,
    build_preprocessor,
    clean_data,
    input_contract,
    load_data,
    no_action_value,
    optimize_threshold,
    predict_row,
    shap_importance,
    shap_local,
    train_model,
)

DATA_PATH = Path("data/01_raw/customer_data.csv")


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    return clean_data(raw)


def test_clean_telco_coerces_and_encodes(clean):
    assert clean["TotalCharges"].isna().sum() == 0
    assert set(clean["Churn"].unique()).issubset({0, 1})
    zero_tenure = clean["tenure"] == 0
    assert (clean.loc[zero_tenure, "TotalCharges"] == 0.0).all()


def test_engineer_features_no_leakage(clean):
    X, y = build_features(clean)
    assert "Churn" not in X.columns
    for column in [
        "avg_charge_per_month",
        "price_increase",
        "num_services",
        "num_support",
        "is_month_to_month",
        "is_new_customer",
        "has_automatic_payment",
    ]:
        assert column in X.columns
    assert len(X) == len(y) == len(clean)


def test_preprocessor_is_dense_finite():
    frame = pd.DataFrame(
        {
            "SeniorCitizen": [0, 1],
            "tenure": [1.0, 20.0],
            "MonthlyCharges": [50.0, 80.0],
            "TotalCharges": [50.0, 1600.0],
            "avg_charge_per_month": [50.0, 80.0],
            "price_increase": [0.0, 0.0],
            "num_services": [1, 4],
            "num_support": [0, 2],
            "is_month_to_month": [1, 0],
            "is_new_customer": [1, 0],
            "has_automatic_payment": [0, 1],
            "gender": ["Female", "Male"],
            "Partner": ["No", "Yes"],
            "Dependents": ["No", "Yes"],
            "InternetService": ["DSL", "No"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "OnlineSecurity": ["No", "No internet service"],
            "OnlineBackup": ["Yes", "No internet service"],
            "DeviceProtection": ["No", "No internet service"],
            "TechSupport": ["No", "No internet service"],
            "StreamingTV": ["No", "No internet service"],
            "StreamingMovies": ["No", "No internet service"],
        }
    )
    transformed = build_preprocessor().fit_transform(frame)
    assert np.isfinite(transformed).all()


def test_threshold_and_uplift_contract():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    probabilities = rng.random(size=500)
    best = optimize_threshold(y_true, probabilities)
    assert 0.05 <= best["threshold"] <= 0.95
    assert best["precision"] <= 1.0
    assert no_action_value(y_true) == -int(y_true.sum()) * DEFAULT_BUSINESS.cost_churn


def test_train_predict_and_explain(tmp_path: Path, clean):
    X_raw = clean[[c for c in clean.columns if c != "Churn"]]
    result = train_model(
        raw_path=DATA_PATH,
        params={"n_estimators": 20, "max_depth": 2},
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
    )
    assert result["metrics"]["roc_auc"] > 0.5
    assert (tmp_path / "model.joblib").exists()

    customer = {name: spec["default"] for name, spec in input_contract().items()}
    scored = predict_row(result["pipeline"], customer)
    assert 0.0 <= scored["probability"] <= 1.0

    row = X_raw.head(1)
    background = result["background"]
    local = shap_local(result["pipeline"], row, background=background)
    assert local["abs_contribution"].sum() > 0
    global_importance = shap_importance(result["pipeline"], X_raw.head(300), background=background)
    assert global_importance["mean_abs_shap"].sum() > 0
    assert global_importance["feature"].is_unique


def test_demo_maps_component_values_to_the_right_fields(tmp_path: Path):
    contract = input_contract()
    result = train_model(
        raw_path=DATA_PATH,
        params={"n_estimators": 10, "max_depth": 2},
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
    )
    bundle = {
        "pipeline": result["pipeline"],
        "model_name": "gradient_boosting",
        "threshold": result["metrics"]["business_threshold"]["threshold"],
        "business_context": result["metrics"]["business_context"],
        "metrics": result["metrics"],
        "background": result["background"],
    }

    demo = build_demo(bundle=bundle)
    score_fn = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "_score")
    expected_labels = [name.replace("_", " ").capitalize() for name in ordered_fields(contract)]
    assert [component.label for component in score_fn.inputs] == expected_labels

    values = [component.value for component in score_fn.inputs]
    figure, markdown, action, shap_figure, table = score_fn.fn(*values)
    assert figure is not None
    assert "Probabilité de résiliation" in markdown
    assert action
    assert shap_figure is not None
    assert not table.empty
