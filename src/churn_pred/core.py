"""Churn prediction business logic.

This module is the single source of truth for the Churn prediction use case:
data cleaning, feature engineering, model candidates, business-threshold
optimisation and SHAP-based explanations.

The notebooks are the workshop: they import these functions instead of
re-implementing them. The saved model is a single scikit-learn ``Pipeline``
that embeds feature engineering, so inference only needs the raw contract
fields of a customer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_pred import config

TARGET_COLUMN = "Churn"
ID_COLUMN = config.ID_COLUMN
RANDOM_STATE = 42
TEST_SIZE = 0.2

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
SUPPORT_COLUMNS = ["OnlineSecurity", "TechSupport"]
PAYMENT_COLUMNS = ["PaymentMethod"]

RAW_CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    *SERVICE_COLUMNS,
]
RAW_NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

ENGINEERED_NUMERIC_FEATURES = [
    "avg_charge_per_month",
    "price_increase",
    "num_services",
    "num_support",
    "is_month_to_month",
    "is_new_customer",
    "has_automatic_payment",
]
NUMERIC_FEATURES = [*RAW_NUMERIC_FEATURES, *ENGINEERED_NUMERIC_FEATURES]
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES

CANDIDATE_MODELS = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]


@dataclass(frozen=True)
class BusinessContext:
    """Cost model used to pick the decision threshold of the churn model.

    Attributes:
        cost_churn: margin lost when a customer leaves (per churned customer).
        cost_contact: cost of one retention contact (offer, call, discount).
        rescue_rate: share of targeted customers actually retained.
    """

    cost_churn: float = 500.0
    cost_contact: float = 50.0
    rescue_rate: float = 0.30


DEFAULT_BUSINESS = BusinessContext()


def _is_text_column(series: pd.Series) -> bool:
    """Return True for object/string columns across pandas 2 and 3."""
    if pd.api.types.is_object_dtype(series):
        return True
    return isinstance(series.dtype, pd.StringDtype)


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the immutable raw Telco dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset and document every treatment.

    Treatments:
        - strip whitespace on text columns (e.g. ``" Yes"`` -> ``"Yes"``);
        - coerce ``TotalCharges`` to numeric (11 blank rows in the raw file);
        - Remove new customers blank total charges;
        - encode the target ``Churn`` as 1/0 and drop unlabelled rows;
        - drop duplicated customer ids.
    """
    out = df.copy()
    for column in out.columns:
        if _is_text_column(out[column]):
            out[column] = out[column].astype("string").str.strip()

    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out = out.dropna(subset=["TotalCharges"])

    out[TARGET_COLUMN] = out[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    out = out.dropna(subset=[TARGET_COLUMN])
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)

    if ID_COLUMN in out.columns:
        out = out.drop_duplicates(subset=[ID_COLUMN])
    return out.reset_index(drop=True)


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series for `column`, missing column -> constant default."""
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a text series for `column`, missing column -> empty strings."""
    if column in frame.columns:
        return frame[column].astype("string")
    return pd.Series("", index=frame.index, dtype="string")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-driven features computed from the customer's own row only.

    No aggregated or historical feature is used, so the function stays valid
    at scoring time for a single customer (no leakage, no train/test drift).
    """
    out = df.copy()
    tenure = _numeric_column(out, "tenure").clip(lower=0)
    monthly = _numeric_column(out, "MonthlyCharges")
    total = _numeric_column(out, "TotalCharges")

    months = tenure.where(tenure > 0, 1)
    out["avg_charge_per_month"] = (total / months).round(4)
    out["price_increase"] = (out["avg_charge_per_month"] - monthly).round(4)

    out["num_services"] = sum(
        (out[column].astype("string") == "Yes").astype(int)
        for column in SERVICE_COLUMNS
        if column in out.columns
    )
    out["num_support"] = sum(
        (out[column].astype("string") == "Yes").astype(int)
        for column in SUPPORT_COLUMNS
        if column in out.columns
    )
    out["is_month_to_month"] = (
        _text_column(out, "Contract").str.lower().eq("month-to-month").astype(int)
    )
    out["is_new_customer"] = (tenure <= 12).astype(int)
    out["has_automatic_payment"] = (
        _text_column(out, "PaymentMethod")
        .str.contains("automatic", case=False, na=False)
        .astype(int)
    )
    return out


def build_features(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the engineered feature matrix and the target."""
    engineered = engineer_features(df)
    X = engineered.drop(columns=[target_column], errors="ignore")
    y = engineered[target_column] if target_column in engineered.columns else None
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing transformer (imputation, scaling, one-hot encoding)."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_classifier(model_name: str = "gradient_boosting", params: dict | None = None):
    """Instantiate a classifier by name."""
    params = dict(params or {})
    if model_name == "logistic_regression":
        params.setdefault("max_iter", 2000)
        params.setdefault("C", 1.0)
        return LogisticRegression(random_state=RANDOM_STATE, **params)
    if model_name == "random_forest":
        params.setdefault("n_estimators", 400)
        params.setdefault("max_depth", 8)
        params.setdefault("min_samples_leaf", 10)
        params.setdefault("class_weight", "balanced_subsample")
        params.setdefault("n_jobs", -1)
        return RandomForestClassifier(random_state=RANDOM_STATE, **params)
    if model_name == "gradient_boosting":
        params.setdefault("n_estimators", 300)
        params.setdefault("max_depth", 3)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("subsample", 0.9)
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "xgboost is required for the 'xgboost' candidate. "
                "Install it with: pip install xgboost"
            ) from exc
        params.setdefault("n_estimators", 400)
        params.setdefault("max_depth", 4)
        params.setdefault("learning_rate", 0.05)
        params.setdefault("subsample", 0.9)
        params.setdefault("colsample_bytree", 0.9)
        params.setdefault("reg_lambda", 1.0)
        params.setdefault("eval_metric", "logloss")
        return XGBClassifier(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Unknown model '{model_name}'. Available: {CANDIDATE_MODELS}")


def build_pipeline(model_name: str = "gradient_boosting", params: dict | None = None) -> Pipeline:
    """Build the full pipeline: feature engineering -> preprocessing -> classifier."""
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(engineer_features)),
            ("preprocessor", build_preprocessor()),
            ("classifier", build_classifier(model_name, params)),
        ]
    )


def available_models() -> list[str]:
    """Return the candidate models that can actually run in this environment."""
    models = ["logistic_regression", "random_forest", "gradient_boosting"]
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return models
    return [*models, "xgboost"]


def evaluate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare candidate models with stratified cross-validation."""
    model_names = model_names or available_models()
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "accuracy": "accuracy",
        "recall": "recall",
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name in model_names:
        pipeline = build_pipeline(name)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def expected_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    business: BusinessContext = DEFAULT_BUSINESS,
) -> float:
    """Net value of a churn-campaign decision for a set of predictions.

    Net value = value of the churners actually retained
    - cost of every contact
    - value of the churners we lost (contacted but not rescued + not contacted).

    The "no campaign" baseline (nobody is contacted, every churner is lost)
    is ``-n_churners * cost_churn``: the interesting metric is therefore the
    *uplift* of a campaign versus that baseline, see `campaign_uplift`.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    saved = tp * business.rescue_rate * business.cost_churn
    campaign = (tp + fp) * business.cost_contact
    lost = (fn + tp) * business.cost_churn
    return float(saved - campaign - lost)


def no_action_value(
    y_true: np.ndarray | pd.Series, business: BusinessContext = DEFAULT_BUSINESS
) -> float:
    """Net value when no retention action is taken (every churner is lost)."""
    y_true = np.asarray(y_true)
    return float(-float(y_true.sum()) * business.cost_churn)


def campaign_uplift(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    business: BusinessContext = DEFAULT_BUSINESS,
    n_random: int = 200,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Compare the model campaign with no action and with random targeting.

    Random targeting with the same budget is the honest benchmark: it shows
    that the model adds value beyond "call as many clients as possible".
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    model_value = expected_value(y_true, predictions, business)
    baseline = no_action_value(y_true, business)
    targeted = int(predictions.sum())

    rng = np.random.default_rng(random_state)
    random_values = []
    for _ in range(n_random):
        if targeted == 0:
            random_values.append(baseline)
            continue
        picks = rng.choice(len(y_true), size=targeted, replace=False)
        random_predictions = np.zeros(len(y_true), dtype=int)
        random_predictions[picks] = 1
        random_values.append(expected_value(y_true, random_predictions, business))
    random_mean = float(np.mean(random_values))
    return {
        "threshold": float(threshold),
        "targeted": targeted,
        "model_net_value": model_value,
        "no_action_net_value": baseline,
        "uplift_vs_no_action": model_value - baseline,
        "random_targeting_net_value": random_mean,
        "uplift_vs_random": model_value - random_mean,
        "business_context": asdict(business),
    }


def optimize_threshold(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    business: BusinessContext = DEFAULT_BUSINESS,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
) -> dict[str, Any]:
    """Pick the threshold that maximises the expected business value.

    The default 0.5 cut-off is arbitrary for an imbalanced problem: this
    function selects the threshold that maximises the net value of the
    retention campaign (value of retained customers minus campaign cost
    minus value of the churners we missed).
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    grid = np.linspace(min_threshold, max_threshold, 181)
    best: dict[str, Any] | None = None
    for threshold in grid:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        row = {
            "threshold": float(threshold),
            "net_value": expected_value(y_true, predictions, business),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "targeted": int(tp + fp),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
        if best is None or row["net_value"] > best["net_value"]:
            best = row
    assert best is not None
    return best


def threshold_curve(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    business: BusinessContext = DEFAULT_BUSINESS,
    n_points: int = 91,
) -> pd.DataFrame:
    """Return the expected value curve over the thresholds."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    rows = []
    for threshold in np.linspace(0.05, 0.95, n_points):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "net_value": expected_value(y_true, predictions, business),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "targeted": int(tp + fp),
            }
        )
    return pd.DataFrame(rows)


def recommendation(
    probability: float,
    threshold: float,
    business: BusinessContext = DEFAULT_BUSINESS,
) -> dict[str, str]:
    """Translate a probability into an operational decision."""
    if probability >= threshold:
        level = "HIGH"
        action = "Contacter dans les 48h (offre de rétention + suivi technique)"
    elif probability >= max(threshold * 0.6, 0.2):
        level = "MEDIUM"
        action = "Surveiller et mettre en place une campagne de nurturing"
    else:
        level = "LOW"
        action = "Aucune action spécifique (parcours standard)"
    gain = business.rescue_rate * business.cost_churn - business.cost_contact
    message = (
        f"Valeur attendue d'un contact réussi: {gain:+.0f} EUR, "
        f"coût du contact: {business.cost_contact:.0f} EUR, "
        f"perte par client perdu: {business.cost_churn:.0f} EUR."
    )
    return {"level": level, "action": action, "message": message}


def _parent_feature(transformed_name: str, columns: list[str]) -> str:
    """Map a transformed feature name back to its original column."""
    cleaned = transformed_name.split("__", 1)[-1]
    for column in sorted(columns, key=len, reverse=True):
        if cleaned == column or cleaned.startswith(f"{column}_"):
            return column
    return cleaned


def _raw_shap_values(explainer: Any, matrix: np.ndarray) -> np.ndarray:
    """Extract a (n_samples, n_features) array of shap values across shap versions."""
    try:
        values = explainer.shap_values(matrix)
    except Exception:  # pragma: no cover - depends on shap version
        values = explainer(matrix).values
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _explainer_for(
    pipeline: Pipeline, matrix: np.ndarray, background: np.ndarray | None = None
) -> Any:
    """Build the shap explainer matching the classifier family.

    Tree models: when a background sample is available, shap runs in
    interventional mode with ``model_output="probability"`` so the
    contributions are expressed in probability points (readable for the
    business). Without background, shap falls back to the path-dependent
    explainer, whose contributions live in the log-odds scale.
    Linear models always need a background sample, otherwise every
    contribution collapses to zero.
    """
    import shap

    classifier = pipeline.named_steps["classifier"]
    tree_models = {"GradientBoostingClassifier", "RandomForestClassifier", "XGBClassifier"}
    if hasattr(classifier, "estimators_") or type(classifier).__name__ in tree_models:
        if background is not None and len(background) > 1:
            try:
                return shap.TreeExplainer(
                    classifier,
                    data=background,
                    feature_perturbation="interventional",
                    model_output="probability",
                )
            except Exception:  # pragma: no cover - depends on shap version
                pass
        return shap.TreeExplainer(classifier)
    reference = background if background is not None else matrix
    return shap.LinearExplainer(classifier, reference[: min(100, len(reference))])


def _transform_for_explainer(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply the pipeline feature engineering + preprocessing for shap."""
    preprocessor = pipeline.named_steps["preprocessor"]
    engineered = engineer_features(X)
    return np.asarray(preprocessor.transform(engineered), dtype=float)


def _aggregate_shap(pipeline: Pipeline, values: np.ndarray) -> tuple[list[str], pd.DataFrame]:
    """Sum one-hot columns back to their parent feature."""
    preprocessor = pipeline.named_steps["preprocessor"]
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    parents = [_parent_feature(name, NUMERIC_FEATURES + CATEGORICAL_FEATURES) for name in names]
    frame = pd.DataFrame(values, columns=names)
    grouped = frame.T.groupby(parents).sum().T
    return list(grouped.columns), grouped


def shap_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_rows: int = 500,
    random_state: int = RANDOM_STATE,
    background: np.ndarray | None = None,
) -> pd.DataFrame:
    """Global SHAP feature importance aggregated on the original features.

    One-hot encoded columns are summed back to their parent feature so the
    business reads a list of customer attributes, not encoding artefacts.
    """
    matrix = _transform_for_explainer(pipeline, X)
    reference = background if background is not None else matrix[: min(100, len(matrix))]
    explainer = _explainer_for(pipeline, matrix, reference)
    sample = matrix[:max_rows]
    values = _raw_shap_values(explainer, sample)
    features, grouped = _aggregate_shap(pipeline, values)
    importance = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": grouped.abs().mean().to_numpy(),
            "mean_shap": grouped.mean().to_numpy(),
        }
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def shap_explanation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_rows: int = 500,
    background: np.ndarray | None = None,
):
    """Return a `shap.Explanation` aggregated on the original features.

    Used for the beeswarm/summary plots: the values are already summed back
    per original feature, so the axes are readable by the business.
    """
    import shap

    matrix = _transform_for_explainer(pipeline, X)
    reference = background if background is not None else matrix[: min(100, len(matrix))]
    explainer = _explainer_for(pipeline, matrix, reference)
    values = _raw_shap_values(explainer, matrix[:max_rows])
    features, grouped = _aggregate_shap(pipeline, values)
    base_value = float(np.ravel(np.asarray(explainer.expected_value))[-1])
    engineered = engineer_features(X.iloc[:max_rows])
    parent_data = (
        engineered[features] if all(f in engineered.columns for f in features) else engineered
    )
    return shap.Explanation(
        values=grouped.to_numpy(),
        base_values=np.full(len(grouped), base_value),
        data=parent_data.to_numpy(),
        feature_names=features,
    )


def shap_local(
    pipeline: Pipeline, row: pd.DataFrame, background: np.ndarray | None = None
) -> pd.DataFrame:
    """Local SHAP explanation for one customer, aggregated on original features."""
    return shap_local_values(pipeline, row, background=background)


def shap_local_values(
    pipeline: Pipeline, row: pd.DataFrame, background: np.ndarray | None = None
) -> pd.DataFrame:
    """Signed contribution of each original feature for one customer.

    `background` is the transformed sample of the training population; it is
    required for linear models (the expected value of the model) and ignored
    for tree models.
    """
    engineered = engineer_features(row)
    matrix = _transform_for_explainer(pipeline, engineered)
    explainer = _explainer_for(pipeline, matrix, background)
    values = _raw_shap_values(explainer, matrix)
    features, grouped = _aggregate_shap(pipeline, values)
    result = pd.DataFrame(
        {
            "feature": features,
            "contribution": grouped.iloc[0].to_numpy(),
            "value": [engineered.iloc[0].get(feature, np.nan) for feature in features],
        }
    )
    result["abs_contribution"] = result["contribution"].abs()
    return result.sort_values("abs_contribution", ascending=False).reset_index(drop=True)


def predict_row(pipeline: Pipeline, customer: dict[str, Any]) -> dict[str, Any]:
    """Score one customer (raw contract fields) with the persisted pipeline."""
    row = pd.DataFrame([customer])
    probability = float(pipeline.predict_proba(row)[0, 1])
    return {"probability": probability, "prediction": int(probability >= 0.5)}


def train_model(
    raw_path: str | Path | None = None,
    model_name: str = "gradient_boosting",
    params: dict | None = None,
    business: BusinessContext = DEFAULT_BUSINESS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train, evaluate and persist the Telco churn model."""
    df = clean_data(load_data(raw_path))
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = build_pipeline(model_name, params)
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    default_predictions = (probabilities >= 0.5).astype(int)
    business_threshold = optimize_threshold(y_test, probabilities, business)

    metrics = {
        "model": model_name,
        "params": params or {},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "accuracy_at_0.5": float(accuracy_score(y_test, default_predictions)),
        "precision_at_0.5": float(precision_score(y_test, default_predictions, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_test, default_predictions, zero_division=0)),
        "f1_at_0.5": float(f1_score(y_test, default_predictions, zero_division=0)),
        "net_value_at_0.5": expected_value(y_test, default_predictions, business),
        "business_threshold": business_threshold,
        "business_context": asdict(business),
    }

    model_path = Path(model_path) if model_path else config.MODELS_DIR / config.MODEL_FILENAME
    model_path.parent.mkdir(parents=True, exist_ok=True)
    background = _transform_for_explainer(
        pipeline,
        X_train.sample(n=min(100, len(X_train)), random_state=random_state),
    )
    joblib.dump(
        {
            "pipeline": pipeline,
            "model_name": model_name,
            "params": params or {},
            "threshold": business_threshold["threshold"],
            "business_context": asdict(business),
            "metrics": metrics,
            "background": background,
        },
        model_path,
    )

    metrics_path = (
        Path(metrics_path) if metrics_path else config.ARTIFACTS_DIR / config.METRICS_FILENAME
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "probabilities": probabilities,
        "background": background,
    }


def load_bundle(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load the persisted bundle (pipeline + threshold + business context)."""
    model_path = Path(model_path) if model_path else config.MODELS_DIR / config.MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def input_contract() -> dict[str, dict[str, Any]]:
    """Describe the raw fields expected by the model (Gradio + API contract)."""
    return {
        "gender": {"type": "categorical", "choices": ["Female", "Male"], "default": "Female"},
        "SeniorCitizen": {"type": "numeric", "default": 0, "min": 0, "max": 1},
        "Partner": {"type": "categorical", "choices": ["No", "Yes"], "default": "No"},
        "Dependents": {"type": "categorical", "choices": ["No", "Yes"], "default": "No"},
        "tenure": {"type": "numeric", "default": 12, "min": 0, "max": 100},
        "PhoneService": {"type": "categorical", "choices": ["No", "Yes"], "default": "Yes"},
        "MultipleLines": {
            "type": "categorical",
            "choices": ["No", "Yes", "No phone service"],
            "default": "No",
        },
        "InternetService": {
            "type": "categorical",
            "choices": ["DSL", "Fiber optic", "No"],
            "default": "DSL",
        },
        "OnlineSecurity": {
            "type": "categorical",
            "choices": ["No", "Yes", "No internet service"],
            "default": "No",
        },
        "OnlineBackup": {
            "type": "categorical",
            "choices": ["No", "Yes", "No internet service"],
            "default": "No",
        },
        "DeviceProtection": {
            "type": "categorical",
            "choices": ["No", "Yes", "No internet service"],
            "default": "No",
        },
        "TechSupport": {
            "type": "categorical",
            "choices": ["No", "Yes", "No internet service"],
            "default": "No",
        },
        "StreamingTV": {
            "type": "categorical",
            "choices": ["No", "Yes", "No internet service"],
            "default": "No",
        },
        "StreamingMovies": {
            "type": "categorical",
            "choices": ["No", "Yes", "No internet service"],
            "default": "No",
        },
        "Contract": {
            "type": "categorical",
            "choices": ["Month-to-month", "One year", "Two year"],
            "default": "Month-to-month",
        },
        "PaperlessBilling": {"type": "categorical", "choices": ["No", "Yes"], "default": "Yes"},
        "PaymentMethod": {
            "type": "categorical",
            "choices": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "default": "Electronic check",
        },
        "MonthlyCharges": {"type": "numeric", "default": 70.0, "min": 0.0, "max": 200.0},
        "TotalCharges": {"type": "numeric", "default": 900.0, "min": 0.0, "max": 10000.0},
    }
