from __future__ import annotations

import argparse
import json

from churn_pred.config import RAW_DIR, RAW_FILENAME
from churn_pred.core import BusinessContext, train_model

raw_path = RAW_DIR / RAW_FILENAME

def main() -> None:
    """Train the Telco churn model and persist the serving bundle."""
    parser = argparse.ArgumentParser(description="Train the Telco churn model")
    parser.add_argument("--raw-path", type=str, default=raw_path)
    parser.add_argument("--model-name", type=str, default="gradient_boosting")
    parser.add_argument("--cost-churn", type=float, default=500.0)
    parser.add_argument("--cost-contact", type=float, default=50.0)
    parser.add_argument("--rescue-rate", type=float, default=0.30)
    args = parser.parse_args()

    business = BusinessContext(
        cost_churn=args.cost_churn, cost_contact=args.cost_contact, rescue_rate=args.rescue_rate
    )
    result = train_model(
        raw_path=args.raw_path, model_name=args.model_name, business=business
    )
    metrics = result["metrics"]
    payload = {
        key: value
        for key, value in metrics.items()
        if key in {"model", "roc_auc", "pr_auc", "business_threshold"}
    }
    payload["model_path"] = str(result["model_path"])
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
