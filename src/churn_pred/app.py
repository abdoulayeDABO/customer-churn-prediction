"""Gradio application: operational churn scoring.

The app exposes three tabs:
    1. score a single customer (probability, decision, SHAP explanation);
    2. score a portfolio (CSV upload -> targeted clients, expected value);
    3. model card (metrics, threshold, business assumptions, known limits).

Run it with: `uv run churn-gradio` or `python -m churn_pred.app`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from churn_pred.core import (
    BusinessContext,
    input_contract,
    load_bundle,
    recommendation,
    shap_local,
)

RISK_COLORS = {"HIGH": "#c0392b", "MEDIUM": "#e67e22", "LOW": "#27ae60"}

ESSENTIAL_FIELDS = ["Contract", "tenure", "MonthlyCharges", "InternetService"]


def ordered_fields(contract: dict[str, dict[str, Any]]) -> list[str]:
    """Field names in the order the Gradio components are created.

    The visible fields come first, then the accordion fields, both following the
    contract order. The scoring callback must use this exact order to map the
    component values back to their field names.
    """
    essentials = [name for name in ESSENTIAL_FIELDS if name in contract]
    extras = [name for name in contract if name not in essentials]
    return [*essentials, *extras]


def _bundle_context(bundle: dict[str, Any]) -> BusinessContext:
    """Rebuild the business context stored with the model."""
    return BusinessContext(**bundle.get("business_context", {}))


def _probability_figure(probability: float, threshold: float):
    """Horizontal gauge with the decision threshold marked."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 1.6))
    axis.barh([0], [probability], color="#c0392b", height=0.45)
    axis.barh([0], [1 - probability], left=[probability], color="#d5dbdb", height=0.45)
    axis.axvline(threshold, color="#2c3e50", linestyle="--", linewidth=2)
    axis.text(threshold, 0.32, f" seuil {threshold:.2f}", fontsize=9, color="#2c3e50")
    axis.set_xlim(0, 1)
    axis.set_yticks([])
    axis.set_xlabel("Probabilité de résiliation")
    axis.set_title(f"Risque de churn: {probability:.1%}")
    figure.tight_layout()
    return figure


def _shap_figure(explanations: pd.DataFrame, top_n: int = 8):
    """Horizontal bar chart of the local SHAP contributions."""
    import matplotlib.pyplot as plt

    top = explanations.head(top_n).sort_values("contribution")
    colors = ["#c0392b" if value < 0 else "#2980b9" for value in top["contribution"]]
    figure, axis = plt.subplots(figsize=(7, 0.45 * len(top) + 1))
    axis.barh(top["feature"], top["contribution"], color=colors)
    axis.axvline(0, color="#2c3e50", linewidth=1)
    axis.set_xlabel("Contribution au risque de churn (effet négatif = reduces risk)")
    axis.set_title("Explication SHAP du score")
    figure.tight_layout()
    return figure


def score_customer(
    bundle: dict[str, Any], values: dict[str, Any]
) -> tuple[Any, str, str, Any, pd.DataFrame]:
    """Score one customer and return the figure, the decision and the SHAP table."""
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    business = _bundle_context(bundle)

    row = pd.DataFrame([values])
    probability = float(pipeline.predict_proba(row)[0, 1])
    decision = recommendation(probability, threshold, business)
    explanations = shap_local(pipeline, row, background=bundle.get("background"))

    markdown = (
        f"### Niveau de risque: {decision['level']}\n"
        f"**Probabilité de résiliation: {probability:.1%}** "
        f"(seuil décisionnel: {threshold:.2f})\n\n"
        f"**Action recommandée:** {decision['action']}\n\n"
        f"{decision['message']}"
    )
    table = explanations[["feature", "value", "contribution"]].rename(
        columns={"feature": "Feature", "value": "Valeur", "contribution": "Contribution SHAP"}
    )
    figure = _probability_figure(probability, threshold)
    shap_figure = _shap_figure(explanations)
    return figure, markdown, decision["action"], shap_figure, table


def score_portfolio(bundle: dict[str, Any], file_path: str | None) -> tuple[pd.DataFrame, str]:
    """Score a CSV portfolio and summarise the campaign."""
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    business = _bundle_context(bundle)

    if not file_path:
        return pd.DataFrame(), "Aucun fichier fourni."

    frame = pd.read_csv(file_path)
    if "customerID" in frame.columns:
        frame = frame.set_index("customerID")
    probabilities = pipeline.predict_proba(frame)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    decisions = [recommendation(float(p), threshold, business) for p in probabilities]

    result = frame.copy()
    result["churn_probability"] = probabilities
    result["churn_prediction"] = predictions
    result["risk_level"] = [decision["level"] for decision in decisions]
    result["recommended_action"] = [decision["action"] for decision in decisions]

    targeted = int(predictions.sum())
    expected_loss_avoided = float(
        probabilities[predictions == 1].sum() * business.rescue_rate * business.cost_churn
    )
    campaign_cost = float(targeted * business.cost_contact)
    summary = (
        f"**{len(result)} clients scorés** — churn moyen prédit: **{probabilities.mean():.1%}** "
        f"(base: {business.cost_churn:.0f} EUR de marge perdue par client).\n\n"
        f"- Clients ciblés (risque ≥ {threshold:.2f}): **{targeted}** "
        f"({targeted / max(len(result), 1):.1%} du portefeuille)\n"
        f"- Valeur attendue si les clients ciblés sont retenus: "
        f"**{expected_loss_avoided:,.0f} EUR**\n"
        f"- Coût de la campagne: **{campaign_cost:,.0f} EUR**\n"
        f"- Gain net potentiel: **{expected_loss_avoided - campaign_cost:,.0f} EUR**"
    )
    return result.reset_index(), summary


def _contract_rows(contract: dict[str, dict[str, Any]]) -> str:
    """Render the input contract as a markdown table."""
    rows = []
    for name, spec in contract.items():
        if spec["type"] == "categorical":
            allowed = ", ".join(str(choice) for choice in spec["choices"])
        else:
            allowed = f"[{spec.get('min')} ; {spec.get('max')}]"
        rows.append(f"| {name} | {spec['type']} | {allowed} |")
    return "\n".join(rows)


def model_card(bundle: dict[str, Any]) -> str:
    """Render the model card (metrics, assumptions, limits)."""
    metrics = bundle.get("metrics", {})
    business = _bundle_context(bundle)
    contract = input_contract()
    rows = _contract_rows(contract)
    metrics_table = "\n".join(
        f"| {key} | {value} |"
        for key, value in metrics.items()
        if isinstance(value, (int, float, str)) and key not in {"model", "params"}
    )
    return f"""## Carte du modèle

### Identité
- Modèle: `{bundle.get("model_name")}`
- Paramètres: `{json.dumps(bundle.get("params", {}))}`
- Seuil décisionnel (optimisé valeur métier): **{float(bundle["threshold"]):.3f}**
- Volume de donnees attendu: {len(contract)} champs clients

### Performances (jeu de test)
| Métrique | Valeur |
|---|---|
{metrics_table}

### Hypothèses économiques
- Marge perdue par client résiliant: **{business.cost_churn:.0f} EUR**
- Coût d'un contact de rétention: **{business.cost_contact:.0f} EUR**
- Taux de rétention des clients contactés: **{business.rescue_rate:.0%}**

### Contrat de données
| Champ | Type | Valeurs attendues |
|---|---|---|
{rows}

### Limites connues
- Modèle entraîné sur un jeu de données public IBM (7 043 clients, 26,5 % de churn) : les
  performances réelles doivent être revalidées sur les données de l'opérateur.
- Le jeu de données ne contient pas de variable de date : pas de validation temporelle possible.
- Les hypothèses économiques (coûts, taux de rétention) sont paramétrables et doivent être
  validées par la direction commerciale avant toute campagne.
- Le score est une aide à la décision : la décision finale reste humaine.
"""


def build_demo(bundle: dict[str, Any] | None = None, model_path: str | Path | None = None):
    """Build the Gradio application."""
    import gradio as gr

    bundle = bundle if bundle is not None else load_bundle(model_path)
    contract = input_contract()
    field_order = ordered_fields(contract)
    essential = [name for name in ESSENTIAL_FIELDS if name in contract]
    extra = [name for name in field_order if name not in essential]

    def _component(name: str):
        spec = contract[name]
        label = name.replace("_", " ").capitalize()
        if spec["type"] == "categorical":
            return gr.Radio(
                choices=list(spec["choices"]),
                value=spec["default"],
                label=label,
            )
        return gr.Number(
            value=spec["default"],
            label=label,
            precision=2,
        )

    with gr.Blocks(title="Predicteur de churn Telco") as demo:
        gr.Markdown(
            "# Predicteur de churn clients \n"
            "**Du processus métier au processus intelligent** : le modèle score chaque client, "
            "la décision est justifiée (SHAP) et l'action est pré-paramétrée."
        )
        with gr.Tabs():
            with gr.Tab("Score client"):
                with gr.Row():
                    with gr.Column(scale=1):
                        inputs = {name: _component(name) for name in essential}
                        # Remaining fields in an accordion
                        with gr.Accordion("Options supplémentaires", open=False):
                            for name in extra:
                                inputs[name] = _component(name)
                        button = gr.Button("Calculer le risque", variant="primary")
                    with gr.Column(scale=2):
                        gauge = gr.Plot(label="Probabilité de résiliation")
                        decision = gr.Markdown()
                        action = gr.Textbox(label="Action recommandée", lines=1)
                        shap_plot = gr.Plot(label="Explication SHAP")
                        shap_table = gr.Dataframe(label="Contributions SHAP", interactive=False)

                def _score(*values):
                    # Map component values back to field names (component order)
                    payload = dict(zip(field_order, values, strict=True))
                    return score_customer(bundle, payload)

                button.click(
                    _score,
                    inputs=list(inputs.values()),
                    outputs=[gauge, decision, action, shap_plot, shap_table],
                )

            with gr.Tab("Scoring de masse"):
                csv_input = gr.File(
                    label="CSV clients (mêmes colonnes que le jeu Telco, 'Churn' optionnel)",
                    file_types=[".csv"],
                    type="filepath",
                )
                csv_button = gr.Button("Scorer le portefeuille", variant="primary")
                csv_summary = gr.Markdown()
                csv_output = gr.Dataframe(label="Clients scorés", interactive=False)
                csv_button.click(
                    lambda path: score_portfolio(bundle, path),
                    inputs=csv_input,
                    outputs=[csv_output, csv_summary],
                )

            with gr.Tab("Carte du modèle"):
                gr.Markdown(model_card(bundle))

    return demo


def main() -> None:
    """Entry point: `churn-gradio`."""
    import gradio as gr

    parser = argparse.ArgumentParser(description="Serve the churn scoring app with Gradio")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()

    demo = build_demo(model_path=args.model_path)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
