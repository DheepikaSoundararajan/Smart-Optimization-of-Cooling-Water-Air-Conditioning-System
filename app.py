"""
Overall project performance dashboard.

Reads ``output/metrics.json`` (produced by ``ml_model.py``) and renders a
single-page Dash app covering:

  * headline metric scorecard + bar chart (accuracy / precision / recall /
    F1 / ROC-AUC / specificity / balanced accuracy / MCC)
  * 90% requirement gate
  * confusion matrix heatmap
  * ROC curve
  * per-class precision / recall / F1 table
  * sensor feature importance
  * cross-validation stability
  * dataset + model provenance

Run:  python app.py   ->   http://127.0.0.1:8050
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "output" / "metrics.json"

REQUIRED = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "specificity",
    "balanced_accuracy",
    "matthews_corrcoef",
]

NAVY = "#1D3557"
BLUE = "#2E86AB"
TEAL = "#2A9D8F"
RED = "#E63946"
LIGHT = "#F1F5F9"


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise SystemExit(
            f"Missing {METRICS_PATH}.\n"
            "Run the training pipeline first:\n\n    python ml_model.py"
        )
    with open(METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def pct(value: float) -> str:
    return f"{value:.2%}"


def scorecard(head: dict, target: float, mcc_target: float) -> html.Div:
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "roc_auc": "ROC-AUC",
        "specificity": "Specificity",
        "balanced_accuracy": "Balanced Acc.",
        "matthews_corrcoef": "MCC",
    }
    cards = []
    for key in REQUIRED:
        value = head[key]
        # MCC is bounded [-1, 1] and gated separately; the rest are rates.
        gate = mcc_target if key == "matthews_corrcoef" else target
        ok = value >= gate
        cards.append(
            html.Div(
                [
                    html.Div(labels[key], className="kpi-label"),
                    html.Div(
                        pct(value) if key != "matthews_corrcoef" else f"{value:.3f}",
                        className="kpi-value",
                    ),
                    html.Div(
                        "PASS" if ok else "BELOW TARGET",
                        className="kpi-flag " + ("pass" if ok else "fail"),
                    ),
                ],
                className="kpi-card",
            )
        )
    return html.Div(cards, className="kpi-grid")


def build_app(metrics: dict) -> Dash:
    head = metrics["headline_metrics"]
    check = metrics["target_check"]
    target = check["required"]
    mcc_target = check.get("mcc_required", 0.75)
    cm = metrics["confusion_matrix"]
    cv = metrics["cross_validation"]
    data = metrics["dataset"]

    # --- headline bar chart -------------------------------------------------
    headline_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    headline_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=headline_labels,
            y=[head[k] for k in headline_keys],
            text=[pct(head[k]) for k in headline_keys],
            textposition="outside",
            marker_color=BLUE,
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        )
    )
    fig_bar.add_hline(
        y=target,
        line_dash="dash",
        line_color=RED,
        annotation_text=f"Required {target:.0%}",
        annotation_position="bottom right",
    )
    fig_bar.update_layout(
        yaxis=dict(range=[0, 1.08], title="Score"),
        title="Headline Performance vs. 90% Requirement",
        margin=dict(t=60, b=40, l=50, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # --- confusion matrix ---------------------------------------------------
    fig_cm = go.Figure(
        go.Heatmap(
            z=cm["matrix"],
            x=cm["labels"],
            y=cm["labels"],
            colorscale="Blues",
            text=[[f"{v}" for v in row] for row in cm["matrix"]],
            texttemplate="%{text}",
            textfont={"size": 22, "color": NAVY},
            showscale=False,
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix (test split)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
        margin=dict(t=60, b=50, l=60, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=460,
        height=420,
    )

    # --- ROC curve ----------------------------------------------------------
    rc = metrics["roc_curve"]
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=rc["fpr"],
            y=rc["tpr"],
            mode="lines",
            name=f"Model (AUC {head['roc_auc']:.3f})",
            line=dict(color=TEAL, width=3),
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random classifier",
            line=dict(color="#94A3B8", dash="dash"),
        )
    )
    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(t=60, b=50, l=60, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=460,
        height=420,
    )

    # --- feature importance -------------------------------------------------
    fi = metrics["feature_importances"]
    fig_fi = go.Figure(
        go.Bar(
            x=[d["importance"] for d in fi],
            y=[d["feature"] for d in fi],
            orientation="h",
            marker_color="#457B9D",
            text=[f"{d['importance']:.3f}" for d in fi],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig_fi.update_layout(
        title="Sensor Feature Importance (Gini)",
        xaxis=dict(range=[0, max(d["importance"] for d in fi) * 1.2], title="Importance"),
        margin=dict(t=60, b=40, l=90, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=340,
    )

    # --- cross-validation stability ----------------------------------------
    fig_cv = go.Figure(
        go.Bar(
            x=[f"Fold {i + 1}" for i in range(len(cv["fold_scores"]))],
            y=cv["fold_scores"],
            marker_color=TEAL,
            text=[pct(s) for s in cv["fold_scores"]],
            textposition="outside",
        )
    )
    fig_cv.add_hline(
        y=cv["mean_accuracy"],
        line_dash="dot",
        line_color=RED,
        annotation_text=f"mean {pct(cv['mean_accuracy'])}",
    )
    fig_cv.update_layout(
        title=f"{cv['strategy']} Cross-Validation Accuracy",
        yaxis=dict(range=[0.8, 1.02], title="Accuracy"),
        margin=dict(t=60, b=40, l=50, r=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=340,
    )

    # --- per-class table ----------------------------------------------------
    report_rows = []
    for cls, vals in metrics["per_class_report"].items():
        report_rows.append(
            {
                "Class": cls,
                "Precision": pct(vals["precision"]),
                "Recall": pct(vals["recall"]),
                "F1-Score": pct(vals["f1_score"]),
                "Support": vals["support"],
            }
        )

    overall_pass = check["all_pass"]
    gate = html.Div(
        [
            html.Span(
                "90% REQUIREMENT: " + ("ALL HEADLINE METRICS PASS" if overall_pass else "NOT MET"),
                className="gate-text",
            ),
            html.Span(
                f"  accuracy {pct(head['accuracy'])} | precision {pct(head['precision'])} "
                f"| recall {pct(head['recall'])} | F1 {pct(head['f1_score'])} "
                f"| ROC-AUC {pct(head['roc_auc'])}",
                className="gate-detail",
            ),
        ],
        className="gate pass" if overall_pass else "gate fail",
    )

    title = metrics["project_title"]

    app = Dash(__name__)
    app.title = "Cooling Water Optimization | Performance Dashboard"

    app.layout = html.Div(
        [
            # Header
            html.Div(
                [
                    html.H1("Smart Optimization of Cooling Water Air Conditioning System"),
                    html.P(
                        "Machine Learning based energy & water efficiency optimization "
                        "- overall project performance dashboard",
                        className="subtitle",
                    ),
                ],
                className="header",
            ),
            gate,
            # KPI scorecard
            scorecard(head, target, mcc_target),
            # Row 1
            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
                             className="card card-wide"),
                    html.Div(dcc.Graph(figure=fig_cm, config={"displayModeBar": False}),
                             className="card"),
                ],
                className="row",
            ),
            # Row 2
            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_roc, config={"displayModeBar": False}),
                             className="card"),
                    html.Div(dcc.Graph(figure=fig_fi, config={"displayModeBar": False}),
                             className="card card-wide"),
                ],
                className="row",
            ),
            # Row 3: table + CV + provenance
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Per-Class Metrics"),
                            dash_table.DataTable(
                                data=report_rows,
                                columns=[
                                    {"name": c, "id": c}
                                    for c in ["Class", "Precision", "Recall", "F1-Score", "Support"]
                                ],
                                style_header={
                                    "backgroundColor": NAVY,
                                    "color": "white",
                                    "fontWeight": "bold",
                                },
                                style_cell={"textAlign": "center", "padding": "10px"},
                                style_data_conditional=[
                                    {"if": {"row_index": "odd"}, "backgroundColor": LIGHT}
                                ],
                                page_size=5,
                            ),
                            html.H3("Confusion Matrix Detail"),
                            html.Ul(
                                [
                                    html.Li(f"True Negative  : {cm['true_negative']}"),
                                    html.Li(f"False Positive : {cm['false_positive']}"),
                                    html.Li(f"False Negative : {cm['false_negative']}"),
                                    html.Li(f"True Positive  : {cm['true_positive']}"),
                                ],
                                className="detail-list",
                            ),
                        ],
                        className="card",
                    ),
                    html.Div(dcc.Graph(figure=fig_cv, config={"displayModeBar": False}),
                             className="card card-wide"),
                    html.Div(
                        [
                            html.H3("Run Provenance"),
                            html.Ul(
                                [
                                    html.Li(f"Dataset : {data['source']}"),
                                    html.Li(f"Samples : {data['samples']} "
                                            f"(train {data['train_samples']} / test {data['test_samples']})"),
                                    html.Li(f"Features: {', '.join(data['features'])}"),
                                    html.Li(f"Model   : {metrics['model']['type']} "
                                            f"(n_estimators={metrics['model']['n_estimators']}, "
                                            f"max_depth={metrics['model']['max_depth']})"),
                                    html.Li(f"Trained : {metrics['generated_at']}"),
                                ],
                                className="detail-list",
                            ),
                        ],
                        className="card",
                    ),
                ],
                className="row",
            ),
            html.Div(
                "Overall project performance - ML-based Smart Optimization of "
                "Cooling Water Air Conditioning System for Energy and Water Efficiency",
                className="footer",
            ),
        ],
        className="page",
    )

    return app


def main() -> None:
    metrics = load_metrics()
    app = build_app(metrics)
    print("Dashboard: http://127.0.0.1:8050  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=8050, debug=False)


if __name__ == "__main__":
    main()
