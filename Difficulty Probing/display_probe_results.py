"""Display probe classification metrics for 3-bin or 5-bin difficulty labels.

Example:
    python display_probe_results.py --input data/results/codecontests_probe.csv

The same equal-width rating edges are used for the real and predicted ratings,
so the metrics measure whether the probe places a problem in the correct
difficulty range. The output figure is written to
``img/probe_bin_metrics_detailed.png``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parent


def bin_labels(values, edges):
    """Convert ratings to integer bins, clipping predictions outside the edges."""
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def score_bins(real, predicted, n_bins):
    # Use one set of edges for both series so labels have the same meaning.
    edges = np.linspace(real.min(), real.max(), n_bins + 1)
    y_true = bin_labels(real, edges)
    y_pred = bin_labels(predicted, edges)
    labels = np.arange(n_bins)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    ranges = []
    for index in labels:
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            ranges.append(f"{lower:,.0f}–{upper:,.0f} (inclusive)")
        else:
            ranges.append(f"{lower:,.0f}–<{upper:,.0f}")
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_bin": pd.DataFrame(
            {
                "bin": [f"B{index + 1}" for index in labels],
                "rating range": ranges,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "examples": support,
            }
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/results/codecontests_probe.csv",
        help="CSV containing pred_difficulty and real_difficulty columns",
    )
    parser.add_argument(
        "--bins",
        choices=("3", "5", "both"),
        default="both",
        help="Number of rating bins to display",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "img")
    args = parser.parse_args()

    data = pd.read_csv(args.input)[["pred_difficulty", "real_difficulty"]].dropna()
    if len(data) < 2:
        raise SystemExit("The input must contain at least two complete prediction rows.")

    real = data["real_difficulty"].to_numpy(float)
    predicted = data["pred_difficulty"].to_numpy(float)
    choices = (3, 5) if args.bins == "both" else (int(args.bins),)
    results = {n: score_bins(real, predicted, n) for n in choices}

    metrics = ("precision", "recall", "f1", "accuracy")
    summary = pd.DataFrame(
        {n: {metric: results[n][metric] for metric in metrics} for n in choices}
    ).T
    print("Overall metrics (precision, recall, and F1 are macro averages):")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    for n_bins in choices:
        print(f"\n{n_bins}-bin detail:")
        print(results[n_bins]["per_bin"].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(choices), 1, squeeze=False, figsize=(10, 3.2 * len(choices))
    )
    for axis, n_bins in zip(axes[:, 0], choices):
        detail = results[n_bins]["per_bin"].copy()
        for metric in ("precision", "recall", "f1"):
            detail[metric] = detail[metric].map(lambda value: f"{value:.3f}")
        axis.axis("off")
        axis.set_title(
            f"{n_bins} bins — overall accuracy: {results[n_bins]['accuracy']:.3f}",
            pad=12,
        )
        table = axis.table(
            cellText=detail.values,
            colLabels=detail.columns,
            cellLoc="center",
            loc="center",
            colWidths=[0.10, 0.30, 0.15, 0.15, 0.15, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    fig.tight_layout()
    output = args.output_dir / "probe_bin_metrics_detailed.png"
    fig.savefig(output, dpi=180)
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
