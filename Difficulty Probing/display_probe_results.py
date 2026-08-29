"""Display probe classification metrics for 3-bin or 5-bin difficulty labels.

Example:
    python display_probe_results.py --input data/results/codecontests_probe.csv

The same equal-width rating edges are used for the real and predicted ratings,
so the metrics measure whether the probe places a problem in the correct
difficulty range. The output figure is written to ``img/probe_bin_metrics.png``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent


def bin_labels(values, edges):
    """Convert ratings to integer bins, clipping predictions outside the edges."""
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def score_bins(real, predicted, n_bins):
    # Use one set of edges for both series so labels have the same meaning.
    edges = np.linspace(real.min(), real.max(), n_bins + 1)
    y_true = bin_labels(real, edges)
    y_pred = bin_labels(predicted, edges)
    return {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
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
    table = pd.DataFrame(results, index=metrics).T
    print(table.to_string(float_format=lambda value: f"{value:.3f}"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(metrics))
    width = 0.8 / len(choices)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for index, n_bins in enumerate(choices):
        values = [results[n_bins][metric] for metric in metrics]
        offset = (index - (len(choices) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=f"{n_bins} bins")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Difficulty probe bin-classification metrics")
    ax.set_xticks(x, [metric.title() for metric in metrics])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = args.output_dir / "probe_bin_metrics.png"
    fig.savefig(output, dpi=180)
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
