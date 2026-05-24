"""Generate summary charts from saved pipeline results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result"
REPORT_DIR = RESULT_ROOT / "report"


def _ensure_report_dir():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_DIR



def _load_metrics(dataset_name):
    metrics_path = RESULT_ROOT / dataset_name / f"{dataset_name}_model_metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)



def _plot_metric_comparison(metrics_df, title, output_path):
    plot_df = metrics_df.sort_values("rmse", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(plot_df["Model"], plot_df["rmse"], color="#4c78a8")
    axes[0].set_title("RMSE")
    axes[0].set_xlabel("RMSE")

    axes[1].barh(plot_df["Model"], plot_df["rmse_log"], color="#f58518")
    axes[1].set_title("RMSE Log")
    axes[1].set_xlabel("RMSE Log")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def _plot_house_submission(output_path):
    submission_path = RESULT_ROOT / "house" / "house_submission.csv"
    if not submission_path.exists():
        return

    submission = pd.read_csv(submission_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(submission["SalePrice"], bins=30, color="#54a24b", edgecolor="white")
    ax.set_title("House Prices Submission Distribution")
    ax.set_xlabel("Predicted SalePrice")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def main():
    report_dir = _ensure_report_dir()

    vn_metrics = _load_metrics("vn")
    if vn_metrics is not None:
        _plot_metric_comparison(vn_metrics, "VN Real-Estate Model Comparison", report_dir / "vn_report_metrics.png")

    house_metrics = _load_metrics("house")
    if house_metrics is not None:
        _plot_metric_comparison(house_metrics, "House Prices Model Comparison", report_dir / "house_report_metrics.png")
        _plot_house_submission(report_dir / "house_submission_distribution.png")

    print(f"Report charts saved to {report_dir}")


if __name__ == "__main__":
    main()
