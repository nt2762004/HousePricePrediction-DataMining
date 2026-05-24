"""Entry point for the AI project."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.report import main as report_main
from src.train import run_house_pipeline, run_vn_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Run the AI project pipelines or generate reports.")
    parser.add_argument("--dataset", choices=["vn", "house"], help="Run a training pipeline for the selected dataset.")
    parser.add_argument("--report", action="store_true", help="Generate summary charts from saved results.")
    parser.add_argument("--vn-raw", default=str(PROJECT_ROOT / "data" / "VN_BĐS_data" / "property_cleaned.csv"), help="Path to the raw VN data CSV.")
    parser.add_argument("--vn-clean", default=str(PROJECT_ROOT / "data" / "VN_BĐS_data" / "property_final_clean.csv"), help="Where to save the cleaned VN data.")
    parser.add_argument("--house-train", default=str(PROJECT_ROOT / "data" / "house-prices-data" / "train.csv"), help="Path to the Kaggle training CSV.")
    parser.add_argument("--house-test", default=str(PROJECT_ROOT / "data" / "house-prices-data" / "test.csv"), help="Path to the Kaggle test CSV.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "submission.csv"), help="Output path for generated predictions.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.report:
        return report_main()

    if args.dataset == "vn":
        return run_vn_pipeline(args.vn_raw, args.vn_clean)

    if args.dataset == "house":
        return run_house_pipeline(args.house_train, args.house_test, args.output)

    parser.error("Specify either --report or --dataset {vn,house}.")


if __name__ == "__main__":
    main()
