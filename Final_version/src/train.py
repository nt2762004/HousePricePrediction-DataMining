"""Training entry points for the AI project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.model import (
    build_feature_names,
    build_preprocessor,
    eval_on_logscale,
    get_permutation_importance,
    train_random_forest_model,
    train_ridge_model,
    train_xgboost_model,
)
from src.preprocess import (
    VN_MODEL_CATEGORICAL_COLUMNS,
    VN_MODEL_NUMERIC_COLUMNS,
    add_house_features,
    add_vn_base_features,
    build_vn_training_frame,
    export_vn_clean_data,
    load_house_prices,
    load_vn_raw_data,
    prepare_house_frames,
    prepare_vn_clean_dataframe,
)


DEFAULT_RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def _resolve_path(path, base_dir):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _prepare_result_dir(dataset_name):
    result_dir = PROJECT_ROOT / "result" / dataset_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _save_model_comparison_plot(results, output_path, title):
    plot_data = results.sort_values("rmse", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(plot_data["Model"], plot_data["rmse"], color="#4c78a8")
    axes[0].set_title("RMSE")
    axes[0].set_xlabel("RMSE")

    axes[1].barh(plot_data["Model"], plot_data["rmse_log"], color="#f58518")
    axes[1].set_title("RMSE Log")
    axes[1].set_xlabel("RMSE Log")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_series_plot(series, output_path, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_series = series.head(30).sort_values(ascending=True)
    ax.barh(plot_series.index, plot_series.values, color="#54a24b")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_scatter_plot(y_true_log, y_pred_log, output_path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_log, y_pred_log, alpha=0.5, s=18)
    min_val = min(float(np.min(y_true_log)), float(np.min(y_pred_log)))
    max_val = max(float(np.max(y_true_log)), float(np.max(y_pred_log)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--")
    ax.set_xlabel("Giá thực tế (log)")
    ax.set_ylabel("Giá dự đoán (log)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_training_curve(evals_result, output_path, title):
    if not evals_result:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    train_key = next(iter(evals_result))
    train_metric = next(iter(evals_result[train_key]))

    for dataset_name, metrics in evals_result.items():
        values = metrics.get(train_metric)
        if values is not None:
            ax.plot(values, label=dataset_name)

    ax.set_title(title)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(train_metric)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _make_house_final_model(best_name):
    if best_name == "Ridge":
        from sklearn.linear_model import Ridge

        return Ridge(random_state=DEFAULT_RANDOM_STATE, max_iter=10000)

    if best_name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=700,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            max_features=0.8,
            min_samples_leaf=1,
            min_samples_split=2,
            bootstrap=True,
        )

    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=-1,
    )



def run_vn_pipeline(raw_path, clean_output_path):
    result_dir = _prepare_result_dir("vn")
    raw_path = _resolve_path(raw_path, PROJECT_ROOT)
    clean_output_path = _resolve_path(clean_output_path, PROJECT_ROOT)
    df_raw = load_vn_raw_data(raw_path)
    df_clean = prepare_vn_clean_dataframe(df_raw)
    export_vn_clean_data(df_clean, clean_output_path)

    X, y, df_model = build_vn_training_frame(df_clean)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=DEFAULT_RANDOM_STATE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=0.2,
        random_state=DEFAULT_RANDOM_STATE,
    )

    preprocessor = build_preprocessor(VN_MODEL_NUMERIC_COLUMNS, VN_MODEL_CATEGORICAL_COLUMNS)

    ridge_result = train_ridge_model(X_train, y_train, X_valid, y_valid, preprocessor, random_state=DEFAULT_RANDOM_STATE)
    rf_result = train_random_forest_model(X_train, y_train, X_valid, y_valid, preprocessor, random_state=DEFAULT_RANDOM_STATE)
    xgb_result = train_xgboost_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        preprocessor,
        random_state=DEFAULT_RANDOM_STATE,
        n_estimators=2000,
    )

    results = pd.DataFrame(
        [
            {"Model": ridge_result["name"], **ridge_result["metrics"]},
            {"Model": rf_result["name"], **rf_result["metrics"]},
            {"Model": xgb_result["name"], **xgb_result["metrics"]},
        ]
    ).sort_values("rmse")

    print("\nVN results:")
    print(results.to_string(index=False))

    results.to_csv(result_dir / "vn_model_metrics.csv", index=False)
    _save_model_comparison_plot(results, result_dir / "vn_model_comparison.png", "VN Real-Estate Model Comparison")

    feature_names = build_feature_names(xgb_result["preprocessor"], VN_MODEL_NUMERIC_COLUMNS, VN_MODEL_CATEGORICAL_COLUMNS)
    importances = get_permutation_importance(
        xgb_result["estimator"],
        xgb_result["valid_matrix"],
        y_valid,
        feature_names,
        random_state=DEFAULT_RANDOM_STATE,
    )
    print("\nTop VN feature importances:")
    print(importances.head(10).to_string())
    importances.to_csv(result_dir / "vn_feature_importance.csv", header=["importance"])
    _save_series_plot(importances, result_dir / "vn_feature_importance.png", "VN Feature Importance", "Importance")
    _save_training_curve(
        xgb_result.get("evals_result"),
        result_dir / "vn_xgb_training_curve.png",
        "VN XGBoost Training Progress",
    )

    X_full = pd.concat([X_train, X_valid], axis=0)
    y_full = pd.concat([y_train, y_valid], axis=0)
    preprocessor_final = build_preprocessor(VN_MODEL_NUMERIC_COLUMNS, VN_MODEL_CATEGORICAL_COLUMNS).fit(X_full)
    X_full_p = preprocessor_final.transform(X_full)
    X_test_p = preprocessor_final.transform(X_test)

    final_model = xgb_result["estimator"]
    final_model.fit(X_full_p, y_full)
    pred_test_log = final_model.predict(X_test_p)
    test_metrics = eval_on_logscale(pred_test_log, y_test)

    print("\nVN hold-out test metrics:")
    print(test_metrics)
    pd.DataFrame([test_metrics]).to_csv(result_dir / "vn_holdout_metrics.csv", index=False)
    _save_scatter_plot(y_test, pred_test_log, result_dir / "vn_holdout_scatter.png", "VN Hold-out: Predicted vs Actual")

    with open(result_dir / "vn_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "clean_path": str(clean_output_path),
                "test_metrics": test_metrics,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "results": results,
        "clean_path": clean_output_path,
        "test_metrics": test_metrics,
        "model": final_model,
        "preprocessor": preprocessor_final,
    }



def run_house_pipeline(train_path, test_path, output_path):
    result_dir = _prepare_result_dir("house")
    train_path = _resolve_path(train_path, PROJECT_ROOT)
    test_path = _resolve_path(test_path, PROJECT_ROOT)
    output_path = _resolve_path(output_path, PROJECT_ROOT)
    train_df, test_df = load_house_prices(train_path, test_path)
    train_df, test_df = prepare_house_frames(train_df, test_df)

    train_df["SalePrice_log"] = np.log1p(train_df["SalePrice"])
    X = train_df.drop(columns=["SalePrice", "SalePrice_log"])
    y = train_df["SalePrice_log"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=DEFAULT_RANDOM_STATE,
    )

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [col for col in num_cols if col != "Id"]

    preprocessor = build_preprocessor(num_cols, cat_cols)

    ridge_result = train_ridge_model(X_train, y_train, X_valid, y_valid, preprocessor, random_state=DEFAULT_RANDOM_STATE)
    rf_result = train_random_forest_model(X_train, y_train, X_valid, y_valid, preprocessor, random_state=DEFAULT_RANDOM_STATE)
    xgb_result = train_xgboost_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        preprocessor,
        random_state=DEFAULT_RANDOM_STATE,
        n_estimators=1000,
    )

    results = pd.DataFrame(
        [
            {"Model": ridge_result["name"], **ridge_result["metrics"]},
            {"Model": rf_result["name"], **rf_result["metrics"]},
            {"Model": xgb_result["name"], **xgb_result["metrics"]},
        ]
    ).sort_values("rmse")

    print("\nHouse Prices results:")
    print(results.to_string(index=False))
    results.to_csv(result_dir / "house_model_metrics.csv", index=False)
    _save_model_comparison_plot(results, result_dir / "house_model_comparison.png", "House Prices Model Comparison")
    _save_training_curve(
        xgb_result.get("evals_result"),
        result_dir / "house_xgb_training_curve.png",
        "House Prices XGBoost Training Progress",
    )

    best_name = results.iloc[0]["Model"]

    X_full = pd.concat([X_train, X_valid], axis=0)
    y_full = pd.concat([y_train, y_valid], axis=0)
    preprocessor_full = build_preprocessor(num_cols, cat_cols).fit(X_full)
    X_full_p = preprocessor_full.transform(X_full)
    X_test_p = preprocessor_full.transform(test_df)

    best_model = _make_house_final_model(best_name)
    best_model.fit(X_full_p, y_full)

    test_preds_log = best_model.predict(X_test_p)
    submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": np.expm1(test_preds_log)})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    submission.to_csv(result_dir / "house_submission.csv", index=False)

    print(f"\nSaved submission to {output_path}")
    print(f"Saved additional results to {result_dir}")

    with open(result_dir / "house_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "submission_path": str(output_path),
                "best_model": best_name,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "results": results,
        "submission_path": output_path,
        "model": best_model,
        "preprocessor": preprocessor_full,
    }



def build_parser():
    parser = argparse.ArgumentParser(description="Run AI project training pipelines.")
    parser.add_argument("--dataset", choices=["vn", "house"], default="vn", help="Choose which pipeline to run.")
    parser.add_argument("--vn-raw", default=str(PROJECT_ROOT / "data" / "VN_BĐS_data" / "property_cleaned.csv"), help="Path to the raw VN data CSV.")
    parser.add_argument("--vn-clean", default=str(PROJECT_ROOT / "data" / "VN_BĐS_data" / "property_final_clean.csv"), help="Where to save the cleaned VN data.")
    parser.add_argument("--house-train", default=str(PROJECT_ROOT / "data" / "house-prices-data" / "train.csv"), help="Path to the Kaggle training CSV.")
    parser.add_argument("--house-test", default=str(PROJECT_ROOT / "data" / "house-prices-data" / "test.csv"), help="Path to the Kaggle test CSV.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "submission.csv"), help="Output path for generated predictions.")
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == "vn":
        return run_vn_pipeline(args.vn_raw, args.vn_clean)
    return run_house_pipeline(args.house_train, args.house_test, args.output)


if __name__ == "__main__":
    main()
