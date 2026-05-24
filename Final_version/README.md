This folder is the refactored version of the original notebook project.

## Structure

```text
Final_version/
├── data/               # Sample data, lightweight exports only
├── notebooks/          # Tutorial notebooks / documentation space
├── src/                # Main source code
│   ├── __init__.py
│   ├── preprocess.py   # Data cleaning and feature engineering
│   ├── model.py        # Models, metrics, and evaluation helpers
│   └── train.py        # Training pipelines for each dataset
├── requirements.txt    # Python dependencies
├── main.py             # Main entry point
└── README.md           # Project overview
```

## What changed

- Notebook logic has been moved into reusable Python modules under `src/`.
- `main.py` is the entry point for running a dataset pipeline.
- `requirements.txt` lists the runtime dependencies.
- The original notebooks are kept as tutorial material under `notebooks/`.
- `.gitignore` excludes large outputs, caches, and generated artifacts.

## How to run

### Vietnam real-estate pipeline

```bash
python main.py --dataset vn
```

### House Prices pipeline

```bash
python main.py --dataset house
```

### Generate report charts

```bash
python main.py --report
```

## Output

- The VN pipeline exports the cleaned dataset to `data/VN_BĐS_data/property_final_clean.csv`.
- The House Prices pipeline exports `submission.csv`.
- The report command creates summary charts under `result/report/`.

## Results Gallery

The figures below are generated automatically after running the pipelines and are stored under `result/`.

### Vietnam Real-Estate

#### Model Comparison

![VN Model Comparison](result/vn/vn_model_comparison.png)

#### XGBoost Training Curve

![VN XGBoost Training Curve](result/vn/vn_xgb_training_curve.png)

#### Feature Importance

![VN Feature Importance](result/vn/vn_feature_importance.png)

#### Hold-out Prediction vs Actual

![VN Hold-out Scatter](result/vn/vn_holdout_scatter.png)

### House Prices

#### Model Comparison

![House Model Comparison](result/house/house_model_comparison.png)

#### XGBoost Training Curve

![House XGBoost Training Curve](result/house/house_xgb_training_curve.png)

#### Submission Distribution

If you generate the report with `python main.py --report`, GitHub will also show summary charts under `result/report/`.
