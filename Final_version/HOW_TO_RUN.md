# How to Run

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Run the Vietnam real-estate pipeline

```bash
python main.py --dataset vn
```

This will:
- read `VN_BĐS_data/property_cleaned.csv`
- clean and prepare the data
- export `VN_BĐS_data/property_final_clean.csv`
- train and evaluate Ridge, Random Forest, and XGBoost
- save metrics and plots under `result/vn/`

## 3) Run the House Prices pipeline

```bash
python main.py --dataset house
```

This will:
- read `house-prices-data/train.csv` and `house-prices-data/test.csv`
- build features
- train and compare models
- create `submission.csv`
- save metrics and plots under `result/house/`

## 4) Notebook tutorials

The notebooks inside `notebooks/` are documentation copies of the original workflow. They are not required to run the production pipeline.

## 5) Generate report plots

After running one or both pipelines, generate a summary report from the saved CSV files:

```bash
python main.py --report
```

This will create additional report charts under `result/report/`.
