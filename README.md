# Data Mining Project: Real Estate Price Analysis and Prediction

This project contains Python source code (Jupyter Notebooks) to perform the data mining process, from data cleaning and exploratory data analysis (EDA) to building house price prediction models.

The project works on 2 main datasets:
1.  **House Prices Data** (Taken from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition on Kaggle).
2.  **Vietnam Real Estate Data** (Crawled from the website [Nha dat Cafeland](https://nhadat.cafeland.vn/)).

## Directory Structure

```
├── eda_vn_bds.ipynb              # Notebook for cleaning and EDA for VN data
├── VNBĐS_predict.ipynb           # Notebook for training price prediction model for VN data
├── price_predict.ipynb           # Notebook for price prediction for House Prices dataset
├── README.md                     # Project description file
├── README_vnbđs.md               # Detailed description file for Vietnam Real Estate prediction
├── VN_BĐS_data/                  # Folder containing Vietnam Real Estate data
│   ├── property_cleaned.csv      # Initial raw data
│   └── property_final_clean.csv  # Cleaned data (input for model)
├── house-prices-data/            # Folder containing House Prices data
    ├── train.csv
    ├── test.csv
    └── ...
└── batdongsan.ipynb              # Script to crawl VN real estate data
```

## Notebook Details

### 1. `price_predict.ipynb` (House Prices Prediction)
This notebook does a similar process but applies it to the House Prices dataset on Kaggle.

*   **Goal:** Practice and compare the process on a standard dataset.
*   **Main steps:**
    *   **Load Data:** Read `train.csv` and `test.csv`.
    *   **Data Analysis:** Check data types, missing data, house price distribution.
    *   **Feature Engineering:** Create combined features like `TotalSF` (Total Area), `TotalBath` (Total Bathrooms), `HouseAge` (House Age).
    *   **Processing Pipeline:**
        *   Fill missing data (Imputation).
        *   Scaling and Encoding.
    *   **Model Training:**
        *   **Ridge Regression:** Linear regression with regularization.
        *   **Random Forest:** Decision tree model.
        *   **XGBoost:** Powerful Boosting model.
    *   **Evaluation:** Compare RMSE between models and draw comparison charts.
    *   **Submission:** Create prediction result file for the Test set.

### 2. `eda_vn_bds.ipynb` (Cleaning & EDA - VN Real Estate)
This notebook performs preprocessing and initial data analysis for the Vietnam Real Estate dataset.

*   **Goal:** Convert raw data into clean data, remove noise, and understand data distribution.
*   **Main steps:**
    *   **Data Cleaning:**
        *   Process text strings to extract numbers for Area, Number of bedrooms, Number of toilets, Number of floors, Frontage road.
        *   Standardize price unit to **Million VND** (handle cases like "billion", "million", "price/m2").
    *   **Handling Missing Values:**
        *   Fill data smartly based on type (House vs Land). For example: Land has 0 rooms, House uses median value.
    *   **Outlier Removal:**
        *   Remove abnormal values for area (< 10m2 or > 1000m2), unit price (too cheap or too expensive), and total price.
        *   Use Percentile method to cut extreme distribution tails.
    *   **Visualization:**
        *   Price distribution chart (Histogram).
        *   Correlation chart between Price and Area by City/Province.
        *   Correlation matrix (Heatmap).
    *   **Result:** Export `property_final_clean.csv` file to use for model training.

### 3. `VNBĐS_predict.ipynb` (Modeling - VN Real Estate)
This notebook focuses on building and evaluating machine learning models to predict house prices in Vietnam.

*   **Goal:** Build the most accurate house price prediction model possible.
*   **Main steps:**
    *   **Feature Engineering (Create new features):**
        *   Create combined variables like `Type_City` (Type + City).
        *   Calculate `total_floor_area`, `road_potential`.
        *   Use **Target Encoding** for categorical variables.
    *   **Preprocessing:**
        *   Logarithmize target variable (House Price) to make it normal distribution (`np.log1p`).
        *   Standardize numerical data (StandardScaler) and One-Hot encoding for categorical data.
    *   **Model Training:**
        *   Test models: **Ridge Regression**, **Random Forest**, **XGBoost**.
        *   Use `GridSearchCV` to find optimal parameters.
        *   Evaluate model using RMSE and R2 Score on both Log scale and real scale.
    *   **Evaluation & Explanation:**
        *   Compare performance between models.
        *   Analyze feature importance (Permutation Importance).
        *   Check on independent Test set (Hold-out set).

## Installation Requirements

To run the notebooks, you need to install the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost category_encoders
```