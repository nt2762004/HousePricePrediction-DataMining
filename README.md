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

---

# Phân tích và Dự đoán Giá Bất Động Sản

Dự án này bao gồm các mã nguồn Python (Jupyter Notebooks) để thực hiện quy trình khai phá dữ liệu (Data Mining), từ làm sạch dữ liệu, phân tích khám phá (EDA) đến xây dựng mô hình dự đoán giá nhà.

Dự án làm việc trên 2 bộ dữ liệu chính:
1.  **Dữ liệu House Prices** (Được lấy từ cuộc thi [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) trên Kaggle.).
2.  **Dữ liệu Bất động sản Việt Nam** (Được cào từ trang web [Nhà đất Cafeland](https://nhadat.cafeland.vn/)).

## Cấu trúc Thư mục

```
├── eda_vn_bds.ipynb              # Notebook làm sạch và EDA cho dữ liệu VN
├── VNBĐS_predict.ipynb           # Notebook huấn luyện mô hình dự đoán giá cho dữ liệu VN
├── price_predict.ipynb           # Notebook dự đoán giá cho bộ dữ liệu House Prices
├── README.md                     # File mô tả dự án
├── README_vnbđs.md               # File mô tả chi tiết dự đoán BĐS Việt Nam
├── VN_BĐS_data/                  # Thư mục chứa dữ liệu BĐS Việt Nam
│   ├── property_cleaned.csv      # Dữ liệu thô ban đầu
│   └── property_final_clean.csv  # Dữ liệu sau khi đã làm sạch (đầu vào cho model)
├── house-prices-data/            # Thư mục chứa dữ liệu House Prices
    ├── train.csv
    ├── test.csv
    └── ...
└── batdongsan.ipynb              # Script cào data bất động sản VN
```

## Chi tiết các Notebook

### 1. `price_predict.ipynb` (Dự đoán House Prices)
Notebook này thực hiện quy trình tương tự nhưng áp dụng cho bộ dữ liệu House Prices ở trên Kaggle.

*   **Mục tiêu:** Thực hành và so sánh quy trình trên một bộ dữ liệu chuẩn khác.
*   **Các bước chính:**
    *   **Load dữ liệu:** Đọc `train.csv` và `test.csv`.
    *   **Phân tích dữ liệu:** Kiểm tra kiểu dữ liệu, dữ liệu thiếu, phân phối giá nhà.
    *   **Feature Engineering:** Tạo các đặc trưng tổng hợp như `TotalSF` (Tổng diện tích), `TotalBath` (Tổng số phòng tắm), `HouseAge` (Tuổi nhà).
    *   **Pipeline xử lý:**
        *   Điền dữ liệu thiếu (Imputation).
        *   Chuẩn hóa (Scaling) và Mã hóa (Encoding).
    *   **Huấn luyện mô hình:**
        *   **Ridge Regression:** Hồi quy tuyến tính có điều chuẩn.
        *   **Random Forest:** Mô hình cây quyết định.
        *   **XGBoost:** Mô hình Boosting mạnh mẽ.
    *   **Đánh giá:** So sánh RMSE giữa các mô hình và vẽ biểu đồ so sánh.
    *   **Submission:** Tạo file kết quả dự đoán cho tập Test.

### 2. `eda_vn_bds.ipynb` (Làm sạch & EDA - VN BĐS)
Notebook này thực hiện các bước tiền xử lý và phân tích dữ liệu ban đầu cho bộ dữ liệu Bất động sản Việt Nam.

*   **Mục tiêu:** Chuyển đổi dữ liệu thô thành dữ liệu sạch, loại bỏ nhiễu và hiểu rõ phân phối dữ liệu.
*   **Các bước chính:**
    *   **Làm sạch dữ liệu (Data Cleaning):**
        *   Xử lý chuỗi ký tự để trích xuất số liệu cho Diện tích, Số phòng ngủ, Số toilet, Số tầng, Đường trước nhà.
        *   Chuẩn hóa đơn vị giá về **Triệu VNĐ** (xử lý các trường hợp "tỷ", "triệu", "giá/m2").
    *   **Xử lý dữ liệu thiếu (Missing Values):**
        *   Điền dữ liệu thông minh dựa trên loại hình (Nhà vs Đất). Ví dụ: Đất thì số phòng = 0, Nhà thì điền bằng giá trị trung vị (median).
    *   **Lọc nhiễu (Outlier Removal):**
        *   Loại bỏ các giá trị bất thường về diện tích (< 10m2 hoặc > 1000m2), đơn giá (quá rẻ hoặc quá đắt) và tổng giá.
        *   Sử dụng phương pháp Percentile để cắt đuôi phân phối cực đoan.
    *   **Trực quan hóa (Visualization):**
        *   Biểu đồ phân phối giá (Histogram).
        *   Biểu đồ tương quan giữa Giá và Diện tích theo Tỉnh/Thành.
        *   Ma trận tương quan (Heatmap).
    *   **Kết quả:** Xuất file `property_final_clean.csv` để dùng cho việc huấn luyện mô hình.

### 3. `VNBĐS_predict.ipynb` (Mô hình hóa - VN BĐS)
Notebook này tập trung vào việc xây dựng và đánh giá các mô hình máy học để dự đoán giá nhà tại Việt Nam.

*   **Mục tiêu:** Xây dựng mô hình dự đoán giá nhà chính xác nhất có thể.
*   **Các bước chính:**
    *   **Feature Engineering (Tạo đặc trưng mới):**
        *   Tạo các biến kết hợp như `Type_City` (Loại nhà + Tỉnh).
        *   Tính toán `total_floor_area` (Tổng diện tích sàn), `road_potential` (Tiềm năng mặt tiền).
        *   Sử dụng **Target Encoding** cho các biến phân loại.
    *   **Tiền xử lý (Preprocessing):**
        *   Logarit hóa biến mục tiêu (Giá nhà) để đưa về phân phối chuẩn (`np.log1p`).
        *   Chuẩn hóa dữ liệu số (StandardScaler) và mã hóa One-Hot cho dữ liệu phân loại.
    *   **Huấn luyện Mô hình (Modeling):**
        *   Thử nghiệm các mô hình: **Ridge Regression**, **Random Forest**, **XGBoost**.
        *   Sử dụng `GridSearchCV` để tìm tham số tối ưu.
        *   Đánh giá mô hình bằng RMSE và R2 Score trên cả thang đo Log và thang đo thực tế.
    *   **Đánh giá & Giải thích:**
        *   So sánh hiệu quả giữa các mô hình.
        *   Phân tích độ quan trọng của các đặc trưng (Permutation Importance).
        *   Kiểm tra trên tập Test độc lập (Hold-out set).

## Yêu cầu cài đặt

Để chạy các notebook cần cài đặt các thư viện Python sau:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost category_encoders
```
