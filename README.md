# Dự án Khai phá Dữ liệu: Phân tích và Dự đoán Giá Bất Động Sản

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
