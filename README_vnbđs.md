## 1. `eda_vn_bds.ipynb` (Exploratory Data Analysis & Data Cleaning)

Notebook này thực hiện các bước làm sạch dữ liệu thô và phân tích khám phá dữ liệu (EDA) cho bộ dữ liệu Bất động sản Việt Nam.

### Chi tiết các bước (Cells):

1.  **Setup & Load dữ liệu**:
    *   Import các thư viện cần thiết (`pandas`, `numpy`, `seaborn`, `matplotlib`).
    *   Đọc dữ liệu thô từ file `VN_BĐS_data/property_cleaned.csv`.

2.  **Định nghĩa hàm làm sạch (Cleaning Functions)**:
    *   `clean_area(x)`: Chuẩn hóa cột diện tích (loại bỏ 'm2', chuyển sang số thực).
    *   `extract_number(x)`: Trích xuất số từ chuỗi (dùng cho số phòng ngủ, toilet, tầng).
    *   `clean_road(x)`: Xử lý độ rộng đường (lấy trung bình nếu là khoảng giá trị).
    *   `clean_price_final(row)`: Hàm quan trọng nhất để chuẩn hóa giá nhà về đơn vị **Triệu VNĐ**. Xử lý các trường hợp đơn vị 'tỷ', 'triệu', và giá trên m2.
    *   **Áp dụng**: Chạy các hàm trên để tạo các cột sạch: `cleaned_area`, `cleaned_price`, `cleaned_road`, `bed`, `bath`, `floor`.

3.  **Xử lý dữ liệu thiếu & Feature Engineering cơ bản**:
    *   Tạo cột `is_land`: Phân loại là "Đất" hay "Nhà".
    *   Điền dữ liệu thiếu (Imputation) dựa trên logic:
        *   Nếu là Đất: Số phòng/tầng = 0.
        *   Nếu là Nhà: Điền bằng giá trị trung vị (median) của nhóm Nhà.
    *   Tạo cột `price_per_m2`: Tính đơn giá trên m2 để phục vụ việc lọc nhiễu.

4.  **Lọc nhiễu (Outlier Removal)**:
    *   Lọc theo Diện tích: Giữ lại từ 10m2 đến 1000m2.
    *   Lọc theo Đơn giá/m2: Giữ lại từ 2 triệu/m2 đến 500 triệu/m2.
    *   Lọc theo Tổng giá: Giữ lại từ 100 triệu đến 100 Tỷ.
    *   Lọc theo Percentile: Cắt bỏ 1% giá trị nhỏ nhất và lớn nhất của giá và diện tích để loại bỏ các giá trị cực đoan còn sót lại.

5.  **Trực quan hóa (Visualization)**:
    *   Biểu đồ phân phối giá (Histogram) trước và sau khi log-transform.
    *   Biểu đồ Scatter plot: Tương quan giữa Diện tích và Giá (phân theo Tỉnh/Thành phố).
    *   Heatmap: Ma trận tương quan giữa các biến số.

6.  **Xuất dữ liệu (Export)**:
    *   Lưu bộ dữ liệu đã làm sạch hoàn chỉnh vào file `VN_BĐS_data/property_final_clean.csv` để dùng cho việc training model.

---

## 2. `VNBĐS_predict.ipynb` (Modeling & Prediction)

Notebook này xây dựng, huấn luyện và đánh giá các mô hình học máy để dự đoán giá bất động sản dựa trên dữ liệu đã làm sạch.

### Chi tiết các bước (Cells):

1.  **Setup & Config**:
    *   Import thư viện (`sklearn`, `xgboost`, v.v.).
    *   Định nghĩa hàm đánh giá `eval_on_logscale`: Tính RMSE và R2 score trên cả thang đo log và thang đo thực tế.

2.  **Load dữ liệu**:
    *   Đọc file `VN_BĐS_data/property_final_clean.csv`.

3.  **Tổng quan & Xử lý sơ bộ**:
    *   Kiểm tra dữ liệu thiếu còn sót lại.
    *   Xử lý cột `Loại địa ốc` (xóa dòng thiếu ít) và `Pháp lý` (điền 'Chưa xác định').

4.  **Phân tích biến mục tiêu (Target Analysis)**:
    *   Vẽ biểu đồ phân phối giá nhà.
    *   Thực hiện **Log Transformation** (`np.log1p`) lên giá nhà để phân phối chuẩn hơn, giúp mô hình hồi quy hoạt động tốt hơn.

5.  **Phân tích tương quan (Correlation)**:
    *   Sử dụng `TargetEncoder` để mã hóa các biến phân loại (categorical) nhằm tính toán độ tương quan với giá nhà.
    *   Vẽ Heatmap cho Top 10 biến số (numeric) và Top 10 biến phân loại có tương quan mạnh nhất.

6.  **Trực quan hóa chi tiết**:
    *   Vẽ biểu đồ phân phối cho các biến số quan trọng.
    *   Vẽ biểu đồ tròn (Pie chart) và cột (Bar chart) cho các biến phân loại.

7.  **Feature Engineering (Tạo đặc trưng mới)**:
    *   `Type_City`: Kết hợp Loại nhà + Tỉnh thành.
    *   `total_floor_area`: Tổng diện tích sàn sử dụng.
    *   `total_rooms`: Tổng số phòng.
    *   `road_potential`: Tiềm năng mặt tiền (Diện tích * Độ rộng đường).

8.  **Chuẩn bị dữ liệu (Preprocessing)**:
    *   Tách biến mục tiêu `y` (log price) và biến đầu vào `X`.
    *   Xây dựng `ColumnTransformer`:
        *   Biến số: Chuẩn hóa bằng `StandardScaler`.
        *   Biến phân loại: Mã hóa bằng `OneHotEncoder`.

9.  **Chia tập dữ liệu (Train/Valid/Test)**:
    *   Tách riêng 10% dữ liệu làm tập **Test (Hold-out)** để đánh giá độc lập cuối cùng.
    *   90% còn lại được chia tiếp thành **Train** (80%) và **Validation** (20%).

10. **Huấn luyện mô hình (Modeling)**:
    *   **Ridge Regression**: Mô hình hồi quy tuyến tính có điều chuẩn (Regularization). Sử dụng `GridSearchCV` để tìm tham số `alpha` tốt nhất.
    *   **Random Forest**: Mô hình rừng ngẫu nhiên.
    *   **XGBoost**: Mô hình Gradient Boosting mạnh mẽ. Sử dụng `eval_set` để theo dõi quá trình huấn luyện trên tập Validation.

11. **So sánh kết quả**:
    *   Tổng hợp RMSE và R2 của 3 mô hình.
    *   Vẽ biểu đồ so sánh hiệu năng.

12. **Feature Importance**:
    *   Sử dụng **Permutation Importance** trên mô hình XGBoost để xác định các yếu tố ảnh hưởng nhiều nhất đến giá nhà.

13. **Đánh giá cuối cùng (Final Evaluation)**:
    *   Gộp tập Train và Validation để huấn luyện lại mô hình tốt nhất (XGBoost) trên nhiều dữ liệu hơn.
    *   Dự đoán trên tập **Test (Hold-out)**.
    *   Hiển thị bảng so sánh Giá thực tế vs Giá dự đoán và sai số.
    *   Vẽ biểu đồ Scatter plot dự đoán vs thực tế.

---