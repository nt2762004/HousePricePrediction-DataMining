"""Data preparation helpers for the Vietnam real-estate and House Prices datasets."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


VN_FINAL_COLUMNS = [
    "cleaned_area",
    "cleaned_road",
    "bed",
    "bath",
    "floor",
    "is_land",
    "tag",
    "Loại địa ốc",
    "Pháp lý",
    "cleaned_price",
]

VN_MODEL_NUMERIC_COLUMNS = [
    "cleaned_area",
    "cleaned_road",
    "bed",
    "bath",
    "floor",
    "is_land",
    "total_floor_area",
    "road_potential",
    "total_rooms",
]

VN_MODEL_CATEGORICAL_COLUMNS = ["tag", "Pháp lý", "Loại địa ốc", "Type_City"]

HOUSE_EXTRA_FEATURE_COLUMNS = ["TotalSF", "TotalBath", "HouseAge", "RemodAge"]


def clean_area(x):
    """Convert strings like '77m2' to floats."""
    if pd.isna(x):
        return np.nan
    x = str(x).lower().replace("m2", "").replace(",", ".").strip()
    try:
        value = float(x)
        return value if value > 0 else np.nan
    except Exception:
        return np.nan



def extract_number(x):
    """Extract the first numeric value from a string."""
    if pd.isna(x):
        return np.nan
    match = re.search(r"(\d+(\.\d+)?)", str(x))
    return float(match.group(1)) if match else np.nan



def clean_road(x):
    """Convert road width strings like '10m-12m' to a numeric midpoint."""
    if pd.isna(x):
        return np.nan
    x = str(x).lower().replace("m", "").strip()
    try:
        if "-" in x:
            left, right = x.split("-")
            return (float(left) + float(right)) / 2
        return float(x)
    except Exception:
        return np.nan



def clean_price_final(row):
    """Normalize price into million VND."""
    price_str = str(row["Giá nhà"]).lower()
    area = row["cleaned_area"]

    if pd.isna(price_str) or pd.isna(area) or area == 0:
        return np.nan

    s = price_str.replace("/m2", "").replace(",", ".")
    billion, million = 0.0, 0.0

    try:
        if "tỷ" in s:
            billion_match = re.search(r"([\d\.]+)\s*tỷ", s)
            if billion_match:
                billion = float(billion_match.group(1))
        if "triệu" in s:
            million_match = re.search(r"([\d\.]+)\s*triệu", s)
            if million_match:
                million = float(million_match.group(1))
    except Exception:
        pass

    value_million = (billion * 1000) + million
    if value_million == 0:
        return np.nan

    if "/m2" in price_str or value_million < 100:
        value_million = value_million * area

    return value_million



def load_vn_raw_data(file_path):
    return pd.read_csv(file_path, low_memory=False)



def add_vn_base_features(df):
    df = df.copy()
    df["cleaned_area"] = df["Diện tích"].apply(clean_area)
    df["cleaned_price"] = df.apply(clean_price_final, axis=1)
    df["cleaned_road"] = df["Đường trước nhà"].apply(clean_road)
    df["bed"] = df["Phòng ngủ"].apply(extract_number)
    df["bath"] = df["Số toilet"].apply(extract_number)
    df["floor"] = df["Số tầng"].apply(extract_number)
    return df



def remove_outliers_percentile(df, col, lower=0.01, upper=0.99):
    low_val = df[col].quantile(lower)
    high_val = df[col].quantile(upper)
    return df[(df[col] >= low_val) & (df[col] <= high_val)]



def prepare_vn_clean_dataframe(df):
    df = add_vn_base_features(df)
    df = df.dropna(subset=["cleaned_price", "cleaned_area"]).copy()
    df = df.dropna(subset=["Loại địa ốc"]).copy()

    df["Loại địa ốc"] = df["Loại địa ốc"].astype(str).str.lower()
    df["is_land"] = df["Loại địa ốc"].apply(lambda value: 1 if "đất" in value else 0)

    for col in ["bed", "bath", "floor"]:
        df.loc[df["is_land"] == 1, col] = df.loc[df["is_land"] == 1, col].fillna(0)
        median_val = df.loc[df["is_land"] == 0, col].median()
        df.loc[df["is_land"] == 0, col] = df.loc[df["is_land"] == 0, col].fillna(median_val)

    if "cleaned_road" in df.columns:
        df["cleaned_road"] = df["cleaned_road"].fillna(df["cleaned_road"].median())
    else:
        df["cleaned_road"] = np.nan

    df["price_per_m2"] = df["cleaned_price"] / df["cleaned_area"]

    df = df[(df["cleaned_area"] >= 10) & (df["cleaned_area"] <= 1000)]
    df = df[(df["price_per_m2"] >= 2) & (df["price_per_m2"] <= 500)]
    df = df[(df["cleaned_price"] >= 100) & (df["cleaned_price"] <= 100000)]
    df = remove_outliers_percentile(df, "cleaned_price")
    df = remove_outliers_percentile(df, "cleaned_area")

    return df.reset_index(drop=True)



def build_vn_training_frame(df):
    df = df.copy()
    df["Type_City"] = df["Loại địa ốc"].fillna("").astype(str) + "_" + df["tag"].fillna("").astype(str)
    df["total_floor_area"] = df["cleaned_area"] * df["floor"].replace(0, 1)
    df["total_rooms"] = df["bed"] + df["bath"]
    df["road_potential"] = df["cleaned_area"] * df["cleaned_road"]

    features = VN_MODEL_NUMERIC_COLUMNS + VN_MODEL_CATEGORICAL_COLUMNS
    X = df[features].copy()
    y = np.log1p(df["cleaned_price"])
    return X, y, df



def export_vn_clean_data(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[VN_FINAL_COLUMNS].to_csv(output_path, index=False)
    return output_path



def load_house_prices(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df



def add_house_features(df):
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    return df



def prepare_house_frames(train_df, test_df):
    train_df = add_house_features(train_df)
    test_df = add_house_features(test_df)
    return train_df, test_df
