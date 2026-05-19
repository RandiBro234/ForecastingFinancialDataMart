# preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_forecasting_data(engine):
    query = """
    SELECT *
    FROM v_forecasting_bulanan
    ORDER BY tahun, bulan
    """

    df = pd.read_sql(query, engine)

    df["periode_index"] = range(1, len(df) + 1)
    df["periode_date"] = pd.to_datetime(df["periode"] + "-01")

    numeric_columns = [
        "total_pendapatan",
        "total_biaya",
        "total_anggaran",
        "total_selisih",
        "rata_rata_pendapatan",
        "rata_rata_biaya",
        "jumlah_transaksi",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def check_missing_values(df):
    print("\n=== Missing Values ===")
    print(df.isnull().sum())


def check_duplicates(df):
    print("\n=== Duplicate Rows ===")
    print(df.duplicated().sum())


def descriptive_statistics(df):
    print("\n=== Statistik Deskriptif ===")
    print(df.describe())


def check_time_period(df):
    print("\n=== Cek Periode Data ===")
    print("Periode awal :", df["periode"].min())
    print("Periode akhir:", df["periode"].max())
    print("Jumlah periode:", len(df))


def handle_missing_values(df):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["number"]).columns

    # interpolasi linear
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

    # forward fill & backward fill
    df[numeric_cols] = df[numeric_cols].bfill().ffill()

    return df


def add_log_transform(df, target_col="total_pendapatan"):
    df = df.copy()
    df[f"{target_col}_log"] = np.log1p(df[target_col])
    return df


def train_test_split_time_series(df, target_col="total_pendapatan", train_ratio=0.8):
    split_idx = int(len(df) * train_ratio)

    X = df[["periode_index"]]
    y = df[target_col]

    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
        split_idx,
    )


def scale_data(X_train, X_test, y_train, y_test):
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(
        y_train.values.reshape(-1, 1)
    ).ravel()

    y_test_scaled = y_scaler.transform(
        y_test.values.reshape(-1, 1)
    ).ravel()

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        x_scaler,
        y_scaler,
    )


def inverse_scale_y(y_scaled, y_scaler):
    return y_scaler.inverse_transform(
        np.array(y_scaled).reshape(-1, 1)
    ).ravel()


def plot_boxplot(df, target_col="total_pendapatan", output_path="boxplot.png"):
    plt.figure(figsize=(10, 5))
    plt.boxplot(df[target_col])
    plt.title(f"Boxplot {target_col}")
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Boxplot disimpan ke {output_path}")


def run_preprocessing_report(df):
    check_missing_values(df)
    check_duplicates(df)
    descriptive_statistics(df)
    check_time_period(df)
    plot_boxplot(df)


def preprocess_data(engine):
    df = load_forecasting_data(engine)
    run_preprocessing_report(df)
    df = handle_missing_values(df)
    df = add_log_transform(df)

    return df