from src.config.db import engine
from src.preprocessing.preprocess import (
    load_forecasting_data,
    run_preprocessing_report,
    handle_missing_values,
    train_test_split_time_series,
)
from src.models.gpr_model import train_gpr, predict_gpr
from src.models.sarima_model import train_sarima, predict_sarima
from src.evaluation.metrics import calculate_metrics
from src.visualization.plot_forecast import plot_forecast


# 1. Load data
df = load_forecasting_data(engine)

# 2. Preprocessing report
run_preprocessing_report(df)

# 3. Handle missing value jika ada
df = handle_missing_values(df)

print("\nData siap modeling:")
print(df.head())
print("Jumlah periode:", len(df))

# 4. Split data asli
X_train, X_test, y_train, y_test, split_idx = train_test_split_time_series(
    df,
    target_col="total_pendapatan",
    train_ratio=0.8,
)

# 5. GPR raw
gpr_model = train_gpr(X_train, y_train)
y_pred_gpr, y_std_gpr = predict_gpr(gpr_model, X_test)

# 6. SARIMA raw
sarima_model = train_sarima(y_train)
y_pred_sarima = predict_sarima(sarima_model, len(y_test))

# 7. Evaluasi
gpr_metrics = calculate_metrics(y_test, y_pred_gpr)
sarima_metrics = calculate_metrics(y_test, y_pred_sarima)

print("\n=== Evaluasi GPR ===")
print(gpr_metrics)

print("\n=== Evaluasi SARIMA ===")
print(sarima_metrics)

# 8. Visualisasi
plot_forecast(df, split_idx, y_pred_gpr, y_std_gpr, y_pred_sarima)