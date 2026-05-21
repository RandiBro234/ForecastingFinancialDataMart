import numpy as np

from src.config.db import engine

from src.preprocessing.preprocess import (
    load_forecasting_data,
    run_preprocessing_report,
    handle_missing_values,
    train_test_split_time_series,
)

from src.models.gpr_model import train_gpr, predict_gpr
from src.models.sarima_model import train_sarima, predict_sarima
from src.models.prophet_model import (
    prepare_prophet_data,
    train_prophet,
    predict_prophet,
    save_prophet_components,
)

from src.evaluation.metrics import calculate_metrics

from src.evaluation.diagnostics import (
    adf_test,
    kpss_test,
    ljung_box_test,
    plot_residual_acf_pacf,
    coverage_probability,
)

from src.visualization.plot_forecast import plot_forecast


# =========================
# 1. Load Data
# =========================
df = load_forecasting_data(engine)


# =========================
# 2. Preprocessing Report
# =========================
run_preprocessing_report(df)


# =========================
# 3. Handle Missing Values
# =========================
df = handle_missing_values(df)

print("\nData siap modeling:")
print(df.head())
print("Jumlah periode:", len(df))


# =========================
# 4. Stationarity Test
# =========================
print("\n=== ADF Test Original Series ===")
adf_result = adf_test(df["total_pendapatan"])
print(adf_result)

print("\n=== KPSS Test Original Series ===")
kpss_result = kpss_test(df["total_pendapatan"])
print(kpss_result)


# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test, split_idx = train_test_split_time_series(
    df,
    target_col="total_pendapatan",
    train_ratio=0.8,
)


# =========================
# 6. GPR Modeling
# =========================
gpr_model = train_gpr(X_train, y_train)
y_pred_gpr, y_std_gpr = predict_gpr(gpr_model, X_test)


# =========================
# 7. SARIMA Modeling
# =========================
sarima_model = train_sarima(y_train)
y_pred_sarima = predict_sarima(sarima_model, len(y_test))
y_pred_sarima = np.array(y_pred_sarima)


# =========================
# 8. Prophet Modeling
# =========================
prophet_df = prepare_prophet_data(df)

split_idx_prophet = int(len(prophet_df) * 0.8)

train_prophet_df = prophet_df.iloc[:split_idx_prophet]
test_prophet_df = prophet_df.iloc[split_idx_prophet:]

prophet_model = train_prophet(train_prophet_df)

forecast_prophet = predict_prophet(
    prophet_model,
    periods=len(test_prophet_df),
)

forecast_prophet_test = forecast_prophet.iloc[-len(test_prophet_df):]
y_pred_prophet = forecast_prophet_test["yhat"].values

save_prophet_components(
    prophet_model,
    forecast_prophet,
)


# =========================
# 9. Model Evaluation
# =========================
gpr_metrics = calculate_metrics(y_test, y_pred_gpr)
sarima_metrics = calculate_metrics(y_test, y_pred_sarima)
prophet_metrics = calculate_metrics(y_test, y_pred_prophet)

print("\n=== Evaluasi GPR ===")
print(gpr_metrics)

print("\n=== Evaluasi SARIMA ===")
print(sarima_metrics)

print("\n=== Evaluasi Prophet ===")
print(prophet_metrics)


# =========================
# 10. Residual Diagnostics
# =========================
residual_gpr = y_test.values - y_pred_gpr
residual_sarima = y_test.values - y_pred_sarima
residual_prophet = y_test.values - y_pred_prophet

print("\n=== Ljung-Box Test GPR Residual ===")
ljung_gpr = ljung_box_test(residual_gpr, lags=12)
print(ljung_gpr)

print("\n=== Ljung-Box Test SARIMA Residual ===")
ljung_sarima = ljung_box_test(residual_sarima, lags=12)
print(ljung_sarima)

print("\n=== Ljung-Box Test Prophet Residual ===")
ljung_prophet = ljung_box_test(residual_prophet, lags=12)
print(ljung_prophet)


# =========================
# 11. ACF/PACF Residual Plot
# =========================
plot_residual_acf_pacf(
    residual_sarima,
    output_path="outputs/residual_acf_pacf_sarima.png",
)

plot_residual_acf_pacf(
    residual_gpr,
    output_path="outputs/residual_acf_pacf_gpr.png",
)

plot_residual_acf_pacf(
    residual_prophet,
    output_path="outputs/residual_acf_pacf_prophet.png",
)


# =========================
# 12. GPR Coverage Probability
# =========================
coverage_result = coverage_probability(
    y_test.values,
    y_pred_gpr,
    y_std_gpr,
)

print("\n=== GPR Coverage Probability 95% ===")
print(coverage_result)


# =========================
# 13. Forecast Visualization
# =========================
plot_forecast(
    df,
    split_idx,
    y_pred_gpr,
    y_std_gpr,
    y_pred_sarima,
    y_pred_prophet,
)


# =========================
# 14. Summary
# =========================
print("\n=== Ringkasan Akhir ===")
print(f"GPR MAPE     : {gpr_metrics['MAPE']:.2f}%")
print(f"SARIMA MAPE  : {sarima_metrics['MAPE']:.2f}%")
print(f"Prophet MAPE : {prophet_metrics['MAPE']:.2f}%")

best_model = min(
    {
        "GPR": gpr_metrics["MAPE"],
        "SARIMA": sarima_metrics["MAPE"],
        "Prophet": prophet_metrics["MAPE"],
    },
    key={
        "GPR": gpr_metrics["MAPE"],
        "SARIMA": sarima_metrics["MAPE"],
        "Prophet": prophet_metrics["MAPE"],
    }.get,
)

print(f"Model terbaik berdasarkan MAPE: {best_model}")