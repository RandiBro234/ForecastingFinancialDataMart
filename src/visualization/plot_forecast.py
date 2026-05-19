# plot forecast
import matplotlib.pyplot as plt

def plot_forecast(df, split_idx, y_pred_gpr, y_std_gpr, y_pred_sarima):
    test_df = df.iloc[split_idx:].copy()

    plt.figure(figsize=(12, 6))

    plt.plot(df["periode_index"], df["total_pendapatan"], label="Actual")
    plt.plot(test_df["periode_index"], y_pred_gpr, label="GPR Forecast")
    plt.plot(test_df["periode_index"], y_pred_sarima, label="SARIMA Forecast")

    plt.fill_between(
        test_df["periode_index"],
        y_pred_gpr - 1.96 * y_std_gpr,
        y_pred_gpr + 1.96 * y_std_gpr,
        alpha=0.2,
        label="GPR 95% Confidence Interval"
    )

    plt.xlabel("Periode Kuartal")
    plt.ylabel("Total Pendapatan")
    plt.title("Forecasting Pendapatan Kuartalan")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_result.png", dpi=300)
    plt.close()
    print("Grafik forecast disimpan ke forecast_result.png")