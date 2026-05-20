# plot forecast
import matplotlib.pyplot as plt


def plot_forecast(df, split_idx, y_pred_gpr, y_std_gpr, y_pred_sarima):

    test_df = df.iloc[split_idx:].copy()

    plt.figure(figsize=(14, 7))

    # Actual Data
    plt.plot(
        df["periode_index"],
        df["total_pendapatan"],
        label="Actual Revenue",
        linewidth=2.5,
    )

    # GPR Forecast
    plt.plot(
        test_df["periode_index"],
        y_pred_gpr,
        label="GPR Forecast",
        linewidth=2.5,
    )

    # SARIMA Forecast
    plt.plot(
        test_df["periode_index"],
        y_pred_sarima,
        label="SARIMA Forecast",
        linewidth=2.5,
    )

    # Confidence Interval GPR
    plt.fill_between(
        test_df["periode_index"],
        y_pred_gpr - 1.96 * y_std_gpr,
        y_pred_gpr + 1.96 * y_std_gpr,
        alpha=0.2,
        label="GPR 95% Confidence Interval",
    )

    # Axis & Title
    plt.xlabel("Monthly Period Index")
    plt.ylabel("Total Revenue")
    plt.title("Monthly Financial Revenue Forecasting")

    # Grid
    plt.grid(alpha=0.3)

    # Legend
    plt.legend()

    # Layout
    plt.tight_layout()

    # Save Figure
    plt.savefig("outputs/forecast_result.png", dpi=300)

    plt.close()

    print("Grafik forecast disimpan ke outputs/forecast_result.png")