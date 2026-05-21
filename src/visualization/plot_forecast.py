import matplotlib.pyplot as plt


def plot_forecast(
    df,
    split_idx,
    y_pred_gpr,
    y_std_gpr,
    y_pred_sarima,
    y_pred_prophet,
):
    test_df = df.iloc[split_idx:].copy()

    plt.figure(figsize=(14, 7))

    # =========================
    # Actual Data
    # =========================
    plt.plot(
        df["periode_index"],
        df["total_pendapatan"],
        label="Actual Revenue",
        linewidth=2,
    )

    # =========================
    # GPR Forecast
    # =========================
    plt.plot(
        test_df["periode_index"],
        y_pred_gpr,
        label="GPR Forecast",
        linewidth=2,
    )

    # =========================
    # SARIMA Forecast
    # =========================
    plt.plot(
        test_df["periode_index"],
        y_pred_sarima,
        label="SARIMA Forecast",
        linewidth=2,
    )

    # =========================
    # Prophet Forecast
    # =========================
    plt.plot(
        test_df["periode_index"],
        y_pred_prophet,
        label="Prophet Forecast",
        linewidth=2,
    )

    # =========================
    # GPR Confidence Interval
    # =========================
    plt.fill_between(
        test_df["periode_index"],
        y_pred_gpr - 1.96 * y_std_gpr,
        y_pred_gpr + 1.96 * y_std_gpr,
        alpha=0.2,
        label="GPR 95% Confidence Interval",
    )

    # =========================
    # Styling
    # =========================
    plt.xlabel("Periode Bulanan")
    plt.ylabel("Total Pendapatan")
    plt.title("Forecasting Pendapatan Bulanan")

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # =========================
    # Save
    # =========================
    output_path = "outputs/forecast_result.png"

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"Grafik forecast disimpan ke {output_path}")