import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def adf_test(series):
    result = adfuller(series)

    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary": result[1] < 0.05
    }


def kpss_test(series):
    result = kpss(series, regression="c", nlags="auto")

    return {
        "KPSS Statistic": result[0],
        "p-value": result[1],
        "Stationary": result[1] > 0.05
    }


def ljung_box_test(residuals, lags=12):
    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)

    return {
        "LB Statistic": result["lb_stat"].iloc[0],
        "p-value": result["lb_pvalue"].iloc[0],
        "White Noise": result["lb_pvalue"].iloc[0] > 0.05
    }


def plot_residual_acf_pacf(residuals, output_path="outputs/residual_acf_pacf.png"):
    residuals = np.asarray(residuals)
    max_lags = min(12, (len(residuals) // 2) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_acf(residuals, ax=axes[0], lags=max_lags)
    axes[0].set_title("ACF Residual")

    plot_pacf(residuals, ax=axes[1], lags=max_lags, method="ywm")
    axes[1].set_title("PACF Residual")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot ACF/PACF residual disimpan ke {output_path}")


def coverage_probability(y_true, y_pred, y_std):
    lower = y_pred - 1.96 * y_std
    upper = y_pred + 1.96 * y_std

    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100

    return {
        "Coverage Probability 95%": coverage
    }