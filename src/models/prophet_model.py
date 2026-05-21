from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def prepare_prophet_data(df):
    prophet_df = pd.DataFrame()

    prophet_df["ds"] = df["periode_date"]
    prophet_df["y"] = df["total_pendapatan"]

    return prophet_df


def train_prophet(df_train):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive"
    )

    model.fit(df_train)

    return model


def predict_prophet(model, periods):
    future = model.make_future_dataframe(
        periods=periods,
        freq="MS"
    )

    forecast = model.predict(future)

    return forecast

import matplotlib.pyplot as plt


def save_prophet_components(model, forecast, output_path="outputs/prophet_components.png"):
    fig = model.plot_components(forecast)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Prophet components disimpan ke {output_path}")
