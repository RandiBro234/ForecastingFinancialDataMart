# SARIMA model
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(y_train):
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)
    return result

def predict_sarima(model, steps):
    forecast = model.forecast(steps=steps)
    return forecast