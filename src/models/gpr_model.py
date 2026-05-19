# GPR model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def train_gpr(X_train, y_train):
    kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

def predict_gpr(model, X_test):
    y_pred, y_std = model.predict(X_test, return_std=True)
    return y_pred, y_std