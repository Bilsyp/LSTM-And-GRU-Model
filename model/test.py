import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from Excel file"""
    return pd.read_excel(file_path)


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess data"""
    bandwidth = np.array([df["Bandwidth (Mbps)"]]).reshape(-1, 1)
    bitrate = np.array([df["Bitrate (Mbps)"]]).reshape(-1, 1)
    return bandwidth, bitrate


def split_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple:
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Scale data using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def build_model(input_shape: tuple) -> Sequential:
    """Build FCN model"""
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    return model


def compile_model(model: Sequential) -> None:
    """Compile model"""
    model.compile(optimizer=Adam(learning_rate=0.01), loss="mean_squared_error")


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
) -> None:
    """Train model"""
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Evaluate model"""
    loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return loss, mse, mae, r2


def make_prediction(model: Sequential, data: np.ndarray) -> np.ndarray:
    """Make prediction on new data"""
    return model.predict(data)


if __name__ == "__main__":
    # Load data
    file_path = "../data/bandwidth_video_quality_with_bitrate_V2.xlsx"
    df = load_data(file_path)
    bandwidth, bitrate = preprocess_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(bandwidth, bitrate)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Build and compile model
    model = build_model((X_train_scaled.shape[1],))
    compile_model(model)

    # Train model
    train_model(model, X_train_scaled, y_train)

    # Evaluate model
    loss, mse, mae, r2 = evaluate_model(model, X_test_scaled, y_test)
    print(f"Loss: {loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Make prediction on new data
    data_new = np.array([23.34]).reshape(-1, 1)

    data_new_scaled = scale_data(data_new, data_new)[0]
    prediction = make_prediction(model, data_new_scaled)

    print(f"Prediction: {prediction}")
