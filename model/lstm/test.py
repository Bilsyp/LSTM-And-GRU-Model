import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Convert data to pandas dataframe
data = {
    "timestamp": [
        "2010-01-01 00:00:00",
        "2010-01-02 00:00:00",
        "2010-01-03 00:00:00",
        "2010-01-04 00:00:00",
        "2010-01-05 00:00:00",
    ],
    "value": [10, 20, 15, 30, 25],
}
df = pd.DataFrame(data)

# Convert timestamp column to datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Set timestamp column as index
df.set_index("timestamp", inplace=True)

# Resample data to hourly frequency
df_resampled = df.resample("H").mean()

# Scale data using Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_resampled)

# Split data into training and testing sets
train_size = int(0.8 * len(df_scaled))
train_data, test_data = df_scaled[0:train_size], df_scaled[train_size : len(df_scaled)]


# Split data into X and y
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        X.append(a)
        y.append(dataset[(i + look_back), 0])
    return np.array(X), np.array(y)


look_back = 1
train_X, train_y = create_dataset(train_data, look_back)
test_X, test_y = create_dataset(test_data, look_back)

# Reshape data for LSTM
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Create LSTM model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2)
