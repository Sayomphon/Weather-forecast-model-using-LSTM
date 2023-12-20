import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load data
data = pd.read_csv('data.csv')

# Preprocessing
train_data = data[(data['Date.Year'] >= 2016) & (data['Date.Year'] <= 2017)][['Date.Month', 'Data.Temperature.Max Temp', 'Data.Wind.Speed', 'Data.Precipitation']]
test_data = data[(data['Date.Year'] == 2017)][['Date.Month', 'Data.Temperature.Max Temp', 'Data.Wind.Speed', 'Data.Precipitation']]
mean = train_data.mean()
std = train_data.std()
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
n_steps = 7
batch_size = 64
train_samples = len(train_data) - n_steps
test_samples = len(test_data) - n_steps

# Create data generator
train_data_gen = keras.preprocessing.timeseries_dataset_from_array(
    train_data[['Date.Month', 'Data.Temperature.Max Temp', 'Data.Wind.Speed']].values,
    train_data['Data.Precipitation'].values,
    sequence_length=n_steps,
    batch_size=batch_size
)
test_data_gen = keras.preprocessing.timeseries_dataset_from_array(
    test_data[['Date.Month', 'Data.Temperature.Max Temp', 'Data.Wind.Speed']].values,
    test_data['Data.Precipitation'].values,
    sequence_length=n_steps,
    batch_size=batch_size
)

# Define model
model = keras.Sequential([
    keras.layers.Input(shape=(n_steps, len(train_data.columns) - 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(1)
])

# Compile model
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['mae'])

# Train model
model.fit(train_data_gen, epochs=100, batch_size=batch_size)

# Test model
X_test = np.array([test_data.values[i:i+n_steps,:-1] for i in range(test_samples)])
y_true = test_data[n_steps:]['Data.Precipitation'].values.reshape(-1, 1)
y_pred = model.predict(X_test)

# Print results
mse = tf.keras.losses.mean_squared_error(y_true, y_pred).numpy()
mae = tf.keras.losses.mean_absolute_error(y_true, y_pred).numpy()
print('MSE: ', mse)
print('MAE: ', mae)

# Plot true vs predicted precipitation values
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample index')
plt.ylabel('Precipitation')
plt.legend()
plt.show()