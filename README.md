# Weather forecast model using LSTM model
This code is a Python script using the TensorFlow and Pandas libraries to build, train, and evaluate a simple Long Short-Term Memory (LSTM) neural network for predicting precipitation based on temperature, wind speed, and month data.
## Import libraries
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- tensorflow: An open-source machine learning framework.
- matplotlib.pyplot: For data visualization.
``` python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
``` 
## Load data
- The script loads data from a CSV file named ‘data.csv’ using Pandas.
``` python
data = pd.read_csv('data.csv')
```
## Preprocessing
- Extracts relevant columns for training and testing data, filtering based on the year.
- Computes the mean and standard deviation of the training data and standardizes both training and testing data.
- Defines n_steps (sequence length), batch_size, and calculates the number of samples for training and testing.
``` pytohn
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
```
## Create data generator
- Uses TensorFlow’s timeseries_dataset_from_array to create sequences of input features and target labels for training and testing.
``` python
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
```
## Define model
- A simple sequential model is defined using Keras with an input layer, an LSTM layer with 64 units, and a dense output layer.
``` python
model = keras.Sequential([
    keras.layers.Input(shape=(n_steps, len(train_data.columns) - 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(1)
])
```
## Compile model
- Compiles the model using mean squared error (MSE) as the loss function, Adam optimizer with a specified learning rate, and Mean Absolute Error (MAE) as a metric.
``` python
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['mae'])
```
## Train model
- Trains the model using the training data generator, specifying the number of epochs and batch size.
``` python
model.fit(train_data_gen, epochs=100, batch_size=batch_size)
```
## Test model and print results
- Prepares the test data for prediction and uses the trained model to predict precipitation values.
- Calculates and prints Mean Squared Error (MSE) and Mean Absolute Error (MAE) between the true and predicted precipitation values.
``` python
X_test = np.array([test_data.values[i:i+n_steps,:-1] for i in range(test_samples)])
y_true = test_data[n_steps:]['Data.Precipitation'].values.reshape(-1, 1)
y_pred = model.predict(X_test)

mse = tf.keras.losses.mean_squared_error(y_true, y_pred).numpy()
mae = tf.keras.losses.mean_absolute_error(y_true, y_pred).numpy()
print('MSE: ', mse)
print('MAE: ', mae)
```
## Plot true vs predicted precipitation values
Plots the true vs predicted precipitation values using Matplotlib.
``` python
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample index')
plt.ylabel('Precipitation')
plt.legend()
plt.show()
```
## The result
![Result](https://github.com/Sayomphon/Weather-forecast-model/blob/main/Prediction%20result.PNG)
## Conclusion
This script essentially demonstrates a basic time series forecasting model using an LSTM neural network for predicting precipitation based on historical weather data.

