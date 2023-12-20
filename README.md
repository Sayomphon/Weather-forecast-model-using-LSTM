# Weather-forecast-model


## Import framework and library
``` python
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
```
This code is an implementation of a time series prediction model using the LSTM algorithm in TensorFlow and Keras. 

## Load data
``` python
data = pd.read_csv('data.csv')
```
The dataset used is a climate dataset with features such as temperature, wind speed and precipitation.

## Preprocessing
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
The data is divided into train and test sets and normalized before feeding to the model.

## Create data generator
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
```
A data generator is created for creating batches of sequence data for training and testing purposes. 

## Define model
``` python
model = keras.Sequential([
    keras.layers.Input(shape=(n_steps, len(train_data.columns) - 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(1)
])
```
The LSTM model is defined with one LSTM layer and one dense output layer.

## Compile model
``` python
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['mae'])
```
The model is compiled with mean squared error as the loss function and Adam optimizer with a learning rate of 0.01.

## Train model
``` python
model.fit(train_data_gen, epochs=100, batch_size=batch_size)
```
The model is then trained for 100 epochs using the data generator.

## Test model and print results
``` python
X_test = np.array([test_data.values[i:i+n_steps,:-1] for i in range(test_samples)])
y_true = test_data[n_steps:]['Data.Precipitation'].values.reshape(-1, 1)
y_pred = model.predict(X_test)

mse = tf.keras.losses.mean_squared_error(y_true, y_pred).numpy()
mae = tf.keras.losses.mean_absolute_error(y_true, y_pred).numpy()
print('MSE: ', mse)
print('MAE: ', mae)
```
The model is tested on the test data and the mean squared error and mean absolute error are calculated and printed. 

## Plot true vs predicted precipitation values
``` python
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample index')
plt.ylabel('Precipitation')
plt.legend()
plt.show()
```
A plot is also shown to visualize the predicted precipitation values compared to the true values.

!https://github.com/Sayomphon/Weather-forecast-model/blob/a7fd9b9251ecd4323aa5de55929eef448ba416e3/Prediction%20result.PNG
