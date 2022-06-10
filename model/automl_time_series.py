# -*- coding: utf-8 -*-

# FEDOT api
from fedot.api.main import Fedot
# Tasks to solve
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
# Input data for fit and predict
from fedot.core.data.data import InputData
# Train and test split
from fedot.core.data.data import train_test_data_setup

forecast_length = 5
task = Task(TaskTypesEnum.ts_forecasting,
            TsForecastingParams(forecast_length=forecast_length))


drive = 'content/drive/features'
# Init model for the time series forecasting
model = Fedot(problem='ts_forecasting', task_params=task.task_params)
# outbound_packetsSent/s

# Imports for creating plots
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 18, 7


def plot_results(actual_time_series, predicted_values, len_train_data, y_name='Throughput'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black', linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15, loc='upper left')
    plt.grid()
    plt.show()


def smape_loss(y_true, y_pred):
    import tensorflow.keras.backend as K
    epsilon = 0.1
    summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
    smape = K.abs(y_pred - y_true) / summ * 2.0
    print('SMAPE: %f' % smape)
    return smape


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

label = 'outbound_packetsSent/s'
forecast = None
label_data_list = []
forecast_list = []
test_data_list = []
mean_error = []


def train_and_test_model(csvpath):
    # global forecast, train_data, test_data, label_data
    global chain
    df = pd.read_csv(csvpath)
    label_data = df[label]
    input_data = InputData.from_csv_time_series(task, csvpath, target_column=label)
    train_data, test_data = train_test_data_setup(input_data)
    # Init model for the time series forecasting
    model = Fedot(problem='ts_forecasting', task_params=task.task_params)
    # outbound_packetsSent/s
    chain = model.fit(features=train_data)
    output = model.predict(features=test_data)
    forecast = np.ravel(output)
    label_data_list.append(label_data)
    forecast_list.append(forecast)
    test_data_list.append(test_data)
    plot_results(actual_time_series=label_data,
                 predicted_values=forecast,
                 len_train_data=len(label_data) - forecast_length)
    print(f'Mean absolute error: {mean_absolute_error(test_data.target, forecast):.3f}')
    print(f'Mean squared error: {mean_squared_error(test_data.target, forecast):.3f}')
    print(f'SMAPE error: {smape_loss(test_data.target, forecast):.3f}')

    mean_error.append(mean_squared_error(test_data.target, forecast, squared=False))
    del model


import os

csv_dir = 'gdrive/MyDrive/UCSB/SPRING22/ML_FOR_NETWORKS/dataset/Features'
csv_file = csv_dir + ''

ct = 30
for dir in os.listdir(csv_dir):
    if os.path.isdir(os.path.join(csv_dir, dir)):
        for csv_file in os.listdir(os.path.join(csv_dir, dir)):
            print("*****************************************************")
            print(csv_file)
            if 'DS_Store' in csv_file:
                continue
            ct -= 1
            csv_path = os.path.join(csv_dir, dir, csv_file)
            train_and_test_model(csv_path)
            if ct < 0:
                break

for label_data, test_data in zip(label_data_list, test_data_list):
  output = model.predict(features=test_data)
  forecast = np.ravel(output)
  plot_results(actual_time_series = label_data,
             predicted_values = forecast,
             len_train_data = len(label_data)-forecast_length)