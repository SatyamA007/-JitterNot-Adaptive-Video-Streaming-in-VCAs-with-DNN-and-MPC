from os import listdir

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import gc
import sys
from sklearn import preprocessing

print(f"Tensorflow Version: {tf.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"System Version: {sys.version}")

mpl.rcParams['figure.figsize'] = (17, 5)
mpl.rcParams['axes.grid'] = False
sns.set_style("whitegrid")

pd.set_option('display.max_columns', None)




# Data Loader Parameters
BATCH_SIZE = 25
BUFFER_SIZE = 10000


# LSTM Parameters
# EVALUATION_INTERVAL = 200
# EPOCHS = 4
EVALUATION_INTERVAL = 50
EPOCHS = 1
PATIENCE = 5

past_history = 10
future_target = 3
STEP = 1

# dataset path
csv_dir = 'gdrive/SPRING22/ML_FOR_NETWORKS/dataset/Features'
# csv_dir = 'Features'


ct = 1


temp_shape = (8, 13)

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def create_time_steps(length):
    return list(range(-length, 0))

def plot_train_history(history, title):
    # return
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

def multi_step_plot(history, true_future, prediction):
    # return
    plt.figure(figsize=(18, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=temp_shape))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(future_target))
#
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
# print(multi_step_model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience = PATIENCE, restore_best_weights=True)
val_data_multi_global = []

def train_util_individual_file(csv_path):
    df = pd.read_csv(csv_path)
    # print("DataFrame Shape: {} rows, {} columns".format(*df.shape))
    TRAIN_SPLIT = int(0.8 * df.shape[0])
    # print(TRAIN_SPLIT)
    df = df.assign(label=lambda x: x['outbound_packetsSent/s'])
    df.drop('outbound_packetsSent/s', axis=1, inplace=True)

    dataset = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset = min_max_scaler.fit_transform(dataset)

    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                     TRAIN_SPLIT, past_history,
                                                     future_target, STEP)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                                 TRAIN_SPLIT, None, past_history,
                                                 future_target, STEP)
    print(x_train_multi, y_train_multi, x_val_multi, y_val_multi)

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    #
    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
    #

    multi_step_history = multi_step_model.fit(train_data_multi,
                                              epochs=EPOCHS,
                                              steps_per_epoch=EVALUATION_INTERVAL,
                                              validation_data=val_data_multi,
                                              validation_steps=EVALUATION_INTERVAL,
                                              callbacks=[early_stopping])
    # plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
    val_data_multi_global.append(val_data_multi)
    # for x, y in val_data_multi.take(3):
        # print(x[0].shape, y[0].shape)
        # print(multi_step_model.predict(x)[0].shape)
        # multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])

    # del val_data_multi, train_data_multi
    del train_data_multi
    _ = gc.collect()



for dir in os.listdir(csv_dir):
    if os.path.isdir(os.path.join(csv_dir, dir)):
        for csv_file in os.listdir(os.path.join(csv_dir, dir)):
            print("*****************************************************")
            print(csv_file)
            if 'DS_Store' in csv_file:
                continue
            ct -= 1
            csv_path = os.path.join(csv_dir, dir, csv_file)
            train_util_individual_file(csv_path)
            if ct < 0:
                break

predicted = []
expected = []
for val_data_multi in val_data_multi_global:
    for x, y in val_data_multi.take(2):
        # print(x[0].shape, y[0].shape)
        # print(multi_step_model.predict(x)[0].shape)
        prediction = multi_step_model.predict(x)[0]
        multi_step_plot(x[0], y[0], prediction)
        expected.extend(y[0].numpy().tolist())
        predicted.extend(prediction.tolist())



from sklearn.metrics import mean_squared_error
from math import sqrt
mse = mean_squared_error(expected, predicted)
rmse = sqrt(mse)
print('RMSE: %f' % rmse)

