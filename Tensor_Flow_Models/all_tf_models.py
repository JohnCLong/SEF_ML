import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

np.random.seed(42)
tf.random.set_seed(42)

# import data form csv files
generation_per_type = pd.read_csv('SEF-ML/data/actual_aggregated_generation_per_type.csv')
apx_price = pd.read_csv('SEF-ML/data/apx_day_ahead.csv')
generation_day_ahead = pd.read_csv('SEF-ML/data/day_ahead_aggregated_generation.csv')
derived_system_wide_data = pd.read_csv('SEF-ML/data/derived_system_wide_data.csv')
forecast_day_and_day_ahead_demand_data = pd.read_csv('SEF-ML/data/forecast_day_and_day_ahead_demand_data.csv')
initial_demand_outturn = pd.read_csv('SEF-ML/data/initial_demand_outturn.csv')
interconnectors = pd.read_csv('SEF-ML/data/interconnectors.csv')
loss_of_load_probability = pd.read_csv('SEF-ML/data/loss_of_load_probability.csv')
market_index_data = pd.read_csv('SEF-ML/data/market_index_data.csv')
wind_generation_forecast_and_outturn = pd.read_csv('SEF-ML/data/wind_generation_forecast_and_outturn.csv')
renewable_generation_forecast = pd.read_csv('SEF-ML/data/day_ahead_generation_forecast_wind_and_solar.csv')
forecast_demand = pd.read_csv('SEF-ML/data/forecast_day_and_day_ahead_demand_data.csv')

# set the index of each data frame to begin at 0
generation_per_type.set_index('Unnamed: 0', inplace=True)
apx_price.set_index('Unnamed: 0', inplace=True)
generation_day_ahead.set_index('Unnamed: 0', inplace=True)
derived_system_wide_data.set_index('Unnamed: 0', inplace=True)
forecast_day_and_day_ahead_demand_data.set_index('Unnamed: 0', inplace=True)
initial_demand_outturn.set_index('Unnamed: 0', inplace=True)
interconnectors.set_index('Unnamed: 0', inplace=True)
loss_of_load_probability.set_index('Unnamed: 0', inplace=True)
market_index_data.set_index('Unnamed: 0', inplace=True)
wind_generation_forecast_and_outturn.set_index('Unnamed: 0', inplace=True)
renewable_generation_forecast.set_index('Unnamed: 0', inplace=True)
forecast_demand.set_index('Unnamed: 0', inplace=True)

# sort each data from in order
generation_per_type.sort_index(inplace=True)
apx_price.sort_index(inplace=True)
renewable_generation_forecast.sort_index(inplace=True)
forecast_demand.sort_index(inplace=True)
generation_day_ahead.sort_index(inplace=True)
derived_system_wide_data.sort_index(inplace=True)
forecast_day_and_day_ahead_demand_data.sort_index(inplace=True)
initial_demand_outturn.sort_index(inplace=True)
interconnectors.sort_index(inplace=True)
loss_of_load_probability.sort_index(inplace=True)
market_index_data.sort_index(inplace=True)
wind_generation_forecast_and_outturn.sort_index(inplace=True)

# ---------------------------------------------------------------------
# Data Pre-processing
# function to sum columns
raw_data = pd.concat([generation_per_type, apx_price, renewable_generation_forecast, forecast_demand,
                      generation_day_ahead, derived_system_wide_data, forecast_day_and_day_ahead_demand_data,
                      initial_demand_outturn, interconnectors, loss_of_load_probability,
                      market_index_data, wind_generation_forecast_and_outturn], axis=1, sort=True)


def preprocess_features(raw_data):
    """Prepares input features from California housing data set.

    Args:
    whole_dataframe: A Pandas DataFrame expected to contain data
      from the the "df".
    Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
    """

    # Create a copy of the raw data and then drop all duplicates and label data.
    processed_features = raw_data.copy()
    processed_features = processed_features.loc[:, ~processed_features.columns.duplicated()]

    # Drop unwanted data or fill if applicable
    indexNames = processed_features[processed_features['quantity'] < 10000].index
    processed_features.drop(indexNames, inplace=True)
    processed_features.drop("intnemGeneration", axis=1, inplace=True)
    processed_features.drop('settlementDate', axis=1, inplace=True)
    processed_features.drop('systemSellPrice', axis=1, inplace=True)
    processed_features.drop('sellPriceAdjustment', axis=1, inplace=True)
    processed_features.drop('FossilOil', axis=1, inplace=True)
    processed_features['initialWindForecast'].fillna(method='ffill', inplace=True)
    processed_features['latestWindForecast'].fillna(method='ffill', inplace=True)
    processed_features['reserveScarcityPrice'].fillna(0, inplace=True)
    processed_features['drm2HourForecast'].fillna(processed_features['drm2HourForecast'].mean(), inplace=True)
    processed_features['lolp1HourForecast'].fillna(0, inplace=True)
    processed_features.dropna(inplace=True)



    # Separate targets and features.
    processed_target = processed_features['indicativeNetImbalanceVolume'].copy()

    # Create a synthetic features.
    processed_features.loc[:, 'RenewablePrediction'] = (processed_features.loc[:, 'solar'] +
                                                        processed_features.loc[:, 'wind_off'] +
                                                        renewable_generation_forecast.loc[:, 'wind_on'])
    processed_features['Val_Diff'] = processed_features['initialWindForecast'] \
                                     - processed_features['latestWindForecast']

    processed_features['Solar_Frac'] = processed_features['solar'] / processed_features['quantity']

    processed_features['Wind_Frac'] = (processed_features['wind_off'] + processed_features['wind_on'])\
                                      / processed_features['quantity']

    processed_features['Renewable_Frac'] = processed_features['RenewablePrediction'] / processed_features['quantity']
    processed_features.indicativeNetImbalanceVolume = processed_features.indicativeNetImbalanceVolume.shift(2)
    processed_features.indicativeNetImbalanceVolume = processed_features.indicativeNetImbalanceVolume.fillna(0)

    processed_features["NIV_shift_1hr"] = processed_target.shift(2).fillna(processed_target.mean())

    processed_features["NIV_shift_4hr"] = processed_target.shift(8).fillna(processed_target.mean())

    # Rename columns
    processed_target = processed_target.rename("NIV")
    processed_features.rename({'quantity': 'Generation', 'systemBuyPrice': 'ImbalancePrice',
                               'indicativeNetImbalanceVolume': 'Shift_NIV'}, axis='columns', inplace=True)

    return processed_features, processed_target


def log_normalize(series):
    return series.apply(lambda x: np.log(x+1.0))


def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x: (min(max(x, clip_to_min), clip_to_max)))

def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1250, 1000])


def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
#    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


[processed_features, processed_targets] = preprocess_features(raw_data)
# ----------------------------------------------------------------------------------------------------------------------
series = processed_targets[4:].values.reshape(929,55)
series = series[..., np.newaxis].astype(np.float32)
n_steps = 54
X_train, y_train = series[:729, :n_steps], series[:729, -1]
X_valid, y_valid = series[729:829, :n_steps], series[729:829, -1]
X_test, y_test = series[829:, :n_steps], series[829:, -1]

# ----------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
for col in range(3):
    plt.sca(axes[col])
    plot_series(X_valid[col, :, 0], y_valid[col, 0],
                y_label=("$x(t)$" if col == 0 else None))
plt.show()

y_pred = X_valid[:, -1]
naive_loss = np.sqrt(np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))

plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
root_logdir = os.path.join(".", "Tensor_Flow_Models/RNN_test/my_logs")
root_modeldir = os.path.join(".", "Tensor_Flow_Models/RNN_test/models")


def get_run_logdir(name):
    import time
    run_id = time.strftime(f"{name}_run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def get_run_modeldir(name):
    import time
    run_id = time.strftime(f"{name}_MLP_run_%Y_%m_%d-%H_%M_%S.h5")
    return os.path.join(root_modeldir, run_id)

n = 1

run_logdir = get_run_logdir(f"{n}_SLP")
model_dir = get_run_modeldir(f"{n}_SLP")
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_dir, save_best_only=True)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[54, 1]),
    keras.layers.Dense(n)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid),
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), tensorboard_cb,
                               checkpoint_cb])

FCN_loss = np.sqrt(model.evaluate(X_valid, y_valid))

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

FCN_y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], FCN_y_pred[0, 0])
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
n = 100

run_logdir = get_run_logdir(f"{n}_RNN")
model_dir = get_run_modeldir(f"{n}_RNN")
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir, histogram_freq=2)
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_dir, save_best_only=True)

RNN_model = keras.models.Sequential([
    keras.layers.SimpleRNN(n, input_shape=[None, 1], activation = "relu"),
    keras.layers.Dense(1)
])

RNN_model.compile(loss="mse", optimizer="adam")
RNN_history = RNN_model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid),
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), tensorboard_cb,
                                       checkpoint_cb])

RNN_loss = np.sqrt(RNN_model.evaluate(X_valid, y_valid))

plot_learning_curves(RNN_history.history["loss"], RNN_history.history["val_loss"])
plt.show()
RNN_y_pred = RNN_model.predict(X_valid)

plot_series(X_valid[0, :, 0], y_valid[0, 0], RNN_y_pred[0, 0])
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
layer_1 = 361
layer_2 = 261
layer_3 = 11
layer_4 = 291
layer_5 = 411

run_logdir = get_run_logdir(f"{layer_1}_{layer_2}_DRNN")
model_dir = get_run_modeldir(f"{layer_1}_{layer_2}_DRNN")
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_dir, save_best_only=True)
DRNN_model = keras.models.Sequential([
    keras.layers.SimpleRNN(layer_1, return_sequences=True, input_shape=[None, 1], activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.SimpleRNN(layer_2, return_sequences=True, input_shape=[None, 1], activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.SimpleRNN(layer_3, return_sequences=True, input_shape=[None, 1], activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.SimpleRNN(layer_4, return_sequences=True, input_shape=[None, 1], activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.SimpleRNN(layer_5, return_sequences=False, activation="relu"),
    keras.layers.Dense(1)
])

DRNN_model.compile(loss="mse", optimizer="adam")
DRNN_history = DRNN_model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid),
                              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                                         tensorboard_cb, checkpoint_cb])

DRNN_loss = np.sqrt(DRNN_model.evaluate(X_valid, y_valid))

plot_learning_curves(DRNN_history.history["loss"], DRNN_history.history["val_loss"])
plt.show()
DRNN_y_pred = DRNN_model.predict(X_valid)

plot_series(X_valid[0, :, 0], y_valid[0, 0], DRNN_y_pred[0, 0])
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
print(f"the naive loss is {naive_loss}")
print(f"the fully connected network loss is {FCN_loss}")
print(f"the RNN loss is {RNN_loss}")
print(f"the DRNN loss is {DRNN_loss}")
