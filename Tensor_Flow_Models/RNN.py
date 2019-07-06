import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

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


[processed_features, processed_targets] = preprocess_features(raw_data)
processed_features_copy = processed_features.copy()

processed_features['APXPrice'] = clip(processed_features['APXPrice'], 0, 200)
processed_features['Biomass'] = clip(processed_features['Biomass'], 0, 4000)
processed_features['Nuclear'] = clip(processed_features['Nuclear'], 4000, 10000)
processed_features['OffWind'] = clip(processed_features['OffWind'], 0, 5000)
processed_features['OffWind'] = clip(processed_features['OffWind'], 0, 11000)
processed_features['ImbalancePrice'] = clip(processed_features['ImbalancePrice'], -100, 250)

processed_features['FossilHardCoal'] = log_normalize(processed_features['FossilHardCoal'])
processed_features['HydroPumpedStorage'] = log_normalize(processed_features['HydroPumpedStorage'])
processed_features['HydroRunOfRiver'] = log_normalize(processed_features['HydroRunOfRiver'])
processed_features['solar'] = log_normalize(processed_features['solar'])
processed_features['Other'] = log_normalize(processed_features['Other'])

# ----------------------------------------------------------------------------------------------------------------------
# calculate the correlation matrix, isolate the NIV correlations and then order by the abs value (descending)

processed_test_features = processed_features.loc[processed_features.index > 2018100000, :]
processed_test_targets = processed_targets.loc[processed_targets.index > 2018100000]

processed_features = processed_features.loc[processed_features.index > 2017050000, :]
processed_targets = processed_targets.loc[processed_targets.index > 2017050000]

X_train_all = processed_features.loc[processed_features.index < 2018050000, :]
y_train = processed_targets.loc[processed_targets.index < 2018050000]

X_valid_all = processed_features.loc[processed_features.index > 2018050000, :]
y_valid = processed_targets.loc[processed_targets.index > 2018050000]
X_valid_all = X_valid_all.loc[X_valid_all.index < 2018100000, :]
y_valid = y_valid.loc[y_valid.index < 2018100000]

# Normalize the validation data and separate into X and y variables data frames.
#cols_all = ['ImbalancePrice', 'solar', 'Solar_Frac', 'APXPrice','Biomass', 'Other', 'wind_off', 'initialWindForecast',
#            'Wind_Frac', 'Val_Diff', ]

cols_all = ['Biomass', 'FossilGas', 'FossilHardCoal', 'HydroPumpedStorage',
       'HydroRunOfRiver', 'Nuclear', 'OffWind', 'OnWind', 'Other', 'Solar',
       'settlementPeriod', 'APXPrice', 'APXVolume', 'solar', 'wind_off',
       'wind_on', 'TSDF', 'Generation', 'buyPriceAdjustment', 'Shift_NIV',
       'reserveScarcityPrice', 'ITSDO', 'intewGeneration',
       'intfrGeneration', 'intirlGeneration', 'intnedGeneration',
       'drm2HourForecast', 'lolp1HourForecast', 'N2EXPrice', 'N2EXVolume',
       'initialWindForecast', 'latestWindForecast', 'windOutturn',
       'RenewablePrediction', 'Val_Diff', 'Solar_Frac', 'Wind_Frac',
       'Renewable_Frac', 'NIV_shift_1hr', 'NIV_shift_4hr']
X_train = X_train_all.loc[:, cols_all]
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = ((X_train - X_train_mean) / X_train_std)

X_valid = X_valid_all.loc[:, cols_all]
X_valid = ((X_valid-X_train_mean)/X_train_std)

X_test = processed_test_features.loc[:, cols_all]
X_test = ((X_test-X_train_mean)/X_train_std)

y_test = processed_test_targets

train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid.values, y_valid.values))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))


# ----------------------------------------------------------------------------------------------------------------------
# build MLP model
root_logdir = os.path.join(os.curdir, "my_logs")
root_modeldir = os.path.join(os.curdir, "Tensor_Flow_Models/models")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def get_run_modeldir():
    import time
    run_id = time.strftime("MLP_run_%Y_%m_%d-%H_%M_%S.h5")
    return os.path.join(root_modeldir, run_id)


run_logdir = get_run_logdir()
model_dir = get_run_modeldir()
"""

def build_model(n_hidden=3, n_neurons=128, learning_rate=0.001, input_shape=(40,), l2_reg=0.01):
    model = keras.models.Sequential()
    inpt_options = {"input_shape": input_shape}
    options = {"kernel_regularizer": keras.regularizers.l2(l2_reg)}

    for layer in range(n_hidden):
        model.add(keras.layers.LSTM(n_neurons, activation="relu", kernel_initializer="he_normal", return_sequences=True,
                                    **inpt_options, **options))
        model.add(keras.layers.Dropout(0.2))
        inpt_options = {}

    model.add(keras.layers.LSTM(n_neurons, activation="relu", kernel_initializer="he_normal", **inpt_options, **options))
    model.add(keras.layers.Dense(1, **inpt_options))
    optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_dir, save_best_only=True)
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [3],
    "n_neurons": [118],
    "learning_rate": [0.0001],
    "l2_reg": [0.00001],
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=1, cv=2)
rnd_search_cv.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid),
                  callbacks=[earlystopping, tensorboard_cb, checkpoint_cb])

print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)
model = rnd_search_cv.best_estimator_.model
history = model.history

"""""

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_dir, save_best_only=True)
earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
optimizer = keras.optimizers.SGD

model = keras.models.Sequential()
model.add(keras.layers.simpleRNN(128, activation="relu", kernel_initializer="he_normal", input_shape=[None, 40]))
model.compile(loss="mean_squared_error", optimizer=optimizer)

history = model.fit(X_train, y_train,
                    epochs=200,
                    validation_data=(X_valid, y_valid),
                    validation_steps=3,
                    callbacks=[earlystopping, tensorboard_cb, checkpoint_cb],
                    batch_size=32
                    )


mse_test = model.evaluate(X_test, y_test)


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

model_trained = model
# keras.models.load_model("Tensor_Flow_Models/models/MLP_run_2019_06_10-11_12_00.h5")
y_MLP_prediction = model_trained.predict(X_test)
MLP_mse = mean_squared_error(y_test, y_MLP_prediction)
MLP_rmse = np.sqrt(MLP_mse)

# convert periods to days
days = np.arange(len(y_MLP_prediction))/48
max_days = 10*48

# Plot data on one graph.
plt.rcParams["figure.figsize"] = (30, 10)
plt.plot(days[:max_days], y_test.values[:max_days], color='k', linewidth=2, linestyle='dashed')
plt.plot(days[:max_days], y_MLP_prediction[:max_days], color='b')
plt.xlabel('Days')
plt.ylabel('NIV')
plt.title('MLP Model')
plt.show()
