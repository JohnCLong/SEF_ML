import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# from sklearn.externals import joblib


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

processed_test_features = processed_features.loc[processed_features.index < 2016500000, :]
processed_test_targets = processed_targets.loc[processed_targets.index < 2016500000]

processed_features = processed_features.loc[processed_features.index > 2016500000, :]
processed_targets = processed_targets.loc[processed_targets.index > 2016500000]

X_train_all = processed_features.loc[processed_features.index < 2018030000, :]
y_train = processed_targets.loc[processed_targets.index < 2018030000]

X_valid_all = processed_features.loc[processed_features.index > 2018030000, :]
y_valid = processed_targets.loc[processed_targets.index > 2018030000]

# Normalize the validation data and separate into X and y variables data frames.
cols_all = ['ImbalancePrice', 'solar', 'Solar_Frac', 'APXPrice','Biomass', 'Other', 'wind_off', 'initialWindForecast',
            'Wind_Frac', 'Val_Diff', ]

X_train = X_train_all.loc[:, cols_all]
X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std

X_valid = X_valid_all.loc[:, cols_all]
X_valid = (X_valid-X_train_mean)/X_train_std

X_test = processed_test_features.loc[:, cols_all]
X_test = (X_test-X_train_mean)/X_train_std

y_test = processed_test_targets

# ----------------------------------------------------------------------------------------------------------------------
# train each sklearn model
lass = LassoCV(cv=40)

t1 = time.time()
lass.fit(X_train, y_train)
t_lasso_cv = time.time() - t1

elipson = 1e-4

log_alphas = -np.log10(lass.alphas_ + elipson)

plt.figure()
plt.plot(log_alphas, lass.mse_path_, ':')
plt.plot(log_alphas, lass.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(lass.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent (train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
plt.show()

# calculate the predictions from each model.
y_lass_prediction = lass.predict(X_test)
lass_mse = mean_squared_error(y_test, y_lass_prediction)
lass_rme = np.sqrt(lass_mse)
plt.show()

print("Lasso model RME = " + str(round(lass_rme, 2)) + 'MWh')


def display_scores(scores):
    print()
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


lass_scores = cross_val_score(lass, X_test, y_test, scoring="neg_mean_squared_error", cv=10)
lass_rmse_scores = np.sqrt(-lass_scores)
display_scores(lass_rmse_scores)

# save model
# joblib.dump(lass, "lasso_model.pkl")
# loaded_model = joblib.load("lasso_model.pkl")

# ----------------------------------------------------------------------------------------------------------------------
# plot data on graphs
# convert periods to days
days = np.arange(len(y_lass_prediction))/48
max_days = 5*48

# plot data on one graph.
plt.figure(figsize=(18, 10))
plt.plot(days[:max_days], y_lass_prediction[:max_days],  color='green', linewidth=2, linestyle='solid', label="LASSO")
plt.plot(days[:max_days], y_test.values[:max_days],  color='black', linewidth=2, linestyle='dashed',
         label="Actual NIV")
plt.ylabel('NIV')
plt.ylabel('Days')
plt.title(' Lasso model: Comparison of First 100 Validation Values')
plt.legend()
plt.show()
