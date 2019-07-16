import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

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

processed_features = processed_features.loc[processed_features.index > 2016000000, :]
processed_targets = processed_targets.loc[processed_targets.index > 2016000000]

X_train_all = processed_features.loc[processed_features.index < 2018030000, :]
y_train = processed_targets.loc[processed_targets.index < 2018030000]

X_validate_all = processed_features.loc[processed_features.index < 2018030000, :]
y_validate = processed_targets.loc[processed_features.index < 2018030000]

#X_validate_all = processed_features.loc[ 2018030000: 2018090000, :]
#y_validate = processed_targets.loc[ 2018030000  : 2018090000]

#X_test_all = processed_features.loc[processed_features.index > 2018090000, :]
#y_test = processed_targets.loc[processed_targets.index > 2018090000]

# Normalize the validation data and separate into X and y variables data frames.
cols_all = ['ImbalancePrice', 'solar', 'Solar_Frac', 'APXPrice',
       'Biomass', 'Other', 'wind_off', 'initialWindForecast', 'Wind_Frac', 'Val_Diff', ]

X_train = X_train_all.loc[:, cols_all]

X_train_mean = X_train.mean()
X_train_std = X_train.std()
X_train = (X_train - X_train_mean) / X_train_std

X_validate = X_validate_all.loc[:, cols_all]
X_validate = (X_validate-X_train_mean)/X_train_std

# ----------------------------------------------------------------------------------------------------------------------
# Model Training
# Train each sklearn model
lin = LinearRegression()
lin.fit(X_train, y_train)

ela = ElasticNet(alpha=0.460)
ela.fit(X_train, y_train)

lass = Lasso(alpha=0.705)
lass.fit(X_train, y_train)

forest_reg = RandomForestRegressor(n_estimators=400, min_samples_split=2, min_samples_leaf=4, max_features='sqrt',
                                   max_depth=10, bootstrap=True,  random_state=42)
forest_reg.fit(X_train, y_train)

# Calculate the predictions from each model.
y_lin_prediction = lin.predict(X_validate)
lin_mse = mean_squared_error(y_validate, y_lin_prediction)
lin_rme = np.sqrt(lin_mse)


y_ela_prediction = ela.predict(X_validate)
ela_mse = mean_squared_error(y_validate, y_ela_prediction)
ela_rme = np.sqrt(ela_mse)

y_lass_prediction = lass.predict(X_validate)
lass_mse = mean_squared_error(y_validate, y_lass_prediction)
lass_rme = np.sqrt(lass_mse)

y_random_forest_prediction = forest_reg.predict(X_validate)
random_forest_mse = mean_squared_error(y_validate, y_random_forest_prediction)
random_forest_rme = np.sqrt(random_forest_mse)

# Static model input for how many periods the data should be shifted and then shift data.


def static_model(data, steps):
    prediction = data.copy()
    prediction = prediction.shift(periods=int(steps))
    return prediction


# Periods = input("How many settlement periods ahead do you want to predict?: ")
periods = 3
static_pred = static_model(y_validate, periods)

# Fill any NaN with 0 and calcualte rme
y_static_pred = static_pred.fillna(0).values
static_mse = mean_squared_error(y_validate, y_static_pred)
static_rmse = np.sqrt(static_mse)


# Cross validation
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


lin_scores = cross_val_score(lin, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Linear Regression Model")
display_scores(lin_rmse_scores)
print(lin_rme)

lass_scores = cross_val_score(lass, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
lass_rmse_scores = np.sqrt(-lass_scores)
print("LASSO Model")
display_scores(lass_rmse_scores)
print(lass_rme)

ela_scores = cross_val_score(ela, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
ela_rmse_scores = np.sqrt(-ela_scores)
print("Elastic Net Model")
display_scores(ela_rmse_scores)
print(ela_rme)

random_forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
random_forest_rmse_scores = np.sqrt(-random_forest_scores)
print("Random Forests Model")
display_scores(random_forest_rmse_scores)
print(random_forest_rme)

print()
print("cross validatiion scores")
print("Static model RME = " + str(round(static_rmse, 2)) + 'MWh')
print("Lasso model RME = " + str(round(np.mean(lass_rme), 2)) + 'MWh')
print("Elastic Net model RME = " + str(round(np.mean(ela_rme), 2)) + 'MWh')
print("Linear Regression model RME = " + str(round(np.mean(lin_rme), 2)) + 'MWh')
print("Random Forest model RME = " + str(round(np.mean(random_forest_rme), 2)) + 'MWh')
print()
print("==============================================================")
print()
print("cross validatiion scores")
print("Static model RME = " + str(round(static_mse, 2)) + 'MWh')
print("Lasso model RME = " + str(round(np.mean(lass_rmse_scores), 2)) + 'MWh')
print("Elastic Net model RME = " + str(round(np.mean(ela_rmse_scores), 2)) + 'MWh')
print("Linear Regression model RME = " + str(round(np.mean(lin_rmse_scores), 2)) + 'MWh')
print("Random Forest model RME = " + str(round(np.mean(random_forest_rmse_scores), 2)) + 'MWh')
print()
# convert periods to days
days = np.arange(len(y_ela_prediction))/48
max_days = 10*48

# Plot data on one graph.
plt.rcParams["figure.figsize"] = (30, 10)
model = [y_static_pred, y_ela_prediction, y_lin_prediction, y_lass_prediction, y_random_forest_prediction, y_validate]
name = ['Static', 'Elastic Net', 'Linear Regression', 'Lasso', 'Random Forests', 'NIV']
colour = ['m', 'b', 'y', 'g', 'c']

for y_data, c, l in zip(model[:-1], colour, name[:-1]):
    plt.plot(days[:max_days], y_data[:max_days], color=c, linewidth=2, linestyle='solid', label=l)
plt.plot(days[:max_days], y_validate.values[:max_days], color='k', linewidth=2, linestyle='dashed')
plt.title('All models: Comparison of First 100 Validation Values')
plt.xlabel('Days')
plt.ylabel('NIV')
plt.legend()
plt.show()

# Plot data on separate graphs
for y_data, l in zip(model, name):
    plt.plot(days[:max_days], y_validate.values[:max_days], color='k', linewidth=2, linestyle='dashed')
    plt.plot(days[:max_days], y_data[:max_days], color='b')
    plt.xlabel('Days')
    plt.ylabel('NIV')
    plt.title(l)
    plt.show()

