import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# ----------------------------------------------------------------------------------------------------------------------


def rename_duplicate_columns(data_frame, duplicate):
    global column
    cols = []
    count = 1
    for column in data_frame.columns:
        if column == duplicate:
            cols.append(duplicate + np.str(count))
            count += 1
            continue
    cols.append(column)
    data_frame.columns = cols
    return


def plot_scatter(data_frame, x_name, y_name):
    data_frame.plot(kind='scatter', x=x_name, y=y_name, alpha=0.5, color='b', figsize=(18, 15))
    plt.title(y_name + " vs " + xcol)
    # plt.savefig('pictures/Data Exploration/' + x_name + '_vs_' + y_name + '.png')

    plt.show()
    return


def plot_line(plot_data, n, size):
    plt.figure(figsize=size)
    max_days = n * 48
    plt.plot(n[:max_days], plot_data[:max_days], color='k', linewidth=2)
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# todo: create the following features: porportion of [solare, wind] for total renewables/ porportion of renewables
#  to toal generaion/ rate of change of wind/ previous periods NIV/

# todo: Q for A & J: previous periods NIV, what would a realistic distance inot the past to look into be?, how to
#  calculate the current volitilty of NIV, do you
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
    processed_features['initialWindForecast'].fillna(method='ffill', inplace=True)
    processed_features['latestWindForecast'].fillna(method='ffill', inplace=True)
    processed_features['reserveScarcityPrice'].fillna(0, inplace=True)
    processed_features['drm2HourForecast'].fillna(processed_features['drm2HourForecast'].mean(), inplace=True)
    processed_features['lolp1HourForecast'].fillna(0, inplace=True)

    # Separate targets and features.
    processed_target = processed_features.pop('indicativeNetImbalanceVolume')

    # Create a synthetic features.
    processed_features.loc[:, 'RenewablePrediction'] = (processed_features.loc[:, 'solar'] +
                                                        processed_features.loc[:, 'wind_off'] +
                                                        renewable_generation_forecast.loc[:, 'wind_on'])
    processed_features['Val_Diff'] = processed_features['initialWindForecast'] \
                                     - processed_features['latestWindForecast']

    # Drop all NaN values, this drops
    processed_features.dropna(inplace=True)

    # Rename columns
    processed_target = processed_target.rename("NIV")
    processed_features.rename({'quantity': 'Generation', 'systemBuyPrice':'ImbalancePrice'}, axis='columns',
                              inplace=True)

    return processed_features, processed_target

[processed_features, processed_targets] = preprocess_features(raw_data)
NIV =pd.DataFrame(processed_targets)
df = pd.concat([NIV, processed_features], axis=1, sort=True)

# ----------------------------------------------------------------------------------------------------------------------
# calculate the correlation matrix, isolate the NIV correlations and then order by the abs value (descending)
correlation_matrix = df.corr()
cm_NIV = correlation_matrix['NIV']
cm_NIV = cm_NIV.reindex(cm_NIV.abs().sort_values(ascending=False).index)
# create a new list of the two 5 most correlated values (starting at 1 as list_of_attributes[0] = 'NIV'
list_of_rows = cm_NIV.index.values
features = list_of_rows[0:10]

correlation_features = df[features].corr()
print(correlation_features)

# ----------------------------------------------------------------------------------------------------------------------
df[features].hist(bins=50, figsize=(20, 15))
# plt.savefig('pictures/Data Exploration/Histogram.png')
plt.show()

scatter_matrix(df[features[0:4]], figsize=(20, 18), diagonal='kde')
# plt.savefig('pictures/Data Exploration/Scatter_Matrix.png')
plt.show()

for xcol in features[1:]:
    plot_scatter(df, xcol, 'NIV')

plt.figure(figsize=(20, 15))
plt.matshow(correlation_matrix)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()
