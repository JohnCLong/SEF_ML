import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error

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

# combine the solar, wind off, wind on into one column describing the renewable genreaiton forecast
renewable_generation_forecast.loc[:, 'RenewablePrediction'] = (
    renewable_generation_forecast.loc[:, 'solar']+renewable_generation_forecast.loc[:, 'wind_off']+
    renewable_generation_forecast.loc[:, 'wind_on']
)

wind_forecast = wind_generation_forecast_and_outturn.loc[:, ['initialWindForecast', 'latestWindForecast', 'windOutturn']]
wind_forecast['Val_Diff'] = wind_forecast['initialWindForecast'] - wind_forecast['latestWindForecast']
wind_forecast.fillna(method='ffill', inplace=True)

# define the features needed to train the model
NIV = derived_system_wide_data.loc[:, 'indicativeNetImbalanceVolume']
forecast_renewables = renewable_generation_forecast.loc[:, 'RenewablePrediction']
forecast_demand = forecast_demand.loc[:, 'TSDF']
forecast_generation = generation_day_ahead.loc[:, 'quantity']

# combine all features into one data frame
data = pd.concat([NIV, generation_per_type, apx_price, renewable_generation_forecast, forecast_demand,
                   generation_day_ahead, initial_demand_outturn, interconnectors, loss_of_load_probability,
                   market_index_data, wind_forecast, ], axis=1, sort=True)
alldf = pd.concat([NIV, forecast_renewables, forecast_demand, forecast_generation, wind_forecast], axis=1, sort=True)
df = data.copy()
df = df.drop("intnemGeneration", axis=1)
df.drop('settlementDate', axis=1)
df.dropna(inplace=True)

df = df.loc[:,~df.columns.duplicated()]

def rename_duplicate_columns(data, duplicate):
    cols = []
    count = 1
    for column in data.columns:
        if column == duplicate:
            cols.append(duplicate+np.str(count))
            count += 1
            continue
    cols.append(column)
    data.columns = cols
    return


# Get names of indexes for which column generation has value less thatn 10GW and drop
indexNames = df[df['quantity'] < 10000].index
df.drop(indexNames, inplace=True)
df = df.rename({'indicativeNetImbalanceVolume': 'NIV', 'quantity': 'Generation'}, axis='columns')

# calculate the correlation matrix, isolate the NIV correlations and then order by the abs value (descending)
correlation_matrix = df.corr()
cm_NIV = correlation_matrix['NIV']
cm_NIV = cm_NIV.reindex(cm_NIV.abs().sort_values(ascending=False).index)
# create a new list of the two 5 most correlated values (starting at 1 as list_of_attributes[0] = 'NIV'
list_of_rows = cm_NIV.index.values
attributes = list_of_rows[1:6]

df[attributes].hist(bins=50, figsize=(20, 15))
plt.show()

scatter_matrix(df[attributes], figsize=(20, 18))
plt.show()

