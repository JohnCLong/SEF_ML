import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
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
    renewable_generation_forecast.loc[:, 'solar']+renewable_generation_forecast.loc[:, 'wind_off'] +
    renewable_generation_forecast.loc[:, 'wind_on']
)

# locate the relevant wind data, then calculate the difference between them
# the wind forcaste data is hourly so fill forward to fill NaN values.
wind_forecast = wind_generation_forecast_and_outturn.loc[:, ['initialWindForecast', 'latestWindForecast',
                                                             'windOutturn']]
wind_forecast['Val_Diff'] = wind_forecast['initialWindForecast'] - wind_forecast['latestWindForecast']
wind_forecast.fillna(method='ffill', inplace=True)

# define the features needed to train the model
NIV = derived_system_wide_data.loc[:, 'indicativeNetImbalanceVolume']
forecast_renewables = renewable_generation_forecast.loc[:, 'RenewablePrediction']
forecast_demand = forecast_demand.loc[:, 'TSDF']
forecast_generation = generation_day_ahead.loc[:, 'quantity']

# combine all features into one data frame
df = pd.concat([NIV, generation_per_type, apx_price, renewable_generation_forecast, forecast_demand,
                generation_day_ahead, initial_demand_outturn, interconnectors, loss_of_load_probability,
                market_index_data, wind_forecast, ], axis=1, sort=True)
# df = pd.concat([NIV, forecast_renewables, forecast_demand, forecast_generation, wind_forecast], axis=1, sort=True)
df = df.drop("intnemGeneration", axis=1)
df.dropna(inplace=True)

# Get names of indexes for which column generation has value less thatn 10GW and drop
indexNames = df[df['quantity'] < 10000].index
df.drop(indexNames, inplace=True)
df = df.rename({'indicativeNetImbalanceVolume': 'NIV', 'quantity': 'Generation'}, axis='columns')

# cols = ['RenewablePrediction', 'TSDF', 'Generation', 'initialWindForecast', 'latestWindForecast']
cols_all = ['Biomass', 'HydroPumpedStorage', 'Other', 'Solar', 'solar', 'wind_off', 'APXPrice', 'initialWindForecast']

# take data fomr 2018 onwards to reduce training time, then split data chronologically.
df = df.loc[df.index > 2018000000, :]
train = df.loc[df.index < 2018090000, :]
validate = df.loc[df.index > 2018090000, :]

# create x data and then normalize it by subtracting the mean and dividing by the standard deviation.
X_train = train.loc[:, cols_all]
X_norm = 100*(X_train-X_train.mean())/X_train.std()


y_train = train.loc[:, 'NIV']

# normalize the validation data and separate into x and y variables data frames.
X_validate = validate.loc[:, cols_all]
X_norm_validate = 100*(X_validate-X_validate.mean())/X_validate.std()

y_validate = validate.loc[:, 'NIV']

# train each sklearn model
lin = LinearRegression()
lin.fit(X_norm, y_train)

# calculate the predictions from each model.
y_lin_prediction = lin.predict(X_norm_validate)
lin_mse = mean_squared_error(y_validate, y_lin_prediction)
lin_rme = np.sqrt(lin_mse)

print("Linear Regression model RME = " + str(round(lin_rme, 2)) + 'MWh')

# convert periods to days
days = np.arange(len(y_lin_prediction))/48
max_days = 5*48

# plot data on one graph.
plt.figure(figsize=(18, 10))
plt.plot(days[:max_days], y_lin_prediction[:max_days],  color='orange', linewidth=2, linestyle='solid',
         label="Linear Regression")
plt.plot(days[:max_days], y_validate.values[:max_days],  color='black', linewidth=2, linestyle='dashed', label="Actual NIV")
plt.ylabel('NIV')
plt.ylabel('Days')
plt.title('Linear Regression Model: Comparison of First 100 Validation Values')
plt.legend()
plt.show()
