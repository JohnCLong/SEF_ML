import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error

generation_per_type = pd.read_csv('SEF-ML/data/actual_aggregated_generation_per_type.csv')
apx_price = pd.read_csv('SEF-ML/data/apx_day_ahead.csv')
generation_forecast = pd.read_csv('SEF-ML/data/day_ahead_generation_forecast_wind_and_solar.csv')
forecast_demand = pd.read_csv('SEF-ML/data/forecast_day_and_day_ahead_demand_data.csv')

generation_per_type.set_index('Unnamed: 0', inplace=True)
apx_price.set_index('Unnamed: 0', inplace=True)
generation_forecast.set_index('Unnamed: 0', inplace=True)
forecast_demand.set_index('Unnamed: 0', inplace=True)

generation_per_type.sort_index(inplace=True)
apx_price.sort_index(inplace=True)
generation_forecast.sort_index(inplace=True)
forecast_demand.sort_index(inplace=True)

generation_forecast.loc[:, 'RenewablePrediction'] = (
    generation_forecast.loc[:, 'solar']+generation_forecast.loc[:, 'wind_off']+generation_forecast.loc[:, 'wind_on']
)

generation_per_type = generation_per_type.loc[:, 'FossilGas']
apx_price = apx_price.loc[:, 'APXPrice']
generation_forecast = generation_forecast.loc[:, 'RenewablePrediction']
forecast_demand = forecast_demand.loc[:, 'TSDF']

df = pd.concat([generation_per_type, apx_price, generation_forecast, forecast_demand], axis=1, sort=True)

df.dropna(inplace=True)

df = df.loc[df.index > 2018000000, :]

train = df.loc[df.index < 2018120000, :]
validate = df.loc[df.index > 2018120000, :]

cols = ['APXPrice', 'TSDF', 'RenewablePrediction']
X_train = train.loc[:, cols]
X_norm = 100*(X_train-X_train.mean())/X_train.std()

y_train = train.loc[:, 'FossilGas']

X_validate = validate.loc[:, cols]
X_norm_validate = 100*(X_validate-X_validate.mean())/X_validate.std()

y_validate = validate.loc[:, 'FossilGas']

lin = LinearRegression()
lin.fit(X_norm, y_train)
ela = ElasticNet(alpha=1000)
ela.fit(X_norm, y_train)

y_lin_prediction = lin.predict(X_norm_validate)
lin_mse = mean_squared_error(y_validate, y_lin_prediction)
lin_rme = np.sqrt(lin_mse)


y_ela_prediction = ela.predict(X_norm_validate)
ela_mse = mean_squared_error(y_validate, y_ela_prediction)
ela_rme = np.sqrt(ela_mse)

plt.plot(y_ela_prediction[:100], data=df, color='blue', linewidth=2, linestyle='solid', label="Elastic Net")
plt.plot(y_lin_prediction[:100], data=df, color='orange', linewidth=2, linestyle='solid', label="Linear Regression")
plt.plot(y_validate.values[:100], data=df, color='black', linewidth=2, linestyle='dashed', label="Actual Gas Genearation")
plt.ylabel('Gas Generaion')
plt.title('Comparrison of First 100 Validaiton Values')
plt.legend()
plt.show()

