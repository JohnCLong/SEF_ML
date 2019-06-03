import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

derived_system_wide_data = pd.read_csv('SEF-ML/data/derived_system_wide_data.csv')
derived_system_wide_data.set_index('Unnamed: 0', inplace=True)
derived_system_wide_data.sort_index(inplace=True)

NIV = derived_system_wide_data.loc[:, 'indicativeNetImbalanceVolume']
NIV.fillna(method='ffill', inplace=True)

# drop all values where generation is less than 10Gw (calulated in All_models.py
indexNames = np.loadtxt('Low_Generaion_Index.csv', delimiter=',')
NIV.drop(indexNames, axis=0, inplace=True)

NIV = NIV.loc[NIV.index > 2018090000]
periods = 1
static_train = NIV.copy()
static_pred = static_train.shift(periods=int(periods))

# fill any NaN with 0 and calcualte rme
y_static_pred = static_pred.fillna(0).values
static_mse = mean_squared_error(static_train, y_static_pred)
static_rme = np.sqrt(static_mse)

index = NIV.index
length = range(len(NIV))

print("Static model RME = " + str(round(static_rme, 2)) + 'MWh')

# convert periods to days
days = np.arange(len(y_static_pred))/48
max_days = 5*48

# plot data on one graph.
plt.figure(figsize=(18, 10))
plt.plot(days[:max_days], y_static_pred[:max_days],  color='maroon', linewidth=2, linestyle='solid',
         label="Static Model")
plt.plot(days[:max_days], NIV.values[:max_days],  color='black', linewidth=2, linestyle='dashed', label="Actual NIV")
plt.ylabel('NIV')
plt.ylabel('Days')
plt.title('Static model: Comparison of First 100 Validation Values')
plt.legend()
plt.show()

