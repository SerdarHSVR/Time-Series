import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Load the dataset
file_path = 'desktop/cankaya.xlsx'
data = pd.read_excel(file_path)

# Display basic information and the first few rows of the dataset
data_info = data.info()
data_head = data.head()

data_info, data_head

# Renaming columns for clarity
data.columns = ['Date', 'PM2.5']

# Dropping the first row which contains the header info
data = data.drop(0).reset_index(drop=True)

# Converting PM2.5 values to numeric (handling commas and forcing errors as NaN)
data['PM2.5'] = pd.to_numeric(data['PM2.5'].str.replace(',', '.'), errors='coerce')

# Checking for any remaining null values and basic statistics after conversion
missing_values = data.isnull().sum()
data_description = data.describe()

missing_values, data_description

# Imputing missing values with linear interpolation
data['PM2.5'] = data['PM2.5'].interpolate(method='linear')

# Plotting the time series to observe trends and seasonality
plt.figure(figsize=(14, 6))
plt.plot(data['Date'], data['PM2.5'], color='blue', label='PM2.5 Levels')
plt.title('PM2.5 Levels Over Time (Çankaya)')
plt.xlabel('Date')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.show()

# Performing the Augmented Dickey-Fuller test
adf_test = adfuller(data['PM2.5'].dropna())

# Extracting and displaying test results
adf_results = {
    'ADF Statistic': adf_test[0],
    'p-value': adf_test[1],
    'Critical Values': adf_test[4]
}

adf_results

# Splitting data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data['PM2.5'][:train_size], data['PM2.5'][train_size:]

# Using AIC to determine the best lag (p) for the AR model
best_aic = np.inf
best_lag = 1
for lag in range(1, 25):  # testing lags from 1 to 24
    try:
        model = AutoReg(train, lags=lag, old_names=False).fit()
        if model.aic < best_aic:
            best_aic = model.aic
            best_lag = lag
    except:
        continue

best_lag, best_aic

# Fitting the AR model with the best lag
final_model = AutoReg(train, lags=best_lag, old_names=False).fit()

# Generating predictions on the test set
predictions = final_model.predict(start=len(train), end=len(data) - 1, dynamic=False)

# Plotting actual vs. predicted values
plt.figure(figsize=(14, 6))
plt.plot(data['Date'][train_size:], test, color='blue', label='Actual PM2.5 Levels')
plt.plot(data['Date'][train_size:], predictions, color='red', linestyle='dashed', label='AR Model Predictions')
plt.title('AR Model Predictions vs Actual PM2.5 Levels')
plt.xlabel('Date')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.show()