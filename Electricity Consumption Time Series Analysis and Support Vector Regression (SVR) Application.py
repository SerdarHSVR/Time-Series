import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

# Load dataset
file_path = 'ElecConsumption.csv'
elec_consumption_data = pd.read_csv(file_path)

elec_consumption_data['DateTime'] = pd.to_datetime(elec_consumption_data['DateTime'])
elec_consumption_data.set_index('DateTime', inplace=True)

subset_data = elec_consumption_data[:5000]

scaler = StandardScaler()
subset_data['Consumption_Standardized'] = scaler.fit_transform(subset_data[['Consumption']])

window_size = 16
X, y = [], []

for i in range(window_size, len(subset_data)):
    X.append(subset_data['Consumption_Standardized'].iloc[i-window_size:i].values)
    y.append(subset_data['Consumption_Standardized'].iloc[i])

X, y = np.array(X), np.array(y)

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

plt.figure(figsize=(10, 6))
autocorrelation_plot(subset_data['Consumption'])
plt.title('Autocorrelation of Subset Data')
plt.show()

param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1, 0.01],
}

tscv = TimeSeriesSplit(n_splits=5)
svr = SVR()

random_search = RandomizedSearchCV(
    estimator=svr,
    param_distributions=param_grid,
    n_iter=10,
    scoring="neg_mean_squared_error",
    cv=tscv,
    random_state=42,
    verbose=1,
)

# Train the model
random_search.fit(X_train, y_train)

# Results
best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)
