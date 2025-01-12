# Time Series Analysis

Welcome to the **Time Series Analysis** repository! This repository contains implementations and experiments on statistical methods, machine learning models, and forecasting techniques for time-series data. Each file in the repository focuses on a specific topic, detailed below:

## File Descriptions

### 1. `Çankaya_PM2.5_Air_Quality_Analysis_and_Autoregressive_Model_Application.py`
- **Topic**: PM2.5 Air Quality Analysis and Autoregressive Model
- **Description**: This script analyzes PM2.5 air quality data from Çankaya, focusing on data cleaning, trend analysis, and forecasting using an Autoregressive (AR) model. Key highlights include stationarity testing with the Augmented Dickey-Fuller test, AIC-based lag selection, and visual comparisons between actual and predicted PM2.5 levels.

### 2. `Electricity_Consumption_Time_Series_Analysis_and_Support_Vector_Regression_Application.py`
- **Topic**: Electricity Consumption and Support Vector Regression
- **Description**: Focuses on electricity consumption data analysis and predictive modeling using Support Vector Regression (SVR). Includes hyperparameter optimization with RandomizedSearchCV and TimeSeriesSplit, as well as visualization of autocorrelation patterns and predictive results.

### 3. `Modeling_and_Forecasting_Techniques_in_Time_Series_Analysis.ipynb`
- **Topic**: Statistical and Machine Learning Techniques
- **Description**: A comprehensive notebook showcasing various time-series modeling approaches, including ARIMA, SARIMA, and ensemble methods. It provides step-by-step guidance on preprocessing, feature engineering, and evaluation metrics for forecasting tasks.

### 4. `Performance_Analysis_and_Hyperparameter_Optimization_of_Machine_Learning_Models.ipynb`
- **Topic**: Hyperparameter Optimization and Performance Analysis
- **Description**: Explores advanced hyperparameter tuning techniques for machine learning models applied to time-series data. Includes detailed comparisons and visualizations of model performance under different configurations.

### 5. `Statistical_and_Machine_Learning_Techniques_for_Time_Series_Data.ipynb`
- **Topic**: Hybrid Modeling for Time Series
- **Description**: Combines statistical and machine learning techniques to enhance time-series predictions. The notebook covers feature extraction, hybrid model design, and case studies demonstrating practical applications.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `statsmodels`

Install dependencies via:
```bash
pip install -r requirements.txt
```

### Running the Scripts
- For `.py` files, run the scripts directly in the terminal using:
  ```bash
  python <filename>.py
  ```
- For the Jupyter Notebook (`.ipynb`), open it using Jupyter Notebook or JupyterLab and execute the cells interactively.

### Example
To visualize PM2.5 trends and model predictions in `Çankaya_PM2.5_Air_Quality_Analysis_and_Autoregressive_Model_Application.py`, run:
```bash
python Çankaya_PM2.5_Air_Quality_Analysis_and_Autoregressive_Model_Application.py
```
This will generate plots showing the time-series data and AR model forecasts.

## Repository Structure
```
Time-Series/
├── Çankaya_PM2.5_Air_Quality_Analysis_and_Autoregressive_Model_Application.py
├── Electricity_Consumption_Time_Series_Analysis_and_Support_Vector_Regression_Application.py
├── Modeling_and_Forecasting_Techniques_in_Time_Series_Analysis.ipynb
├── Performance_Analysis_and_Hyperparameter_Optimization_of_Machine_Learning_Models.ipynb
├── Statistical_and_Machine_Learning_Techniques_for_Time_Series_Data.ipynb
└── README.md
```

## Contributions
Contributions are welcome! If you'd like to improve the existing code or add new time-series methods, feel free to fork this repository and submit a pull request.
