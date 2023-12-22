#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("all.csv")
df.head()


# In[6]:


import matplotlib.pyplot as plt
aal = df[df["Name"] == "AAL"]
plt.figure(figsize=(14,7))
plt.plot(pd.to_datetime(aal['date']), aal['close'], label='Closing Price', color='blue')
plt.title('AAL Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[8]:


# Descriptive Statistics
desc_stats = all['close'].describe()
desc_stats


# In[9]:


# Visual inspection

rolling_mean = all['close'].rolling(window=30).mean()
rolling_std = all['close'].rolling(window=30).std()

plt.figure(figsize=(14,7))
plt.plot(aal['date'], aal['close'], label='Closing Price', color='blue')
plt.plot(aal['date'], rolling_mean, label='30-day Moving Average', color='red')
plt.plot(aal['date'], rolling_std, label='30-day Rolling Std Dev', color='green')
plt.title('AAL Stock Closing Prices with Moving Average and Std Dev')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[12]:


# Price Distribution
plt.figure(figsize=(14,7))
plt.hist(aal['close'], bins=30, color='blue',edgecolor='black')
plt.title('Price Distribution of AAL')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[13]:


# Price Correlation
plt.figure(figsize=(14,7))
plt.scatter(aal['open'], aal['close'], alpha=0.5, color='purple',edgecolor='black')
plt.title('Correlation between Opening $ Closing Prices for AAL')
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[14]:


# Autocorrelation Analysis
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,4))

plot_acf(aal['close'], lags=50, ax=axes[0])
axes[0].set_title('Autocorrelation Function ACF')

plot_pacf(aal['close'], lags=50, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function PACF')

plt.tight_layout()
plt.show()


# In[26]:


# Split
train_size = int(len(aal) * 0.8)
train, test = aal.iloc[:train_size], aal.iloc[train_size:]
len(train), len(test)


# In[27]:


def moving_average_forecast(series, window_size):
    """Forecast using Moving Average"""
    return series.rolling(window=window_size).mean().shift(-window_size)

# Apply the moving average to the training data
window_size = 30
train['Forecast_MA'] = moving_average_forecast(train['close'], window_size)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(train['date'], train['close'], label='Actual Price', color='blue')
plt.plot(train['date'], train['Forecast_MA'], label='Moving Average Forecast', color='red', linestyle='dashed')
plt.title('Moving Average Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[15]:


import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("path_to_your_file/all_stocks_5yr.csv")
aal_data = df[df['Name'] == 'AAL']

# Split the data
train_size = int(len(aal_data) * 0.8)
train, test = aal_data.iloc[:train_size], aal_data.iloc[train_size:]

def lstm_forecast(train, test, look_back=5, epochs=10):
    data = train['close'].values.astype('float32')
    data = np.reshape(data, (-1, 1))

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # Convert data to supervised learning problem
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        y.append(data[i + look_back, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=1, verbose=1)

    # Making predictions
    test_data = df['close'].iloc[len(train)-look_back:len(train)+len(test)].values
    test_data = test_data.reshape(-1, 1)
    test_data = scaler.transform(test_data)
    X_test = []
    for i in range(look_back, len(test_data)):
        X_test.append(test_data[i-look_back:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
   
    test['Forecast_LSTM'] = predictions
    return test

# Apply LSTM forecasting
test = lstm_forecast(train, test, look_back=5, epochs=50)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(train['date'], train['close'], label='Training Data')
plt.plot(test['date'], test['close'], label='Actual Price')
plt.plot(test['date'], test['Forecast_LSTM'], label='LSTM Forecast')
plt.title('LSTM Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[30]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

def holt_winters_forecast(train, test):
    model = ExponentialSmoothing(train['close'], trend='add', seasonal='add', seasonal_periods=252)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    test['Forecast_HW'] = forecast.values
    return test

# Apply Holt-Winters forecasting
test_hw = holt_winters_forecast(train, test)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(train['date'], train['close'], label='Training Data', color='blue')
plt.plot(test['date'], test['close'], label='Actual Price', color='green')
plt.plot(test['date'], test_hw['Forecast_HW'], label='Holt-Winters Forecast', color='red', linestyle='dashed')
plt.title('Holt-Winters Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[31]:


from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(train, test, p, d, q):
    model = ARIMA(train['close'], order=(p, d, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    test['Forecast_ARIMA'] = forecast.values
    return test

# Apply ARIMA forecasting using parameters derived from ACF and PACF plots
# Using p=1, d=1, q=1 for this demonstration
test_arima = arima_forecast(train, test, p=1, d=1, q=1)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(train['date'], train['close'], label='Training Data', color='blue')
plt.plot(test['date'], test['close'], label='Actual Price', color='green')
plt.plot(test['date'], test_arima['Forecast_ARIMA'], label='ARIMA Forecast', color='red', linestyle='dashed')
plt.title('ARIMA Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[32]:


from sklearn.linear_model import LinearRegression
import numpy as np

def linear_regression_forecast(train, test):
    X_train = np.array(range(len(train))).reshape(-1, 1)
    y_train = train['close'].values

    X_test = np.array(range(len(train), len(train) + len(test))).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test data
    test['Forecast_LR'] = model.predict(X_test)
    return test

# Apply linear regression forecasting
test = linear_regression_forecast(train, test)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(train['date'], train['close'], label='Training Data', color='blue')
plt.plot(test['date'], test['close'], label='Actual Price', color='green')
plt.plot(test['date'], test['Forecast_LR'], label='Linear Regression Forecast', color='red', linestyle='dashed')
plt.title('Linear Regression Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[37]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def random_forest_forecast(train, test):
    """Forecast using Random Forest"""
    X_train = np.array(range(len(train))).reshape(-1, 1)
    y_train = train['close'].values

    X_test = np.array(range(len(train), len(train) + len(test))).reshape(-1, 1)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Predict on the test data
    test['Forecast_RF'] = model.predict(X_test)
    return test

# Apply random forest forecasting
test = random_forecast(train, test)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(train['date'], train['close'], label='Training Data', color='blue')
plt.plot(test['date'], test['close'], label='Actual Price', color='green')
plt.plot(test['date'], test['Forecast_RF'], label='Random Forest Forecast', color='red', linestyle='dashed')
plt.title('Random Forest Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[38]:


from sklearn.metrics import r2_score

def evaluate_forecast(true_values, predictions):
    """Compute various evaluation metrics"""
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_values - predictions))
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    r2 = r2_score(true_values, predictions)
   
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}

# Evaluate the Moving Average method
ma_metrics = evaluate_forecast(test['close'].iloc[:-window_size], test['Forecast_MA'].iloc[:-window_size])

# Evaluate the Random Forest method
rf_metrics = evaluate_forecast(test['close'], test['Forecast_RF'])

ma_metrics, rf_metrics


# In[ ]:




