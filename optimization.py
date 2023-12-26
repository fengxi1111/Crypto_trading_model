#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install -q neuralforecast')
get_ipython().system('pip show neuralforecast')

from neuralforecast.models import PatchTST
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from neuralforecast.losses.pytorch import MAE
from neuralforecast.tsdataset import TimeSeriesDataset

import pytorch_lightning as pl
import matplotlib.pyplot as plt


# In[ ]:


import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import MLP
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic, augment_calendar_df

AirPassengersPanel, calendar_cols = augment_calendar_df(df=AirPassengersPanel, freq='M')

Y_train_df = AirPassengersPanel[AirPassengersPanel.ds<AirPassengersPanel['ds'].values[-12]] 
Y_test_df = AirPassengersPanel[AirPassengersPanel.ds>=AirPassengersPanel['ds'].values[-12]].reset_index(drop=True) 


# In[ ]:


model = PatchTST(h=12,
                 input_size=104,
                 patch_len=24,
                 stride=24,
                 revin=False,
                 hidden_size=16,
                 n_heads=4,
                 scaler_type='robust',
                 loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
                 #loss=MAE(),
                 learning_rate=1e-3,
                 max_steps=500,
                 val_check_steps=50,
                 early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='M'
)
nf.fit(df=Y_train_df, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)


# In[ ]:


Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

if model.loss.is_distribution_output:
    plot_df = plot_df[plot_df.unique_id=='ETH'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['PatchTST-median'], c='blue', label='median')
    plt.fill_between(x=plot_df['ds'][-12:], 
                    y1=plot_df['PatchTST-lo-90'][-12:].values, 
                    y2=plot_df['PatchTST-hi-90'][-12:].values,
                    alpha=0.4, label='level 90')
    plt.grid()
    plt.legend()
    plt.plot()
else:
    plot_df = plot_df[plot_df.unique_id=='ETH'].drop('unique_id', axis=1)
    plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
    plt.plot(plot_df['ds'], plot_df['PatchTST'], c='blue', label='Forecast')
    plt.legend()
    plt.grid()


# In[ ]:


import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import gresearch_crypto
import time
import datetime

import pickle
import gc

from tqdm import tqdm

TRAIN_CSV = 'train.csv'
ASSET_DETAILS_CSV = 'asset_details.csv'

pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 350)


# In[ ]:


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# In[ ]:


df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
df_asset_details[df_asset_details.Asset_ID.isin([1,6,4])]


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train = pd.read_csv(TRAIN_CSV)\n\n\ndf_train = reduce_mem_usage(df_train)\ndf_train = df_train[df_train.Asset_ID.isin(assets)]\n\ndf_train.rename({'Asset_ID':'unique_id','timestamp':'ds','Target':'y'},axis=1,inplace=True)\ndf_train.ds = pd.to_datetime(df_train.ds, unit='s')\ndf_train")


# In[ ]:


Y_train_df = df_train[ df_train.ds<= '2022-05-30 23:47:00'] 
Y_test_df  = df_train[(df_train.ds> '2022-05-30 23:47:00')&(df_train.ds<=  '2021-05-30 23:59:00')].reset_index(drop=True) # 12 mins
Y_eval_df  = df_train[(df_train.ds> '2022-05-30 23:59:00')].reset_index(drop=True)


# In[ ]:


model = PatchTST(h=12,
                 input_size=104,
                 patch_len=24,
                 stride=24,
                 hidden_size=16,
                 n_heads=4,
                 scaler_type='robust',
                 loss=MAE(),
                 learning_rate=1e-3,
                 max_steps=100,
                 val_check_steps=50,
                 early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='1min'
)

#patchtst doesnt support exogenous variables yet
nf.fit(df=Y_train_df[['ds','unique_id','y']], val_size=12)


# In[ ]:


forecasts = nf.predict(futr_df=Y_test_df[['ds','unique_id','y']])

Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])


plot_df = plot_df[plot_df.unique_id==1].drop('unique_id', axis=1)[-50:]  
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['PatchTST'], c='blue', label='Forecast')
plt.legend()
plt.grid()


# In[ ]:


Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])


plot_df = plot_df[plot_df.unique_id==6].drop('unique_id', axis=1)[-50:]  
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['PatchTST'], c='blue', label='Forecast')
plt.legend()
plt.title('Ethereum forecast vs actuals for 12 mins')
plt.grid()


# In[ ]:


from neuralforecast.models import NBEATS
model = NBEATS(h=12, input_size=104,
               loss=MAE(),
               stack_types = ['identity', 'trend', 'seasonality'],
               max_steps=100,
               val_check_steps=50,
               early_stop_patience_steps=2)

nf = NeuralForecast(
    models=[model],
    freq='1min'
)

nf.fit(df=Y_train_df[['ds','unique_id','y']], val_size=12)
forecasts = nf.predict(futr_df=Y_test_df[['ds','unique_id','y']])
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id==1].drop('unique_id', axis=1)[-50:]  
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['NBEATS'], c='blue', label='Forecast')
plt.title('NBEATS BTC forecast vs actuals for 12 mins')
plt.legend()
plt.grid()


# In[ ]:


Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id==6].drop('unique_id', axis=1)[-50:]  
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['NBEATS'], c='blue', label='Forecast')
plt.title('NBEATS Ethereum forecast vs actuals for 12 mins')
plt.legend()
plt.grid()


# In[ ]:


from neuralforecast.models import TFT
nf = NeuralForecast(
    models=[TFT(h=12, input_size=104,
                loss=MAE(),
                hist_exog_list=['Count','VWAP','Close','Open','Volume','High','Low'],
                max_steps=100,
                val_check_steps=50,
                early_stop_patience_steps=1,
                scaler_type='robust'),
    ],
    freq='1min'
)
nf.fit(df=Y_train_df, val_size=12)
forecasts = nf.predict(futr_df=Y_test_df)
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])


plot_df = plot_df[plot_df.unique_id==1].drop('unique_id', axis=1)[-50:]  
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['TFT'], c='blue', label='Forecast')
plt.title('TFT BTC forecast vs actuals for 12 mins')
plt.legend()
plt.grid()
Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id','ds'])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id==6].drop('unique_id', axis=1)[-50:]  
plt.plot(plot_df['ds'], plot_df['y'], c='black', label='True')
plt.plot(plot_df['ds'], plot_df['TFT'], c='blue', label='Forecast')
plt.title('TFT Ethereum forecast vs actuals for 12 mins')
plt.legend()
plt.grid()

