#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np 
import pandas as pd 
import random
import gc

from scipy import spatial
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed=4134
fix_all_seeds(seed)


# In[ ]:


#models = ['XGBRegressor', 'LGBMRegressor', 'RandomForestRegressor']
models = ['LGBMRegressor']
# models = ['CatBoostRegressor']
Asset_ID = 2   # Bitcoin Cash
cv_number = 3   # cross-validation
num_models = 10   # number of models under tuning


# In[ ]:


thr_date = datetime.strptime("2023-01-01", '%Y-%m-%d') # Remove date before
df_train = df_train[(df_train['Asset_ID'].isin(asset_list)) & (df_train['ds'] >= thr_date)].reset_index(drop=True)
df_train = df_train[df_train['ds'] >= thr_date].reset_index(drop=True)
pd.set_option('max_columns',100)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def get_features(data, asset_list, Asset_ID, is_row=False):\n    # FE for data as row of DataFrame\n    \n    asset_ids = asset_list\n    \n    def sliced_hourly_fearture(df, start_hour=3, divide_hour=4):\n        # Get hourly fearture\n        \n        list_hours = list(range(0, 24))\n        list_hours = list_hours[start_hour:] + list_hours[:start_hour]\n        divide_hour_start = 0\n        dic = {}\n        divide_tmp = divide_hour\n        for i in range(1, 7):\n            dic[tuple(list_hours[divide_hour_start: divide_hour])] = i\n            divide_hour_start = divide_hour\n            divide_hour += divide_tmp\n        feature = df.ds.dt.hour.apply(lambda x: [v for k, v in dic.items() if x in k][0])\n        return feature\n\n    def get_other_asset_as_feature(asset_ids: list, data, Asset_ID, df_feat):\n        \n        asset_ids.remove(Asset_ID)\n\n        for asset in asset_ids:\n            df_asset = data[data.Asset_ID == asset][[\'ds\', \'Open\', \'High\', \'Low\', \'Close\', \'VWAP\']]\n\n            df_asset[\'hammer_\'+str(asset)] = df_asset[[\'High\', \'Low\']].mean(axis=1) - df_asset[[\'Open\', \'Close\']].mean(axis=1)\n            df_asset[\'OpnCls_sub_VWAP_\'+str(asset)] = df_asset[[\'Open\', \'Close\']].mean(axis=1) - df_asset[\'VWAP\']\n            df_asset[\'OpnVWAP_2_ClsVWAP_\'+str(asset)] = ((df_asset[\'Open\'] - df_asset[\'VWAP\']) / (df_asset[\'Close\'] - df_asset[\'VWAP\'])).replace([np.inf, -np.inf, np.nan], 0.)\n\n            print(f\'df_asset {asset} {df_asset.shape}\')\n            asset_features = [\'ds\', \'hammer_\'+str(asset), \'OpnCls_sub_VWAP_\'+str(asset), \'OpnVWAP_2_ClsVWAP_\'+str(asset)]    \n            df_feat = pd.merge(df_feat, df_asset[asset_features], how=\'left\', on=\'ds\')\n            print(f\'df_feat {asset} {df_asset.shape}\')\n        return df_feat            \n    \n    \n    \n    # Time features     \n    df_feat = data[[\'ds\', \'Count\', \'Open\', \'High\', \'Low\', \'Close\', \'Volume\', \'VWAP\', \'Target\']].copy()[data.Asset_ID == Asset_ID]\n    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)\n    df_feat = df_feat.dropna()    \n    \n#     df_feat[\'ds\'] = pd.to_datetime(df_feat["timestamp"], unit="s", infer_datetime_format=True)\n    print(df_feat.shape)\n    print(\'after dropna 1\', df_feat[\'ds\'].max())\n    print(df_feat.shape)\n\n    df_feat[\'high2low\'] = df_feat[\'High\'] / df_feat[\'Low\']\n    df_feat[\'volume2count\'] = df_feat[\'Volume\'] / (df_feat[\'Count\'] + 1)\n    \n    if is_row:\n        df_feat["hour"] = df_feat[\'ds\'].hour\n        df_feat["dayofweek"] = df_feat[\'ds\'].dayofweek \n        df_feat["month"] = df_feat[\'ds\'].month  # my upgrade\n        df_feat["year"] = df_feat[\'ds\'].year    # my upgrade\n    else:\n        df_feat["hour"] = df_feat[\'ds\'].dt.hour\n        df_feat["dayofweek"] = df_feat[\'ds\'].dt.dayofweek \n        df_feat["month"] = df_feat[\'ds\'].dt.month  # my upgrade\n        df_feat["year"] = df_feat[\'ds\'].dt.year     # my upgrade    \n    \n    # Two new features from the competition tutorial\n    df_feat[\'upper_shadow\'] = df_feat[\'High\'] - np.maximum(df_feat[\'Close\'], df_feat[\'Open\'])\n    df_feat[\'lower_shadow\'] = np.minimum(df_feat[\'Close\'], df_feat[\'Open\']) - df_feat[\'Low\']\n    \n       \n    # LM fearatures\n    df_feat[\'h_l\'] = df_feat[\'High\'] - df_feat[\'Low\']\n    df_feat[\'return\'] = (df_feat[\'Close\'] - df_feat[\'Open\']) / df_feat[\'Open\']\n    df_feat[\'close2open\'] = (df_feat[\'Close\'] - df_feat[\'Open\']) / 2\n    df_feat[\'hlco_ratio\'] = (df_feat[\'High\'] - df_feat[\'Low\'])/(df_feat[\'Close\'] - df_feat[\'Open\'] + 1.001)\n#     df_feat[\'ohlc\'] = np.mean((df_feat[\'High\'].values, df_feat[\'Open\'].values, df_feat[\'Low\'].values, df_feat[\'Close\'].values))\n    df_feat[\'Close_position\']=(data[\'Close\'] - data[\'Low\'])/(data[\'High\'] - data[\'Low\'] + 1.001)\n    df_feat[\'Open_position\']=(data[\'Open\'] - data[\'Low\'])/(data[\'High\'] - data[\'Low\'] + 1.001)\n    \n    #binary features\n    df_feat[\'close_mt_open\'] = (df_feat[\'Close\'] > df_feat[\'Open\']).astype(\'int\')\n    \n#     timestamp = df_train.timestamp.astype(\'datetime64[s]\')\n    df_feat[\'is_weekend\'] = df_feat[\'dayofweek\'].isin([5, 6]).astype(\'int\')\n    df_feat[\'yearday\'] = df_feat[\'ds\'].dt.dayofyear\n    df_feat[\'monthday\'] = df_feat[\'ds\'].dt.day\n    df_feat[\'minute\'] = df_feat[\'ds\'].dt.minute\n    df_feat[\'is_PM\'] = (df_feat[\'hour\'] >= 12).astype(\'int\')\n    df_feat[\'quarter\'] = df_feat[\'ds\'].dt.quarter\n    \n    # quater of hour (as one hot) or one feture {1, 2, 3, 4}\n    df_feat[\'1q\'] = ((df_feat[\'minute\'] < 15) & (df_feat[\'minute\']>=0)).astype(\'int\')\n    df_feat[\'2q\'] = ((df_feat[\'minute\'] < 30) & (df_feat[\'minute\']>=15)).astype(\'int\')\n    df_feat[\'3q\'] = ((df_feat[\'minute\'] < 45) & (df_feat[\'minute\']>=30)).astype(\'int\')\n    df_feat[\'4q\'] = ((df_feat[\'minute\'] < 60) & (df_feat[\'minute\']>=45)).astype(\'int\')\n    \n    #sliced fature\n    df_feat[\'h4_0\'] = sliced_hourly_fearture(df_feat, start_hour=0, divide_hour=4)\n    df_feat[\'h4_1\'] = sliced_hourly_fearture(df_feat, start_hour=1, divide_hour=4)\n    df_feat[\'h4_2\'] = sliced_hourly_fearture(df_feat, start_hour=2, divide_hour=4)\n    df_feat[\'h4_3\'] = sliced_hourly_fearture(df_feat, start_hour=3, divide_hour=4)\n    \n \n    df_feat["close/open"] = df_feat["Close"] / df_feat["Open"] \n#     df_feat["close-open"] = df_feat["Close"] - df_feat["Open"] \n    df_feat["high-low"] = df_feat["High"] - df_feat["Low"] \n    df_feat["high/low"] = df_feat["High"] / df_feat["Low"]\n    \n    # Adding features from SO\n    df_feat[\'hammer\'] = df_feat[[\'High\', \'Low\']].mean(axis=1) - df_feat[[\'Open\', \'Close\']].mean(axis=1)\n    df_feat[\'body\'] = np.abs(df_feat[\'Close\'] - df_feat[\'Open\'])\n    df_feat[\'dozhi\'] = (df_feat[\'upper_shadow\'] + df_feat[\'lower_shadow\']) - df_feat[\'body\']\n    df_feat[\'dozhi_div\'] = ((df_feat[\'upper_shadow\'] + df_feat[\'lower_shadow\']) / df_feat[\'body\']).replace([np.inf, -np.inf, np.nan], 0.)\n    df_feat[\'up_shdw_sub_lo_shdw\'] = df_feat[\'upper_shadow\'] - df_feat[\'lower_shadow\']\n    df_feat[\'up_shdw_div_lo_shdw\'] = (df_feat[\'upper_shadow\'] / df_feat[\'lower_shadow\']).replace([np.inf, -np.inf, np.nan], 0.)\n    df_feat[\'Cls_sub_VWAP\'] = df_feat[\'Close\'] - df_feat[\'VWAP\']\n    df_feat[\'OpnCls_sub_VWAP\'] = df_feat[[\'Open\', \'Close\']].mean(axis=1) - df_feat[\'VWAP\']\n    df_feat[\'OpnVWAP_2_ClsVWAP\'] = ((df_feat[\'Open\'] - df_feat[\'VWAP\']) / (df_feat[\'Close\'] - df_feat[\'VWAP\'])).replace([np.inf, -np.inf, np.nan], 0.)\n    df_feat[\'HighVWAP_2_LowVWAP\'] = ((df_feat[\'High\'] - df_feat[\'VWAP\']) / (df_feat[\'Low\'] - df_feat[\'VWAP\'])).replace([np.inf, -np.inf, np.nan], 0.)\n    \n    if is_row:\n        df_feat[\'mean\'] = df_feat[[\'Open\', \'High\', \'Low\', \'Close\']].mean()\n    else:\n        df_feat[\'mean\'] = df_feat[[\'Open\', \'High\', \'Low\', \'Close\']].mean(axis=1)\n\n    df_feat["high/mean"] = df_feat["High"] / df_feat["mean"]\n    df_feat["low/mean"] = df_feat["Low"] / df_feat["mean"]\n    \n    if is_row:\n        df_feat["median"] = df_feat[["Open", "High", "Low", "Close"]].median()\n    else:\n        df_feat["median"] = df_feat[["Open", "High", "Low", "Close"]].median(axis=1)\n    df_feat["high/median"] = df_feat["High"] / df_feat["median"]\n    df_feat["low/median"] = df_feat["Low"] / df_feat["median"]\n\n    for col in [\'Open\', \'High\', \'Low\', \'Close\', \'VWAP\']:\n        df_feat[f"log_1p_{col}"] = np.log1p(df_feat[col])\n        \n    if is_row:\n        df_feat[\'mean_main\'] = df_feat[[\'Open\', \'Close\']].mean()\n    else:\n        df_feat[\'mean_main\'] = df_feat[[\'Open\', \'Close\']].mean(axis=1)\n\n    df_feat["high/mean_main"] = df_feat["High"] / df_feat["mean_main"]\n    df_feat["low/mean_main"] = df_feat["Low"] / df_feat["mean_main"]\n    \n    if is_row:\n        df_feat["median_main"] = df_feat[["Open", "Close"]].median()\n    else:\n        df_feat["median_main"] = df_feat[["Open", "Close"]].median(axis=1)\n    df_feat["high/median_main"] = df_feat["High"] / df_feat["median_main"]\n    df_feat["low/median_main"] = df_feat["Low"] / df_feat["median_main"]\n\n    #     # ******** Additional features ******\n    df_feat[\'Rel_Upper\'] = ((df_feat[\'High\'] - df_feat[\'VWAP\']) / (df_feat[\'High\'] - df_feat[\'Low\'] + 0.0001))#.replace([np.inf, -np.inf, np.nan], 0.)\n    df_feat[\'Upper_VWAP\'] = ((df_feat[\'High\'] - df_feat[\'VWAP\']) / df_feat[\'VWAP\'] )#.replace([np.inf, -np.inf, np.nan], 0.)\n    df_feat[\'Upper_Vol\'] = ((df_feat[\'High\'] - df_feat[\'VWAP\']) / df_feat[\'Volume\'])#.replace([np.inf, -np.inf, np.nan], 0.)\n    \n    # ******** Additional features ******\n    df_feat[\'is_month_start\'] = df_feat[\'ds\'].dt.is_month_start.astype(int).astype(\'uint8\')\n    df_feat[\'is_month_end\'] = df_feat[\'ds\'].dt.is_month_end.astype(int).astype(\'uint8\')\n    df_feat[\'is_quarter_start\'] = df_feat[\'ds\'].dt.is_quarter_start.astype(int).astype(\'uint8\')\n    df_feat[\'is_quarter_end\'] = df_feat[\'ds\'].dt.is_quarter_end.astype(int).astype(\'uint8\')\n    df_feat[\'is_year_start\'] = df_feat[\'ds\'].dt.is_year_start.astype(int).astype(\'uint8\')\n    df_feat[\'is_year_end\'] = df_feat[\'ds\'].dt.is_year_end.astype(int).astype(\'uint8\')\n\n    # ******** Additional features ******\n    df_feat[\'Dollars\'] = df_feat[\'Volume\'] * df_feat[\'Close\']\n    #Volume_per_trade = Volume/Count\n    df_feat[\'Dollars_per_trade\'] = df_feat.Dollars / (df_feat.Count + 1)\n\n    df_feat[\'log_ret\'] = np.log1p(df_feat.Close/df_feat.Open)\n    df_feat[\'log_ret_H\'] = np.log1p(df_feat.High/df_feat.Close)\n    df_feat[\'log_ret_L\'] = np.log1p(df_feat.Close/df_feat.Low)\n    df_feat[\'log_ret_VWAP\'] = np.log1p(df_feat.Close/df_feat.VWAP)\n    \n    df_feat[\'GK_vol\'] = (1 / 2 * np.log1p(df_feat.High/df_feat.Low) ** 2 - (2 * np.log(2) - 1) * np.log(df_feat.Close/df_feat.Open) ** 2)\n    df_feat[\'RS_vol\'] = np.log(df_feat.High/df_feat.Close)*np.log(df_feat.High/df_feat.Open) + np.log(df_feat.Low/df_feat.Close)*np.log(df_feat.Low/df_feat.Open)\n    print(f"shape before oher feat {df_feat.shape}")\n    # get features from another asset\n    \n    \n    df_feat = get_other_asset_as_feature(asset_ids, data, Asset_ID, df_feat)\n    \n    print(\'after FE\', df_feat[\'ds\'].max())\n    #df_feat = df_feat[~df_feat.isin([np.nan, np.inf, -np.inf]).any(1)].reset_index(drop=True)\n    #df_feat = df_feat.dropna(how="any")\n    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)\n    df_feat = df_feat.dropna()\n    print(\'after dropna 2\', df_feat[\'ds\'].max())\n    \n    return df_feat\n\ndf_train = get_features(df_train, asset_list, Asset_ID=Asset_ID)\ndisplay(df_train)\ndf_train.shape')


# In[ ]:


def get_train_test_data_from_02_to_05_month(df, scaler=False):
    # Get train, train_target, valid, valid_target in 02, 03, 04 months
    
    df_02_05 = df[df.ds.dt.month.isin([2, 3, 4])]
    
    # select 10% fraction in each year for test
    df_02_05_test3 = df_02_05[df.ds.dt.year == 2020].sample(frac=0.1, random_state=seed).sort_values(by='ds')
    df_02_05_test4 = df_02_05[df.ds.dt.year == 2021].sample(frac=0.1, random_state=seed).sort_values(by='ds')
    
    #crate train and test 
    df_02_05_test = pd.concat([df_02_05_test3, df_02_05_test4])

    df_02_05_train = df_02_05[ ~df_02_05.isin(df_02_05_test)].dropna()
    df_02_05_train = df_02_05_train.reset_index(drop=True)
     
    def get_X_y(df):
        # Get dateframe, it's target from 'Target' and drop 'ds'
        y = df.pop('Target')
#         X = df.drop(columns=['ds', 'timestamp']).reset_index(drop=True)        
        X = df.drop(columns=['ds']).reset_index(drop=True)        

        return X, y

    # Get train/valid datasets
    train, train_target = get_X_y(df_02_05_train)
    valid, valid_target = get_X_y(df_02_05_test)
    print(f'Length of the train dataset - {len(train)}, length of the valid dataset - {len(valid)}')
        
    display(train[:3])
    display(valid[:3])
    print('Nulls train: ', train.isnull().sum().sum(), 'Nulls test: ', valid.isnull().sum().sum())
    
    # Standartization
    if scaler:
        scaler = StandardScaler().fit(train)
        train = pd.DataFrame(scaler.transform(train), columns = train.columns)
        valid = pd.DataFrame(scaler.transform(valid), columns = valid.columns)
    
    return train, train_target, valid, valid_target


# In[ ]:


def model_training(model_type, train, train_target):
    # Model training
        
    def hyperopt_model_score(params):
        # Get the modeling score
        
        if model_type=='XGBRegressor':
            clf = XGBRegressor(**params)
        elif model_type=='LGBMRegressor':
            clf = LGBMRegressor(**params)
        elif model_type == 'RandomForestRegressor':
            clf = RandomForestRegressor(**params)
        elif model_type == 'CatBoostRegressor':
            clf = CatBoostRegressor(**params)
            
        current_score = cross_val_score(clf, train, train_target, cv=cv_number).mean()
        print(current_score, params)
        return current_score 

    
    if model_type=='XGBRegressor':
        space_par = {
                    'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.0001),
                    'n_estimators': hp.choice('n_estimators', range(500, 1200)),
                    'max_depth':  hp.choice('max_depth', np.arange(5, 8, dtype=int)),
                    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),
#                     'tree_method': 'gpu_hist'
                    }

    elif model_type=='LGBMRegressor':
        space_par = {
                     'learning_rate': hp.quniform('learning_rate', 0.1, 0.15, 0.01),
                     'n_estimators': hp.choice('n_estimators', range(1000, 3000)),
                     'max_depth':  hp.choice('max_depth', np.arange(8, 13,  dtype=int)),
                     'num_leaves': hp.choice('num_leaves', np.arange(300, 800, dtype=int)),
#                      'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
                     'objective': 'regression',
                     'device': 'gpu'
                    }
    
    elif model_type == 'CatBoostRegressor':
        space_par = {
                     'learning_rate': hp.quniform('learning_rate', 0.1, 0.3, 0.01),
                     'iterations': hp.choice('iterations', range(3000, 8000)),
                     'max_depth':  hp.choice('max_depth', np.arange(9,13, dtype=int)),
                     #'num_leaves': hp.choice('num_leaves', np.arange(300, 1000, dtype=int)),
                     'l2_leaf_reg': hp.quniform('l2_leaf_reg', 0.1, 0.5, 0.05),                     
                     'task_type': 'GPU',
                     'verbose': 0,                                
                    }    
    best = fmin(fn=hyperopt_model_score, space=space_par, algo=tpe.suggest, max_evals=3)
    print('best:\n', best)
    params = space_eval(space_par, best)
    print('Model parameters:\n',params)
        
    return params


# In[ ]:


def model_fit(model_type, train, train_target, valid, valid_target, res, w):
    # Prediction by the model with optimal parameters

    params = model_training(model_type, train, train_target)
    model = LGBMRegressor(**params)        
#     model = CatBoostRegressor(**params)        

    model.fit(train, train_target)
    y_pred = model.predict(valid)
    res.loc[w, 'r2_score'] = round(r2_score(valid_target, y_pred),4)
    res.loc[w, 'corr'] = round(np.corrcoef(valid_target, y_pred)[0][1],4)
    res.loc[w, 'mae'] = round(mean_absolute_error(valid_target, y_pred),4)
    res.loc[w, 'mse'] = round(mean_squared_error(valid_target, y_pred),4)
    res.loc[w, 'rmse'] = round(mean_squared_error(valid_target, y_pred, squared=False),4)
    res.loc[w, 'cos_dist'] = round(1 - spatial.distance.cosine(valid_target, y_pred),4)
    
    return res, model


# In[ ]:


def plotImp(model, name_model, X, acc, num = 100, fig_size = (50, 70)):
    # Draw FI diagram
    
    feature_imp = pd.DataFrame({'Feature':X.columns, 'Value':model.feature_importances_})
    plt.figure(figsize=fig_size)
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:num])
    plt.title(f'{name_model} Features Diagram')
    plt.tight_layout()
    feature_imp.sort_values(by="Value", ascending=False).to_csv(f'res-FE-{Asset_ID}-{name_model}-acc-{acc}-{num_features}-features.csv', index=False)
    plt.show()    


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Get train and valid datasets\ntrain, train_target, valid, valid_target = get_train_test_data_from_02_to_05_month(df_train, scaler=False)\n\n# Results of the models training\nres = pd.DataFrame(columns = ['asset_name', 'num_train', 'num_valid', 'r2_score', 'mae', 'mse', 'rmse', 'corr', 'cos_dist'])\n\n# Training model\n#name_model = models[i]\nname_model = 'LGBMRegressor'\n# name_model = 'CatBoostRegressor'\nres, model =  model_fit(name_model, train, train_target, valid, valid_target, res, 0)\n\n# Draw FI diagram\nplotImp(model, name_model, train, float(res.loc[0,'cos_dist']))\n\nfeature_imp = pd.DataFrame({'Feature':train.columns, 'Value':model.feature_importances_}).sort_values(by='Value', ascending=False)\ndisplay(feature_imp)")


# In[ ]:


selected_features = feature_imp[feature_imp.Value > 0].Feature[:100].to_list()
print(f'Selected features: {selected_features}')
    
res['asset_name'] = asset_name
res['num_train'] = len(train)
res['num_valid'] = len(valid)
res.sort_values(by='r2_score', ascending=False).to_csv(f'res-metric-{Asset_ID}-{name_model}-score-{res.r2_score[0]}-{num_features}-features.csv', index=False)
res


# In[ ]:


selected_features = feature_imp[feature_imp.Value > 0].Feature[:100].to_list()
print(f'Selected features: {selected_features}')


# In[ ]:




