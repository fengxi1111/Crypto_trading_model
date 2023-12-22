# In[46]:


from keras.backend import set_session
from tensorflow import keras
from tensorflow.keras import layers
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics import tsaplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from warnings import simplefilter
simplefilter('ignore')


# In[9]:


# matplotlib defualts
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout = True)
plt.rc('axes', labelsize = 10, labelweight = 'bold', titlesize = 10, titleweight = 'bold', titlepad = 10)


# In[10]:


# matplotlib configuration for higher images
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = 'retina'")


# In[26]:


# check for missing values
missing_values = df.isnull().sum()
missing_values


# In[27]:


df.shape


# In[28]:


# total missing values
total_missing_values = missing_values.sum()
total_missing_values


# In[29]:


# total datasets count present in the dataset
total_values = np.product(df.shape)
total_values


# In[30]:


# total percent of missing values
percent = (total_missing_values / total_values) * 100
percent


# In[31]:


# Data Cleaning with filling the missing values
df = df.fillna(0)
df.head()


# In[32]:


# calculating the dataframe
train_data = pd.melt(df,id_vars=['Page'],var_name='Date',value_name='Visits')
train_data.head()


# In[33]:


train_data['Date'] = pd.DatetimeIndex(train_data['Date'])
train_data.head()


# In[34]:


# Data Visualization
temp = train_data.groupby('Date')['Visits'].mean()
plt.figure(figsize = (15,4))
plt.xlabel('Date')
plt.ylabel('Avg Views')
plt.title('Average number of views per day')
plt.plot(temp,label='Visits')
plt.legend()
plt.show()


# In[35]:


# Data Visualization
temp = train_data.groupby('Date')['Visits'].median()
plt.figure(figsize = (15,4))
plt.xlabel('Date')
plt.ylabel('Avg Views')
plt.title('Median number of views per day')
plt.plot(temp,label='Visits')
plt.legend()
plt.show()


# In[36]:


train_data['year']=train_data.Date.dt.year
train_data['month']=train_data.Date.dt.month
train_data['day']=train_data.Date.dt.day


# In[37]:


# plotting web traffic over the days of month using heatmap
train_data['month_num'] = train_data['month']
train_data['month'].replace('9','9 - September',inplace=True)
train_data['month'].replace('10','10 - October',inplace=True)
train_data['month'].replace('11','11 - November',inplace=True)
train_data['month'].replace('12','12 - December',inplace=True)


# In[38]:


train_data['weekday'] = train_data['Date'].apply(lambda x: x.weekday())
train_data['weekday#'] = train_data['weekday']
train_data['weekday'].replace(0,'01 - Monday',inplace=True)
train_data['weekday'].replace(1,'02 - Tuesday',inplace=True)
train_data['weekday'].replace(2,'03 - Wednesday',inplace=True)
train_data['weekday'].replace(3,'04 - Thursday',inplace=True)
train_data['weekday'].replace(4,'05 - Friday',inplace=True)
train_data['weekday'].replace(5,'06 - Saturday',inplace=True)
train_data['weekday'].replace(6,'07 - Sunday',inplace=True)


# In[39]:


train_group = train_data.groupby(["month", "weekday"])['Visits'].mean().reset_index()
train_group = train_group.pivot(index='weekday', columns='month',values='Visits')
train_group.sort_index(inplace=True)


# In[40]:


sns.set(font_scale=3.5)
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(train_group, annot=False, ax=ax, fmt="d", linewidths=2)
plt.title('Web Traffic of various months(9-12) across weekdays')
plt.show()


# In[41]:


cols_to_drop = ['year','month','day','month_num','weekday','weekday','weekday#']
train_data.drop(cols_to_drop,axis=1,inplace=True)


# In[42]:


# top 10 pages with maximum number of views
top_pages = train_data.groupby('Page')['Visits'].sum().reset_index()
top_pages_list = top_pages.nlargest(10,'Visits')['Page'].tolist()
top5_pages_df = train_data[train_data['Page'].isin(top_pages_list)]
top5_pages_df.head()

