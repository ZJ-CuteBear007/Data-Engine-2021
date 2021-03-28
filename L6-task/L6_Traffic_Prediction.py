import pandas as pd
import numpy as np
from fbprophet import Prophet

from statsmodels.tsa.arima_model import ARMA
#导入数据
train= pd.read_csv('./jetrail/train.csv')
#print(train)
train['Datetime']=pd.to_datetime(train['Datetime'])
train.index=train['Datetime']#将Datetime作为索引
train.drop(['ID','Datetime'],axis=1,inplace=True)
#按照天进行采样

daily_train=train.resample('D').sum()
print(daily_train)

daily_train['ds']=daily_train.index
daily_train['y']=daily_train.Count
daily_train.drop(['Count'],axis = 1, inplace = True)
print(daily_train)
#预测未来7个月的数据

m = Prophet(daily_seasonality=True, seasonality_prior_scale=1)
m.fit(daily_train)
future = m.make_future_dataframe(periods=213)
forecast = m.predict(future)
m.plot_components(forecast)
