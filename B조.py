#!/usr/bin/env python
# coding: utf-8

# In[98]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# In[99]:


path = './'
datasets = pd.read_csv(path + 'train.csv')


# In[100]:



# 1. x, y Data
x = datasets[['id', 'bus_route_id', 'in_out', 'station_code', 'station_name',
              'latitude', 'longitude', '6~7_ride', '7~8_ride', '8~9_ride',
              '9~10_ride', '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff',
              '8~9_takeoff', '9~10_takeoff','10~11_takeoff','11~12_takeoff']].copy() 

# ride 카테고리
x['takeon_avg_6~8'] = (x['6~7_ride'] + x['7~8_ride']) / 2
x['takeon_avg_8~10'] = (x['8~9_ride'] + x['9~10_ride']) / 2
x['takeon_avg_10~12'] = (x['10~11_ride'] + x['11~12_ride']) / 2
x['takeon_avg_ride'] = (x['takeon_avg_6~8'] + x['takeon_avg_8~10'] + x['takeon_avg_10~12']) / 3

# takeoff 카테고리
x['takeoff_avg_6~8'] = (x['6~7_takeoff'] + x['7~8_takeoff']) / 2
x['takeoff_avg_8~10'] = (x['8~9_takeoff'] + x['9~10_takeoff']) / 2
x['takeoff_avg_10~12'] = (x['10~11_takeoff'] + x['11~12_takeoff']) / 2
#x['takeon_avg_takeoff'] = (x['takeoff_avg_6~8'] + x['takeoff_avg_8~10'] + x['takeoff_avg_10~12']) / 3

# date 카테코리
x['date'] = pd.to_datetime(datasets['date']) 
y = datasets[['18~20_ride']]

x['date'] = pd.to_datetime(x['date'])
x['year'] = x['date'].dt.year
x['month'] = x['date'].dt.month
x['day'] = x['date'].dt.day
x['weekday'] = x['date'].dt.weekday
x = x.drop('date', axis=1) 

# 최대값 중앙값 등
# Calculate statistical aggregations
ride_columns = ['6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride', '10~11_ride', '11~12_ride']
takeoff_columns = ['6~7_takeoff', '7~8_takeoff', '8~9_takeoff', '9~10_takeoff', '10~11_takeoff','11~12_takeoff']


# x['takeoff_mean'] = x[takeoff_columns].mean(axis=1)
x['takeoff_median'] = x[takeoff_columns].median(axis=1)
# x['takeoff_std'] = x[takeoff_columns].std(axis=1)

# Calculate maximum and minimum values
x['ride_max'] = x[ride_columns].max(axis=1)
x['ride_min'] = x[ride_columns].min(axis=1)


# Add weekday/weekend feature
x['is_weekend'] = np.where(x['weekday'] < 5, 0, 1)

x['in_out'] = x['in_out'].map({'시내': 0, '시외': 1})
station_name_mapping = {name: i for i, name in enumerate(x['station_name'].unique())}
x['station_name'] = x['station_name'].map(station_name_mapping)
x_encoded = pd.get_dummies(x, columns=['station_name'])
x_encoded = x_encoded.fillna(0)

x = pd.get_dummies(x, columns=['station_name'])
x = x_encoded.fillna(0)


# In[101]:


###########################################기본 전


# In[102]:


import tensorflow as tf
tf.random.set_seed(30) # weight 난수값 조정


# In[103]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, train_size=0.85, random_state=80, shuffle=True
) # 7:3에서 바꿈
print(x)


# In[104]:


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold

param = {
    'n_estimators': [3947],
    'depth': [16],
    'fold_permutation_block': [237],
    'learning_rate': [0.8989964556692867],
    'od_pval': [0.6429734179569129],
    'l2_leaf_reg': [2.169943087966259],
    'random_state': [1417]
}

bagging = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(),
    max_features=7,
    n_estimators=100,
    n_jobs=-1,
    random_state=62
)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
model = GridSearchCV(bagging, param, cv=kfold, refit=True, n_jobs=-1)

depth = param['depth'][0]
l2_leaf_reg = param['l2_leaf_reg'][0]
border_count = param['fold_permutation_block'][0]

print(f"depth: {depth}")
print(f"l2_leaf_reg: {l2_leaf_reg}")
print(f"border_count: {border_count}")


# In[105]:


# MaxAbsScaler  CatBoostRegressor 정확도 :  0.7431 MSE: 6.636857463485287
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# # MinMaxScaler  CatBoostRegressor 정확도 :  0.7426 MSE: 6.759033561188409
#scaler = MinMaxScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)

# StandardScaler 
#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.transform(x_test)

# RobustScaler 
# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# In[106]:


from catboost import CatBoostRegressor


# In[107]:


#3. 훈련 및 평가예측
xgb = XGBRegressor()
cat = CatBoostRegressor()
lgbm = LGBMRegressor()

regressors = [cat]
for model in regressors:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    score = r2_score(y_test, y_predict)
    class_names = model.__class__.__name__
    print('{0} 정확도 : {1: .4}'.format(class_names, score))


# In[108]:


# MSE 계산
mse = mean_squared_error(y_test, y_predict)

# MSE 값 출력
print("MSE:", mse)

import matplotlib.pyplot as plt

plt.scatter(y_test, y_predict)
plt.plot(y_test, y_predict, color='Red')
plt.show()


# In[109]:


# #4. 시각화 - 산점도 그래프 그리기
plt.figure(figsize=(8, 6))
plt.scatter(y_test.values.ravel(), y_predict)
plt.plot([min(y_test.values.ravel()), max(y_test.values.ravel())], [min(y_test.values.ravel()), max(y_test.values.ravel())], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted')
plt.show()


# In[110]:


# #4. 시각화 - 예측 오차의 분포 그래프 그리기
error = y_predict - y_test.values.ravel()
plt.figure(figsize=(8, 6))
plt.hist(error, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Prediction Error Distribution')
plt.show()


# In[111]:


# #4. 시각화 - 상관계수 히트맵
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 1.2)
sns.set(rc = {'figure.figsize':(20, 15)})
sns.heatmap(data=datasets.corr(),
            square = True,
             annot = True,
             cbar = True,
             cmap = 'coolwarm'
            )
plt.show()

