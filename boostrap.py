import numpy as np
import pandas as pd
import xgboost as xgb

data = pd.read_csv("data.csv")
split_rate = 0.8
Train = data.iloc[:int(data.shape[0]*split_rate),:]
Test = data.iloc[int(data.shape[0]*split_rate):,:]

y_test = Test['fmv']
X_test = Test.drop(['fmv'],axis = 1)


y_test_pre_list = []
error_model_list = []

Train_sample = Train.sample(frac = 1, replace = True)
y_train_sample = Train_sample['fmv']
X_train_sample = Train_sample.drop(['fmv'],axis = 1)

model = xgb.XGBRegressor()
model.fit(X_train_sample,y_train_sample)

y_train_sample_pre = model.predict(X_train_sample)
error_model = y_train_sample-y_train_sample_pre
error_model_list.append(error_model)
print(error_model.reshape(error_model))
y_test_pre = model.predict(X_test)
y_test_pre_list.append(y_test_pre)
print(y_test_pre.shape)
