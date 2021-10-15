"""
DESCRIPTION

Copyright (C) Weicong Kong, 9/10/2021
"""
import matplotlib.pyplot as plt

import os

import xgboost as xgb
from sklearn.metrics import mean_squared_error

from utilities import *


# specifying the root of the data location
# data_root = os.path.join('/kaggle', 'input', 'petfinder-pawpularity-score')
data_root = r'C:\Users\wkong\IdeaProjects\kaggle_data\petfinder-pawpularity-score'


# eda training data
train_path = os.path.join(data_root, 'train.csv')
print(train_path)
train = pd.read_csv(train_path)
train.head()

# eda test data
test_path = os.path.join(data_root, 'test.csv')
print(test_path)
test = pd.read_csv(test_path)
test.head()


# training for just using the meta data
features = [c for c in test if c != 'Id']
y_col = 'Pawpularity'
train_X = train[features].values
xgb = xgb.XGBRegressor()
train_y = train[y_col].values
xgb.fit(train_X, train_y)

train_pred = xgb.predict(train_X)
print('xgb default model mse on training data: ', mean_squared_error(train_pred, train_y))

# feature importancce
importance = pd.DataFrame(data=xgb.feature_importances_, index=features, columns=['importancce'])
importance.plot(kind='bar')
plt.show()


# Make a submission first
test_X = test[features].values
test_pred = xgb.predict(test_X)


# output generation
sub = build_submission(test, test_pred)

