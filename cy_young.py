import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn import model_selection
from sklearn.svm import SVC


# train = pd.read_csv('data/AL_2007-2017.csv')
# test = pd.read_csv('data/AL_2018.csv')

train = pd.read_csv('data/AL_2007-2017.csv')
train = pd.read_csv('data/NL_2007-2017.s')
train = train.drop(labels=['y', 'Name', 'Team'], axis=1)
train = train.apply(zscore)
print(train)

# X_train = train.drop(labels=['y', 'Name', 'Team', 'xFIP', 'HR/FB'], axis=1)
# Y_train = train.pop('y')
#
# X_test = test.drop(labels=['Name', 'Team', 'xFIP', 'HR/FB'], axis=1)
#
# dtrain = xgb.DMatrix(X_train, label=Y_train)
# dtest = xgb.DMatrix(X_test)
#
# param = {'max_depth': 1, 'min_child_weight': 2, 'eta': 0.9, 'subsample': 0.8, 'colsample_bytree': 1.0, 'objective': 'binary:logistic', 'eval_metric': 'error'}
# num_round = 999
#
# bst = xgb.train(param, dtrain, num_round, evals=[(dtest, "Test")], early_stopping_rounds=10)
#
# preds = bst.predict(dtest)
# print(preds)
#
# max_value = max(preds)
# max_value_index = int(np.where(preds == max_value)[0])
#
# print(test.ix[max_value_index])
#
# train = pd.read_csv('data/NL_2007-2017.csv')
# test = pd.read_csv('data/NL_2018.csv')
#
# X_train = train.drop(labels=['y', 'Name', 'Team', 'xFIP'], axis=1)
# Y_train = train.pop('y')
#
# X_test = test.drop(labels=['Name', 'Team', 'xFIP'], axis=1)
#
#
# dtrain = xgb.DMatrix(X_train, Y_train)
# dtest = xgb.DMatrix(X_test)
#
#
#
# bst = xgb.train(param, dtrain, num_round)
# preds = bst.predict(dtest)
#
# print(preds)
#
# max_value = max(preds)
# max_value_index = np.where(preds == max_value)
#
# # Multiple winners
# if len(max_value_index[0]) > 1:
#     for index in max_value_index[0]:
#         print(test.ix[max_value_index[0][index]])
#
# max_value = max(preds)
# max_value_index = int(np.where(preds == max_value)[0])
#
# print(test.ix[max_value_index])