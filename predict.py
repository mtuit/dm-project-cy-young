import pandas as pd
import xgboost as xgb
import numpy as np

# Predicting the AL CY Young award winner
data = pd.read_csv('data/AL_2018.csv')
test = data.drop(labels=['Name', 'Team', 'xFIP'], axis=1)

bst = xgb.XGBModel()
bst.load_model('MLB-model.model')
ypreds = bst.predict(test)

max_value = max(ypreds)
max_value_index = int(np.where(ypreds == max_value)[0])

print(data.ix[max_value_index])

# Predicting the NL CY Young award winner
data = pd.read_csv('data/NL_2018.csv')
test = data.drop(labels=['Name', 'Team', 'xFIP'], axis=1)

bst = xgb.XGBModel()
bst.load_model('MLB-model.model')
ypreds = bst.predict(test)

max_value = max(ypreds)
max_value_index = int(np.where(ypreds == max_value)[0])

print(data.ix[max_value_index])

