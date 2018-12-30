import pandas as pd
import xgboost as xgb
import numpy as np

def predict_cy_young(data_paths, model_name, labels_to_drop=None):
    bst = xgb.XGBModel()
    bst.load_model(model_name)

    for data_path in data_paths:
        data = pd.read_csv(data_path)
        to_predict = data.drop(labels=labels_to_drop, axis=1)
        ypreds = bst.predict(to_predict)
        max_value = max(ypreds)
        max_value_index = int(np.where(ypreds == max_value)[0])
        print("The winner for {}".format(data_path))
        print(data.ix[max_value_index])

if __name__ == '__main__':
    data_paths = ['data/AL_2018_Standard.csv', 'data/NL_2018_Standard.csv']
    labels_to_drop = ['Name', 'Team', 'G', 'GS', 'ShO', 'SV', 'HLD', 'BS', 'TBF', 'H', 'R', 'IBB', 'HBP',
                      'BK']
    model_name = 'MLB-Model-Standard-2nd.model'

    predict_cy_young(data_paths, model_name, labels_to_drop)