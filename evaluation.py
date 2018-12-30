import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(data_paths, model_name, plot_title, labels_to_drop=None):
    train = pd.concat([pd.read_csv(x) for x in data_paths])

    X = train.drop(labels=labels_to_drop, axis=1)
    y = train.pop('y').values

    bst = xgb.XGBClassifier()
    bst.load_model(model_name)
    bst.fit(X, y)

    xgb.plot_importance(booster=bst, title=plot_title)

    plt.show()

if __name__ == '__main__':
    data_paths = ['data/AL_2007-2017_Standard.csv', 'data/NL_2007-2017_Standard.csv']
    labels_to_drop = ['y', 'Name', 'Team', 'Season']
    plot_title = "Feature Importance Initially"
    model_name = 'MLB-Model-Standard.model'

    plot_feature_importance(data_paths, model_name, plot_title, labels_to_drop)

    labels_to_drop = ['Season', 'Name', 'Team', 'G', 'GS', 'ShO', 'SV', 'HLD', 'BS', 'TBF', 'H', 'R', 'IBB', 'HBP',
                      'BK', 'y']
    model_name = 'MLB-Model-Standard-2nd.model'
    plot_title = "Feature Importance After Dropping Unimportant Features"

    plot_feature_importance(data_paths, model_name, plot_title, labels_to_drop)