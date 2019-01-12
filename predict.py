import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from numpy import sort


def plot_winners(ypreds, players, plot_title):
    _, ax = plt.subplots(1, 1)

    ypreds = ypreds * 100

    print(ypreds)

    tuples = list(zip(ypreds, players))
    tuples = sorted(tuples, key=lambda x: x[0])
    tuples = [x for x in tuples if (x[0] > 0.5)]
    ypreds, players = zip(*tuples)
    ylocs = np.arange(len(ypreds))

    ax.barh(ylocs, ypreds, align='center')

    ax.set_yticks(ylocs)
    ax.set_yticklabels(players)
    ax.set_title(plot_title)

    ax.set_ylabel("Player")
    ax.set_xlabel("Probability in percentage")
    plt.show()


def predict_cy_young(data_path, model_name, labels_to_drop=None):
    bst = xgb.XGBModel()
    bst.load_model(model_name)

    data = pd.read_csv(data_path)
    to_predict = data.drop(labels=labels_to_drop, axis=1)
    ypreds = bst.predict(to_predict)

    return ypreds

if __name__ == '__main__':
    data_path = 'data/AL_2018_Standard.csv'
    labels_to_drop = ['Name', 'Team', 'GS', 'ShO', 'TBF', 'H', 'R', 'IBB', 'HBP']
    model_name = 'MLB-Model-Standard-2nd.model'
    players_AL = pd.read_csv(data_path).pop('Name').values

    preds = predict_cy_young(data_path, model_name, labels_to_drop)
    plot_winners(preds, players_AL, "Probabilities of winning Cy Young for AL")

    data_path = 'data/NL_2018_Standard.csv'
    players_NL = pd.read_csv(data_path).pop('Name').values
    preds = predict_cy_young(data_path, model_name, labels_to_drop)
    plot_winners(preds, players_NL, "Probabilities of winning Cy Young for NL")