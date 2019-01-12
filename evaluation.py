import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_feature_importance(data_paths, model_name, plot_title, labels_to_drop=None):
    train = pd.concat([pd.read_csv(x) for x in data_paths])

    X = train.drop(labels=labels_to_drop, axis=1)
    y = train.pop('y').values

    bst = xgb.XGBClassifier()
    bst.load_model(model_name)
    bst.fit(X, y)

    xgb.plot_importance(booster=bst, title=plot_title)

    plt.show()


def plot_roc_curve(data_paths, model_name, plot_title, labels_to_drop=None):
    train = pd.concat([pd.read_csv(x) for x in data_paths])
    X = train.drop(labels=labels_to_drop, axis=1)
    y = train.pop('y')

    bst = xgb.XGBClassifier()
    bst.load_model(model_name)
    probs = bst.predict_proba(X)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title(plot_title)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    data_paths = ['data/AL_2007-2017_Standard.csv', 'data/NL_2007-2017_Standard.csv']
    labels_to_drop = ['y', 'Name', 'Team', 'Season']
    model_name = 'MLB-Model-Standard.model'
    plot_title = "Feature Importance of {}".format(model_name)

    plot_feature_importance(data_paths, model_name, plot_title, labels_to_drop)

    plot_title = 'ROC curve of {}'.format(model_name)
    plot_roc_curve(data_paths, model_name, plot_title, labels_to_drop)

    labels_to_drop = ['Season', 'Name', 'Team', 'GS', 'ShO', 'TBF', 'H', 'R', 'IBB', 'HBP', 'y']
    model_name = 'MLB-Model-Standard-2nd.model'
    plot_title = "Feature Importance of {}".format(model_name)

    plot_feature_importance(data_paths, model_name, plot_title, labels_to_drop)

    plot_title = 'ROC curve of {}'.format(model_name)
    plot_roc_curve(data_paths, model_name, plot_title, labels_to_drop)