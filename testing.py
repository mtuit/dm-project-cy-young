import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

def tune_hyperparameters(data_paths, model_name, labels_to_drop=None):
    train = pd.concat([pd.read_csv(x) for x in data_paths])

    X = train.drop(labels=labels_to_drop, axis=1).values
    y = train.pop('y').values

    cv = KFold(n_splits=10)

    auc_baselines = []
    for train, test in cv.split(X):
        y_train, y_test = y[train], y[test]

        mean_train = np.mean(y_train)
        baseline_predictions = np.ones(y_test.shape) * mean_train

        auc_baselines.append(accuracy_score(y_test, (baseline_predictions > 0.5).astype(int)))

    print("Baseline Accuracy is {:.2f}".format(np.mean(auc_baselines)))

    params = {
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': .3,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'binary:logistic',
    }

    params['eval_metric'] = "error"
    for train, test in cv.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(params, dtrain, num_boost_round=999, evals=[(dtest, "Test")], early_stopping_rounds=10)
        print("Best error: {:.2f} with {} rounds".format(
            model.best_score,
            model.best_iteration + 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=999,
        seed=42,
        nfold=10,
        metrics='error',
        early_stopping_rounds=10
    )

    print(cv_results['train-error-mean'])

    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(0, 13)
        for min_child_weight in range(0, 10)
    ]

    min_error = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(
            max_depth,
            min_child_weight))

        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=999,
            seed=42,
            nfold=10,
            metrics='error',
            early_stopping_rounds=10
        )

        # Update best error
        mean_error = cv_results['test-error-mean'].min()
        boost_rounds = cv_results['test-error-mean'].argmin()
        print("\tError {} for {} rounds".format(mean_error, boost_rounds))
        if mean_error < min_error:
            min_error = mean_error
            best_params = (max_depth, min_child_weight)

    print("Best params: {}, {}, error: {}".format(best_params[0], best_params[1], min_error))

    params['max_depth'] = 1
    params['min_child_weight'] = 2

    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i / 10. for i in range(7, 11)]
        for colsample in [i / 10. for i in range(7, 11)]
    ]

    min_error = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        print("CV with subsample={}, colsample={}".format(
            subsample,
            colsample))

        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample

        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=999,
            seed=42,
            nfold=5,
            metrics='error',
            early_stopping_rounds=10
        )

        # Update best score
        mean_error = cv_results['test-error-mean'].min()
        boost_rounds = cv_results['test-error-mean'].argmin()
        print("\tError {} for {} rounds".format(mean_error, boost_rounds))
        if mean_error < min_error:
            min_error = mean_error
            best_params = (subsample, colsample)

    print("Best params: {}, {}, Error: {}".format(best_params[0], best_params[1], min_error))

    params['subsample'] = 0.8
    params['colsample_bytree'] = 1.0

    min_error = float("Inf")
    best_params = None

    for eta in [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))

        # We update our parameters
        params['eta'] = eta

        # Run and time CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=999,
            seed=42,
            nfold=5,
            metrics='error',
            early_stopping_rounds=10
        )

        # Update best score
        mean_error = cv_results['test-error-mean'].min()
        boost_rounds = cv_results['test-error-mean'].argmin()
        print("\tError {} for {} rounds\n".format(mean_error, boost_rounds))
        if mean_error < min_error:
            min_error = mean_error
            best_params = eta

    print("Best params: {}, Error: {}".format(best_params, min_error))

    params['eta'] = 0.9

    print(params)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=999,
        evals=[(dtest, "Test")],
        early_stopping_rounds=10
    )

    print("Best Error: {:.2f} in {} rounds".format(model.best_score, model.best_iteration + 1))

    num_boost_round = model.best_iteration + 1

    best_model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "Test")]
    )

    best_model.save_model(model_name)

if __name__ == '__main__':
    # First iteration
    data_paths=['data/AL_2007-2017_Standard.csv', 'data/NL_2007-2017_Standard.csv']
    labels_to_drop = ['y', 'Name', 'Team', 'Season']
    model_name = 'MLB-Model-Standard.model'

    tune_hyperparameters(data_paths, model_name, labels_to_drop)

    # Second Iteration
    labels_to_drop = ['Season', 'Name', 'Team', 'G', 'GS', 'ShO', 'SV', 'HLD', 'BS', 'TBF', 'H', 'R', 'IBB', 'HBP',
                      'BK', 'y']
    model_name = 'MLB-Model-Standard-2nd.model'

    tune_hyperparameters(data_paths, model_name, labels_to_drop)