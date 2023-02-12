import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, train_test_split, validation_curve)   
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from yellowbrick.model_selection import LearningCurve, ValidationCurve

warnings.filterwarnings("ignore")
np.random.seed(42)

VALID_EXPERIMENTS = ['ada', 'dt', 'knn', 'nn', 'svm']

ada_tuning_params = {'n_estimators': range(100, 2100, 100)}
dt_tuning_params = {'max_depth': range(1, 40), 'min_samples_leaf': range(1, 25)}
knn_tuning_params = {'n_neighbors': range(1, 80), 'metric': ['euclidean', 'manhattan', 'minkowski'], 'weights':['uniform', 'distance']}
nn_tuning_params = {'learning_rate_init': [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
svm_rbf_tuning_params = {'C': np.logspace(-5, 5, 11)}
svm_sigmoid_tuning_params = {'C': np.logspace(-5, 5, 11)}

CLF_OBJS = { 'ada': {'clf': AdaBoostClassifier, 
                    'tuning_params': ada_tuning_params, 
                    'params': {'estimator': DecisionTreeClassifier(max_depth=1)} }, 
            'dt': {'clf': DecisionTreeClassifier,
                    'tuning_params': dt_tuning_params, 'params': {} }, 
            'knn': {'clf': KNeighborsClassifier, 
                    'tuning_params': knn_tuning_params, 
                    'params': {} },
            'nn': {'clf': MLPClassifier, 
                    'tuning_params': nn_tuning_params, 
                    'params': {'max_iter': 5000, 'hidden_layer_sizes': (5, 2), 'activation':'logistic', 'verbose': False}},
            'svm_rbf': {'clf': SVC, 
                    'tuning_params': svm_rbf_tuning_params, 
                    'params': {'kernel': 'rbf', 'verbose': False}},
            'svm_sigmoid': {'clf': SVC, 
                    'tuning_params': svm_sigmoid_tuning_params, 
                    'params': {'kernel': 'sigmoid', 'verbose': False}}
            }

def run_experiments(experiment=None, all=False):
    datasets = ['audit_data.csv', 'bank_personal_loan_modelling.csv']
    for dataset in datasets:
        X_train, X_test, y_train, y_test = load_data(dataset)
        if all: 
            for clf_name, obj in CLF_OBJS.items():
                if clf_name == 'svm':
                    run_model(X_train, X_test, y_train, y_test, dataset, 'svm_sigmoid')
                    run_model(X_train, X_test, y_train, y_test, dataset, 'svm_rbf')
                else:
                    run_model(X_train, X_test, y_train, y_test, dataset, clf_name)
        elif experiment in VALID_EXPERIMENTS:
            if experiment == 'svm':
                run_model(X_train, X_test, y_train, y_test, dataset, 'svm_sigmoid')
                run_model(X_train, X_test, y_train, y_test, dataset, 'svm_rbf')
            else:   
                run_model(X_train, X_test, y_train, y_test, dataset, experiment)

def run_model(X_train, X_test, y_train, y_test, dataset, clf_name):
    # https://datascience.stackexchange.com/questions/989/svm-using-scikit-learn-runs-endlessly-and-never-completes-execution
    if 'svm' in clf_name:
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
    clf_dict = CLF_OBJS[clf_name]
    clf_obj = clf_dict['clf']
    tuning_params = clf_dict['tuning_params']
    params = clf_dict['params']
    dataset_name = dataset.split('.')[0]

    print("Running {} Learner for {}".format(clf_name, dataset_name))
    
    # 1. Output accuracy before tuning hyperparameters 
    clf = clf_obj(**params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv = 10

    initial_score = accuracy_score(y_test, y_pred)
    
    # 2. Generate Validation Curve with default classifier
    scoring = 'accuracy'
    for key, value in tuning_params.items():
        if key == 'hidden_layer_sizes' or key == 'metric':
            continue
        generate_validation_curve(clf_obj(**params), clf_name, key, value, scoring, cv, dataset_name, X_train, y_train)

    # 3. Conduct Grid Search and obtain best parameters
    clf = clf_obj(**params)
    grid_params = [tuning_params]
    tuned_clf = GridSearchCV(clf, grid_params, scoring=scoring, cv=cv, verbose=0)
    tuned_clf.fit(X_train, y_train)

    print("Best parameters set found on development set: {}".format(tuned_clf.best_params_))
    print()

    # 4. Generate Learning Curve with tuned classifier
    clf = clf_obj(**params)
    clf.set_params(**tuned_clf.best_params_)
    sizes = np.linspace(0.1, 1.0, 10)
    generate_learning_curve(clf, clf_name, scoring, sizes, cv, 8, dataset_name, X_train, y_train)

    # 5. Fit classifier with best params and record training time
    final_clf = clf_obj()
    final_clf.set_params(**params) 
    final_clf.set_params(**tuned_clf.best_params_)
    start_time = time.time()
    final_clf.fit(X_train, y_train)
    train_time = time.time() - start_time

    print("Train time: {}".format(train_time))
    print()

    # 6. Calculate test score and record query time
    start_time = time.time()
    y_pred = final_clf.predict(X_test)
    query_time = time.time() - start_time

    print("Query time: {}".format(query_time))
    print()

    # 7. Accuracy after tuning hyperparameters 
    final_score = accuracy_score(y_test, y_pred)
    print("Before Tuned Score: {}".format(initial_score))
    print()
    print("After Tuned Score: {}".format(final_score))
    print()
    print()

    # 8. Record metrics
    record_metrics(clf_name, dataset_name, tuned_clf.best_params_, initial_score, final_score, scoring, train_time, query_time)

    # Generate loss curves if using neural network
    if clf_name == 'nn':
        generate_nn_curves(clf_name, final_clf, dataset_name, final_clf.loss_curve_, X_train, y_train, X_test, y_test)

def load_data(dataset):
    # df = pd.read_csv("data/" + dataset, header=None)
    df = pd.read_csv("data/" + dataset)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Dataset Filename: {}".format(dataset))
    print()

    print("Number of features: {}".format(len(df.columns) - 1))
    print()

    print("Total Samples: {}".format(len(df)))
    print()

    return X_train, X_test, y_train, y_test

def generate_validation_curve(model, clf_name, param_name, param_range, scoring, cv, dataset_name, X_train, y_train):
    if 'svm' in clf_name or 'nn' == clf_name:
        train_scores, test_scores = validation_curve(
            model, X_train, y_train, param_name=param_name, param_range=param_range,
            scoring="accuracy", n_jobs=8)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title("Validation Curve with {}".format(clf_name))
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.semilogx(param_range, train_scores_mean, label="Training score", marker='o', color="#0272a2")
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", marker='o', color="#9fc377")
        plt.legend(loc="best")
        plt.savefig("results/{}_model_complexity_{}_{}.png".format(clf_name, dataset_name, param_name))
        plt.clf()

    else: 
        viz = ValidationCurve(model, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
        viz.fit(X_train, y_train)
        viz.show("results/{}_model_complexity_{}_{}.png".format(clf_name, dataset_name, param_name))
        plt.clf()

def generate_nn_curves(clf_name, clf, dataset, loss_curve, X_train, y_train, X_test, y_test):
    # Following code was taken from the users TomDLT and Chenn on Stack Overflow
    # https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
    plt.title("Loss Curve for {}".format(dataset))
    plt.xlabel("epoch")
    plt.plot(loss_curve)
    plt.savefig("results/{}_loss_curve_{}.png".format(clf_name, dataset))
    plt.clf()

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    """ Home-made mini-batch learning
    -> not to be used in out-of-core setting!
    """
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 1200
    N_BATCH = 50
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    clf = MLPClassifier(**clf.get_params())

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            clf.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(clf.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(clf.score(X_test, y_test))

        epoch += 1

    """ Plot """
    plt.plot(scores_train, alpha=0.8, label="Training score", color="#0272a2")
    plt.plot(scores_test, alpha=0.8, label="Cross-validation score", color="#9fc377")
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.savefig("results/{}_accuracy_over_epochs_{}.png".format(clf_name, dataset))
    plt.clf()
    

def generate_learning_curve(model, clf_name, scoring, sizes, cv, n_jobs, dataset_name, X_train, y_train):
    viz = LearningCurve(model, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=n_jobs)
    viz.fit(X_train, y_train)        
    viz.show("results/{}_learning_curve_{}.png".format(clf_name, dataset_name)) 
    plt.clf()

def record_metrics(clf_name, ds_name, best_params, before_tuned_score, after_tuned_score, scoring, train_time, query_time):
    filename = "results/metrics.csv".format(clf_name)
    with open(filename, 'a+') as f:
        timestamp = time.time()
        f.write("{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(timestamp, clf_name, ds_name, best_params, before_tuned_score, after_tuned_score, scoring, train_time, query_time))