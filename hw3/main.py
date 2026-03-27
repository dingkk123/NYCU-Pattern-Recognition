import pandas as pd
from loguru import logger
import numpy as np
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc
from matplotlib import pyplot as plt
import torch


def gini(sequence):
    _, cnt = np.unique(sequence, return_counts=True)
    prob = cnt / sequence.shape[0]
    g = 1 - np.sum([p**2 for p in prob])
    return g


def entropy(sequence):
    _, cnt = np.unique(sequence, return_counts=True)
    prob = cnt / sequence.shape[0]
    e = -1 * np.sum([p * np.log2(p) for p in prob])
    return e


def plot_feature_importance_adaboost(feature_importance, feature_names):
    num_features = len(feature_importance)
    plt.figure(figsize=(10, 6))
    plt.barh(range(num_features), feature_importance)
    plt.ylabel('Feature')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.yticks(range(num_features), feature_names, rotation=0)
    plt.tight_layout()
    plt.show()


def plot_feature_importance_bagging(importance, names):
    # Create a sorted list of feature importances and names
    sorted_importances = sorted(zip(importance, names), reverse=True)
    importances, names = zip(*sorted_importances)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.show()


def plot_feature_importance_decision(feature_importance, feature_names):
    sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_features, importance_values = zip(*sorted_feature_importance)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), importance_values, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    plt.show()


def main():
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    feature_names = list(train_df.drop(['target'], axis=1).columns)

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
        lambda_reg=0.001
    )

    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=50000,
        learning_rate=0.01,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_preds=y_pred_probs, y_trues=y_test, fpath='roc_curve.png')
    feature_importance = clf_adaboost.compute_feature_importance()
    plot_feature_importance_adaboost(feature_importance, feature_names)

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
        alpha=0.1
    )

    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=6000,
        learning_rate=0.01,
    )

    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    y_pred_probs_np = [np.asarray(y_pred) for y_pred in y_pred_probs]
    y_pred_probs_tensor = [torch.from_numpy(y_pred_np) for y_pred_np in y_pred_probs_np]
    accuracy_ = np.mean([y_pred == y_test for y_pred in y_pred_classes])
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(y_preds=y_pred_probs_tensor, y_trues=y_test, fpath='roc_curve1.png')
    feature_importance = clf_bagging.compute_feature_importance(X_test, y_test)
    plot_feature_importance_bagging(feature_importance, feature_names)

    # Decision Tree
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print("Gini of data is ", gini(data))
    print("Entropy of data is ", entropy(data))

    clf_tree = DecisionTree(criterion='gini', max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = np.mean(y_pred_classes == y_test)
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    feature_importance = clf_tree.compute_feature_importance(feature_names)
    plot_feature_importance_decision(feature_importance, feature_names)


if __name__ == '__main__':
    main()
