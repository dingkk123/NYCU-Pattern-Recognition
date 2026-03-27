import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class LinearRegressionSKLearn:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None
        self.loss_history = []  # plus a list to record the loss in the caculation of LR_GD

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add intercept
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.hstack((self.intercept, self.weights)))


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate=0.0001, epochs=1000):
        self.weights = np.zeros(X.shape[1])
        self.intercept = 0
        self.loss_history = []

        if X.shape == y.shape:  # used to run the test_main dataset
            self.weights = self.weights.reshape(1, -1)

        for ep in range(epochs):
            y_pred = np.dot(X, self.weights) + self.intercept
            loss = compute_mse(y_pred, y)
            self.loss_history.append(loss)

            gd_weight = (2 / len(y)) * np.dot(X.T, y_pred - y)
            gd_bias = np.mean(y_pred - y) * 2

            self.weights -= learning_rate * gd_weight  # update the weights
            self.intercept -= learning_rate * gd_bias  # update the intercept

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss')
        plt.show()


def compute_mse(prediction, ground_truth):
    return np.mean((prediction - ground_truth) ** 2)


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=0.0001, epochs=700000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


"""
    # to check the value of the MSE of LR_CF
    LR_SKLearn = LinearRegressionSKLearn()
    LR_SKLearn.fit(train_x, train_y)
    y_preds_sklearn = LR_SKLearn.predict(test_x)
    mse_sklearn = compute_mse(y_preds_sklearn, test_y)
    logger.info(f'{mse_sklearn=:.4f}')
"""

if __name__ == '__main__':
    main()
