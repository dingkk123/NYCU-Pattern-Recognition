import typing as t
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(self, inputs: npt.NDArray[float], targets: t.Sequence[int], ) -> None:
        n_samples, n_features = inputs.shape
        self.weights = np.zeros(n_features)  # initialize the num of the weight
        self.intercept = 0

        for ep in range(self.num_iterations):
            predictions = self.sigmoid(np.dot(inputs, self.weights) + self.intercept)

            # calculate the loss use cross entropy
            ep_weights = np.dot(inputs.T, predictions - targets) / n_samples
            ep_intercept = np.sum(predictions - targets) / n_samples

            self.weights -= self.learning_rate * ep_weights
            self.intercept -= self.learning_rate * ep_intercept

    def predict(self, inputs: npt.NDArray[float], ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        predicted_probabilities = self.sigmoid(np.dot(inputs, self.weights) + self.intercept)
        predicted_classes = np.where(predicted_probabilities >= 0.5, 1, 0)

        return predicted_probabilities, predicted_classes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self, alpha=50):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        self.alpha = alpha

    def fit(self, inputs: npt.NDArray[float], targets: t.Sequence[int], ) -> None:
        x0 = inputs[targets == 0]
        self.m0 = np.mean(x0, axis=0)
        x1 = inputs[targets == 1]
        self.m1 = np.mean(x1, axis=0)

        # calculate within-class scatter matrix
        delta = x0 - self.m0
        sw0 = np.dot(delta.T, delta)
        delta = x1 - self.m1
        sw1 = np.dot(delta.T, delta)
        self.sw = sw0 + sw1

        # calculate between-class scatter matrix
        delta = self.m1 - self.m0
        delta = delta.reshape(2, 1)
        self.sb = np.dot(delta, delta.T)

        # calculate Fisher's linear discriminant w
        self.w = np.dot(np.linalg.inv(self.sw + self.alpha * np.eye(inputs.shape[1])), (self.m1 - self.m0))
        # print(self.w)

    def predict(self, inputs: npt.NDArray[float], ) -> t.Sequence[t.Union[int, bool]]:
        proj_x_test = (np.dot(inputs, self.w)).reshape(-1, 1) * (self.w / np.dot(self.w, self.w))

        dist_m0 = np.linalg.norm(proj_x_test - self.m0, axis=1)  # calculate distance between
        dist_m1 = np.linalg.norm(proj_x_test - self.m1, axis=1)  # testing data mean and the training data mean

        y_pred = np.where(dist_m0 < dist_m1, 0, 1)  # predict the class label use the mean

        return y_pred

    def plot_projection(self, inputs: npt.NDArray[float]):
        proj_x_test = (np.dot(inputs, self.w)).reshape(-1, 1) * (self.w / np.dot(self.w, self.w))

        upper_bound = np.max(inputs[:, 0]) + 0.5
        lower_bound = np.min(inputs[:, 0]) - 0.5
        x = [lower_bound, upper_bound]
        slope = self.w[1] / self.w[0]
        y = [slope * x[0], slope * x[1]]

        plt.plot(x, y, lw=1, c='k')  # projected line
        y_pred = self.predict(inputs)

        for i in range(inputs.shape[0]):  # data points(on the line)
            color = 'r' if y_pred[i] == 0 else 'b'
            plt.scatter(proj_x_test[i, 0], proj_x_test[i, 1], s=5, c=color)

        x1 = inputs[y_pred == 0]  # scatter data points
        x2 = inputs[y_pred == 1]
        plt.scatter(x1[:, 0], x1[:, 1], s=5, c='r', label='class 1')
        plt.scatter(x2[:, 0], x2[:, 1], s=5, c='b', label='class 2')

        for i in range(inputs.shape[0]):  # the connect line
            plt.plot([inputs[i, 0], proj_x_test[i, 0]], [inputs[i, 1], proj_x_test[i, 1]], lw=0.5, alpha=0.5, c='k')

        title = f'Project Line: w={slope:>.8f}, b={0}'
        plt.title(title)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    auc = roc_auc_score(y_trues, y_preds)
    return auc


def accuracy_score(y_trues, y_preds) -> float:
    correct = np.sum(y_trues == y_preds)
    total = len(y_trues)
    accuracy = correct / total
    return accuracy


def main():
    # Read data

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-3,  # You can modify the parameters as you want
        num_iterations=150000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
