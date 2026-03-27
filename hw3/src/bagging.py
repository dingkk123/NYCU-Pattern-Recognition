import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

class BaggingClassifier:
    def __init__(self, input_dim: int, alpha: float=0.1) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]
        self.alpha = alpha

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        losses_of_models = []
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()  
        for i, model in enumerate(self.learners):
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=self.alpha)

            losses = []

            for epoch in range(num_epochs):
                indices = [random.randint(0, len(X_train) - 1) for _ in range(len(X_train))]
                X_subset = X_train[indices]
                y_subset = y_train[indices]

                y_pred = model(X_subset)

                loss = criterion(y_pred.squeeze(), y_subset)
                
                l2_reg = 0
                for param in model.parameters():
                    l2_reg += torch.sum(param ** 2)
                l2_reg = l2_reg ** 0.5
                loss += self.alpha * l2_reg
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            losses_of_models.append(losses)

        return losses_of_models

    
    def predict_learners(self, X) -> t.Tuple[t.Sequence[int], t.Sequence[float]]:
        X_tensor = torch.from_numpy(X).float()
        predictions_classes = []
        predictions_probs = []
        for model in self.learners:
            with torch.no_grad():
                outputs = model(X_tensor)
                probabilities = torch.sigmoid(outputs)
            
                predicted_classes = (outputs > 0.5).int()  
                predictions_classes.append(predicted_classes.cpu().numpy())
                predictions_probs.append(probabilities.cpu().numpy())
  
        predictions = np.mean(predictions_classes, axis=0) > 0.5
        return predictions, predictions_probs
    
    
    

    def compute_feature_importance(self, X_test, y_test) -> t.Sequence[float]:
       
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).float()
        criterion = nn.BCEWithLogitsLoss()

        baseline_scores = []
        for model in self.learners:
            with torch.no_grad():
                outputs = model(X_test_tensor)
                loss = criterion(outputs.squeeze(), y_test_tensor)
                baseline_scores.append(loss.item())

        feature_importances = np.zeros(X_test.shape[1])
        for i in range(X_test.shape[1]):
            permuted_X_test = X_test.copy()
            np.random.shuffle(permuted_X_test[:, i])
            permuted_X_test_tensor = torch.from_numpy(permuted_X_test).float()

            scores = []
            for model in self.learners:
                with torch.no_grad():
                    outputs = model(permuted_X_test_tensor)
                    loss = criterion(outputs.squeeze(), y_test_tensor)
                    scores.append(loss.item())

            feature_importances[i] = np.mean(scores) - np.mean(baseline_scores)

        return feature_importances
    
