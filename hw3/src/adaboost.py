import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10, lambda_reg: float=0.001):
        self.sample_weights = None
        self.learners = [WeakClassifier(input_dim=input_dim) for _ in range(num_learners)]
        self.alphas = []
        self.lambda_reg = lambda_reg

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        X_train = torch.from_numpy(X_train).float()  
        y_train = torch.from_numpy(y_train).float()

        num_samples = X_train.shape[0]
        self.sample_weights = torch.ones(num_samples) / num_samples  

        losses_of_models = []

        for i, model in enumerate(self.learners):
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            for epoch in range(num_epochs):
                y_pred = model(X_train)
                loss = nn.BCELoss(weight=self.sample_weights.detach())(y_pred.squeeze(), y_train)
                
                ######
                l2_reg = torch.tensor(0.0)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += self.lambda_reg * l2_reg
                ###
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model_loss = loss.item()
            losses_of_models.append(model_loss)
            print(f"Learner {i+1} final loss: {model_loss}")

         
            epsilon = self.sample_weights.dot(torch.abs(y_pred.squeeze() - y_train))
            alpha = 0.5 * torch.log((1.0 - epsilon) / epsilon)
            self.alphas.append(alpha)
            self.sample_weights *= torch.exp(alpha * torch.abs(y_train - y_pred.squeeze()))
            self.sample_weights /= self.sample_weights.sum()

        return losses_of_models
        
    def predict_learners(self, X) -> t.Sequence[int]:
        X = torch.from_numpy(X).float()  
        label = []
        prob = []
        for model in self.learners:
            y_pred_proba = model(X) 
            prob.append(y_pred_proba)
            y_pred_labels = (y_pred_proba >= 0.5).squeeze().detach().cpu().numpy()  
            label.append(y_pred_labels)
            
        return label, prob
    
    def compute_feature_importance(self) -> t.Sequence[float]:
        input_dim = self.learners[0].fc1.weight.size(1)
        feature_importance = torch.zeros(self.learners[0].input_dim)
        for alpha, model in zip(self.alphas, self.learners):
            feature_importance += alpha * torch.sum(torch.abs(model.fc1.weight.data), dim=0)
        return feature_importance.detach().cpu().numpy()

    
