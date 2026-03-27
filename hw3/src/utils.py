import typing as t
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


class WeakClassifier(nn.Module):
  
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)  
        return x

def entropy_loss(outputs, targets):
    raise NotImplementedError


def plot_learners_roc(y_preds: t.List[t.Sequence[float]], y_trues: t.Sequence[int], fpath='./tmp.png'):
    plt.figure(figsize=(10, 8))
    for i, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_pred.detach().numpy()) 
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Learner {i+1} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(fpath)
    plt.show()
   