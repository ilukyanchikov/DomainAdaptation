import torch
import numpy as np


def accuracy_score(y_true: torch.Tensor, y_predict: torch.Tensor):
    """
    :param y_true: Ground truth (correct) labels
    :param y_predict: Predicted labels, as returned by a classifier.
    :return: score
    """
    correct = (y_true == y_predict).sum().item()
    return correct / np.prod(y_predict.shape)


class AccuracyScore:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, y_true: torch.Tensor, y_predict: torch.Tensor):
        self.correct += (y_true == y_predict).sum().item()
        self.total += np.prod(y_predict.shape)

    @property
    def score(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0