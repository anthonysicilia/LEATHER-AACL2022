import torch
from torch.autograd import Variable
import json

def calculate_accuracy(predictions, targets):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    predicted_classes = predictions.topk(1)[1]
    accuracy = torch.eq(predicted_classes.squeeze(1), targets).sum()/targets.size(0)
    return accuracy

def calculate_agreement(a, b):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(a, Variable):
        a = a.data
    if isinstance(b, Variable):
        b = b.data

    a = a.topk(1)[1]
    b = b.topk(1)[1]

    return torch.eq(a.squeeze(1), b.squeeze(1)).sum() / b.size(0)
