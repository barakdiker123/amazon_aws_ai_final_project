# import torch as nn
from collections import OrderedDict
import torch.nn as nn


def get_classifier(hidden_units, num_input_layer):
    """
    Create the model architecture
    with 2 hidden layer of size hidden_units
    """
    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(num_input_layer, hidden_units)),
                ("relu1", nn.ReLU()),
                ("drop1", nn.Dropout(p=0.1)),
                ("fc_final", nn.Linear(hidden_units, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )
    return classifier
