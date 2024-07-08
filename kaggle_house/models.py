from typing import Iterator
from torch import nn

class KaggleHouseModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(in_features, 1))

    def forward(self, X):
        X = self.m(X)
        return X
    
    def parameters(self):
        return self.m.parameters()