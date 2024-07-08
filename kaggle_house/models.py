from typing import Iterator
from torch import nn

class KaggleHouseModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # self.m = nn.Sequential(nn.Linear(in_features, 1))
        self.m = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, X):
        X = self.m(X)
        return X
    
    # def to(self, device):
    #     self.m.to(device)

    def parameters(self):
        return self.m.parameters()