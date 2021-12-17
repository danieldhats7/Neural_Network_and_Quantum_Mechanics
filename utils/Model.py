import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, D_in, D_out):
        super(Model, self).__init__()

        self.c1 = nn.Linear(D_in, 256)
        self.c2 = nn.ReLU()
        self.c3 = nn.Dropout(p=0.1)
        self.c4 = nn.Linear(256, 256)
        self.c5 = nn.ReLU()
        self.c6 = nn.Dropout(p=0.1)
        self.c7 = nn.Linear(256, 128)
        self.c8 = nn.ReLU()
        self.c9 = nn.Dropout(p=0.1)
        self.c10 = nn.Linear(128, 128)
        self.c11 = nn.ReLU()
        self.c12 = nn.Dropout(p=0.1)
        self.c13 = nn.Linear(128, D_out)

    def forward(self, X):
        X = self.c1(X)
        X = self.c2(X)
        X = self.c3(X)
        X = self.c4(X)
        X = self.c5(X)
        X = self.c6(X)
        X = self.c7(X)
        X = self.c8(X)
        X = self.c9(X)
        X = self.c10(X)
        X = self.c11(X)
        X = self.c12(X)
        X = self.c13(X)
        return X