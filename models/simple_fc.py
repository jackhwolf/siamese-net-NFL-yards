import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import no_grad
from torch import from_numpy
from models.util import to_torch

class simple_fc(nn.Module):

    def __init__(self, h1, h2, lr, wd, tr_epochs, **kw):
        super().__init__()
        self.indims = 17
        self.h1 = int(h1) 
        self.h2 = int(h2)
        self.outdims = 1
        self.lr = float(lr)
        self.wd = float(wd)
        self.tr_epochs = int(tr_epochs)
        self.criterion = nn.MSELoss
        self.optimizer = optim.Adam
        self.fc1 = nn.Linear(self.indims, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, self.outdims)

    def learn(self, x, y):
        x, y = to_torch(x), to_torch(y)
        criterion = self.criterion()
        optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.wd)
        closs = 100000
        for i in range(self.tr_epochs):
            print(i, "start")
            pred = self.forward(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            closs = loss.item()
            print(i, closs, "\n=============")
        return closs

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(self.fc1(x))))

    def predict(self, x):
        x = to_torch(x)
        out = None
        with no_grad():
            out = self.forward(x).numpy()
        return out
