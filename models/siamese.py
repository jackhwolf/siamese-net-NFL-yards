import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import no_grad
from torch import from_numpy
from models.util import to_torch, EarlyStopper

offensive_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16]
defensive_cols = [0, 1, 6, 7, 8, 9, 14]
stopper = EarlyStopper()

class siamese(nn.Module):

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
        closs = None
        for i in range(self.tr_epochs):
            print(i, "start")
            pred = self.forward(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            closs = loss.item()
            print(i, closs, "\n=============")
            if stopper(closs):
                break
        return closs

    def forward(self, x):
        fo = self.forward_offense(x)
        fd = self.forward_defense(x)
        out = self.combinatorial_func(fo, fd)
        return out

    def forward_offense(self, x):
        x = x.clone()
        x[:, defensive_cols] = 0
        return self.forward_prop(x)

    def forward_defense(self, x):
        x = x.clone()
        x[:, offensive_cols] = 0
        return self.forward_prop(x)

    def forward_prop(self, x):
        fc1 = F.relu(self.fc1(x))
        fc2 = F.relu(self.fc2(fc1))
        return self.fc3(fc2)

    def combinatorial_func(self, fo, fd):
        return fo - fd

    def predict(self, x):
        x = to_torch(x)
        out = None
        with no_grad():
            out = self.forward(x).numpy()
        return out