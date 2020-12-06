import yaml
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score
from dataset import ExperimentData
import models
from models.util import to_torch

class Experiment:

    def __init__(self, infile):
        self.params = yaml.load(open(infile))
        self.data = ExperimentData(**self.params)
        m = getattr(__import__(f"models.{self.params['model']}"), self.params['model'])
        self.model = getattr(m, self.params['model'])(**self.params)

    def run(self):
        out = self.params
        res = self.fit_eval()
        out.update(res)
        out['train_vc'] = self.data.tr_vc
        self.save(out)
        return out

    def fit_eval(self):
        train, test = self.data.train_test_split()
        loss = self.model.learn(train[0], train[1])
        preds = self.model.predict(test[0])
        crit = self.model.criterion()
        val_loss = crit(to_torch(preds), to_torch(test[1])).item()
        acc = -1
        try:
            acc = accuracy_score(test[1], preds)
        except:
            pass
        out = {'accuracy': acc, 'training_loss': loss, 'val_loss': val_loss}
        out['val_vc'] = np.unique(test[1], return_counts=True)
        out['val_pred_vc'] = np.unique(preds, return_counts=True)
        return out

    def save(self, out):
        path = 'Results/results.pkl'
        out = pd.DataFrame([out])
        if os.path.exists(path):
            data = pd.read_pickle(path)
            out = pd.concat((data, out))
        else:
            os.makedirs('Results', exist_ok=True)
        out.index = np.arange(out.shape[0])
        out.to_pickle(path)
        return

if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(arg)
        # e = Experiment(arg)
        # print(e.run())