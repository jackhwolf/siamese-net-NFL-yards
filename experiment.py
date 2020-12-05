import yaml
from sklearn.metrics import accuracy_score
from dataset import ExperimentData
import models

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
        return out

    def fit_eval(self):
        train, test = self.data.train_test_split()
        loss = self.model.learn(train[0], train[1])
        preds = self.model.predict(test[0])
        acc = -1
        try:
            acc = accuracy_score(test[1], preds)
        except:
            pass
        out = {'accuracy': acc, 'loss': loss}
        return out

if __name__ == "__main__":
    import sys
    e = Experiment(sys.argv[1])
    print(e.run())