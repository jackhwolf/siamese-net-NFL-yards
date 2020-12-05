import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OrdinalEncoder
import os

class ProcessedData:

    def __init__(self, data_path='data/data_10p.csv', **kw):
        self.path = data_path
        self.data = pd.read_csv(self.path).dropna()
        column_info = json.loads(open('data/columns.json').read())
        self.feature_encoders = {}
        self.features = column_info['features']
        self.label = column_info['label']
        self.process()
        self.N, self.D = self.data.shape

    def process(self):
        proc = self._drop_cols()
        proc = self._encode_feature(proc, 'OffensePersonnel')
        proc = self._encode_feature(proc, 'OffenseFormation')
        proc = self._encode_feature(proc, 'FieldPosition')
        proc = self._height_to_int(proc)
        proc = proc.astype(np.float32)
        self.data = proc
    
    def _drop_cols(self):
        mat = self.data[self.features + [self.label]]
        return mat

    def _encode_feature(self, mat, feature_column):
        feat = mat[feature_column].to_numpy().reshape(-1, 1)
        enc = OrdinalEncoder()
        enc.fit(feat)
        self.feature_encoders[feature_column] = enc
        mat.loc[:,feature_column] = enc.transform(feat)
        return mat

    def _height_to_int(self, mat):
        heights = mat['PlayerHeight'].to_numpy()
        for idx, _ in enumerate(heights):
            f, i = heights[idx].split('-')
            heights[idx] = 12*int(f) + int(i)
        mat.loc[:,'PlayerHeight'] = heights
        return mat

    def _inverse_transform_features(self):
        mat = self.data.copy()
        for k, v in self.feature_encoders.items():
            mat.loc[:,k] = v.inverse_transform(mat[k].to_numpy().reshape(-1, 1))
        return mat

class ExperimentData(ProcessedData):

    def __init__(self, task='approximation', split='normal', **kw):
        super().__init__(**kw)
        self.task = task
        self.split = split
        self.tr_perc = float(kw.get('train_perc', 0.8))
        self.tr_vc = None
        self.possibly_threshold(**kw)
        self.possibly_bracket(**kw)

    def possibly_threshold(self, **kw):
        if self.task != 'classification':
            return
        self.threshold = float(kw.get('threshold', '5.0'))
        self.data[self.label] = (self.data[self.label] > self.threshold) * 2 - 1

    def possibly_bracket(self, **kw):
        if self.task != 'multiclass':
            return
        brackets = [float(_) for _ in kw.get('brackets', [5, 10, 20])]
        arr = self.data[self.label].values.copy()
        brackets = [np.min(arr)-1] + brackets + [np.max(arr)+1]
        for i in range(len(brackets)-1):
            b = [brackets[i], brackets[i+1]]
            arr[np.where((arr >= b[0]) & (arr < b[1]))[0]] = i
        self.brackets = brackets
        self.data[self.label] = arr

    def train_test_split(self):
        x, y = self.XY()
        smap = {
            'normal': self._normal_mask,
            'balanced': self._balanced_mask,
            'balanced_fillin': self._normal_mask,
        }
        mask = smap[self.split](y)
        trainx, trainy = x[mask], y[mask]
        if self.split == 'balanced_fillin':
            trainx, trainy = self.fillin(trainx, trainy)
        train = (trainx, trainy)
        test = (x[~mask], y[~mask])
        self.tr_vc = np.unique(trainy, return_counts=True)
        return train, test

    def _normal_mask(self, y):
        L = y.shape[0]
        K = int(L * self.tr_perc)
        mask = np.array([False] * L)
        mask[:K] = True
        mask = np.random.permutation(mask)
        return mask

    def _balanced_mask(self, y):
        vc = np.unique(y, return_counts=True)
        L = y.shape[0]
        K = int(self.tr_perc * np.min(vc[1]))
        mask = np.array([False] * L)
        for i in range(len(vc[0])):
            v, c = vc[0][i], vc[1][i]
            where = np.where(y == v)[0]
            where = np.random.permutation(where)[:K]
            mask[where] = True
        return mask

    def XY(self):
        x = self.data[self.features].to_numpy()
        y = self.data[self.label].to_numpy().reshape(-1,1)
        return (x, y)

    def fillin(self, x, y):
        vc = np.unique(y, return_counts=True)
        L = np.max(vc[1])
        mask = np.arange(y.shape[0])
        for i, v in enumerate(vc[0]):
            suppl = L - vc[1][i]
            where = np.where(y == v)[0]
            mask = np.append(mask, np.random.choice(where, suppl))
        return x[mask, :], y[mask]
