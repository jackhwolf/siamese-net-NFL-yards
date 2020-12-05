import sys
sys.path.append('../')
from dataset import ExperimentData
import matplotlib.pyplot as plt 
import numpy as np

fnames = [
    'raw',
    'binary',
    'binary-balanced',
    'binary-balanced-filled',
    'multi',
    'multi-balanced',
    'multi-balanced-filled',
]

param_descs = [
    ['Approximation Data', 'Randomly sampled training data'],
    ['Binary classification with Threshold=5', 'Randomly sampled training data'],
    ['Binary classification with Threshold=5', 'Class balanced training data'],
    ['Binary classification with Threshold=5', 'Class balanced augmented training data'],
    ['Multiclass classification with Brackets=[5, 10, 20]', 'Randomly sampled training data'],
    ['Multiclass classification with Brackets=[5, 10, 20]', 'Class balanced training data'],
    ['Multiclass classification with Brackets=[5, 10, 20]', 'Class balanced augmented training data'],
]

params = [
    {},
    {'task': 'classification'},
    {'task': 'classification', 'split': 'balanced'},
    {'task': 'classification', 'split': 'balanced_fillin'},
    {'task': 'multiclass'},
    {'task': 'multiclass', 'split': 'balanced'},
    {'task': 'multiclass', 'split': 'balanced_fillin'},
]

L = len(param_descs)

def plot_raw(desc, data, fname):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.set_title(desc[0])
    data.train_test_split()
    _, y = data.XY()
    all_vc = np.unique(y, return_counts=True)
    ax.bar(all_vc[0], all_vc[1])
    if data.task != 'approximation':
        ax.set_xticks(all_vc[0])
        ax.set_xticklabels([int(_) for _ in all_vc[0]])
    fig.savefig('./Presentation/' + fname + '-data.png', bbox_inches='tight')

def plot_training(desc, data, fname):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.set_title(desc[1])
    data.train_test_split()
    _, y = data.XY()
    tr_vc = data.tr_vc
    ax.bar(tr_vc[0], tr_vc[1])
    if data.task != 'approximation':
        ax.set_xticks(tr_vc[0])
        ax.set_xticklabels(tr_vc[1])
        ax.set_yticks(tr_vc[1])
        ax.set_yticklabels(tr_vc[1])
        tr_percs = tr_vc[1]
        tr_percs = tr_percs / np.sum(tr_vc[1])
        for i, tp in enumerate(tr_percs):
            ax.text(tr_vc[0][i]-0.3, tr_vc[1][i], str(np.round(tp * 100, 2)) + '%')
    fig.savefig('./Presentation/' + fname + '-training.png', bbox_inches='tight')

for i in range(L):
    desc = param_descs[i]
    data = ExperimentData(**params[i])
    fname = fnames[i]
    plot_raw(desc, data, fname)
    plot_training(desc, data, fname)