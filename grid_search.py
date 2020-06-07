# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:16:45 2020

@author: Gaurav
"""

import seaborn as sns
from sklearn.datasets import make_gaussian_quantiles, make_blobs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x, y = make_blobs(n_samples=2000, centers=[[1, 2],
                                           [3, 4],
                                           [5, 6],
                                           [-1, 10]],
                  cluster_std=[0.2, 0.3, 0.5, 0.1],
                  n_features=2,random_state=0)
x = np.dot(x, [[1, -3],
               [-2, 1]]) #covariance

sns.scatterplot(x[:,0], x[:,1],color='red')

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(SVC(), param_grid = {'C':[0.1, 1, 100], 'gamma': [0.001, 1, 10]}).fit(x,y)
clf.best_params_
clf.best_estimator_
clf.best_score_
clf.best_index_

df = pd.DataFrame(clf.cv_results_)

def plot_decision_boundary(clf, x_train, y_train):
    x1_min, x1_max = x_train[:,0].min(), x_train[:,0].max()
    x2_min, x2_max = x_train[:,1].min(), x_train[:,1].max()
    x1 = np.linspace(x1_min, x1_max, 100)
    x2 = np.linspace(x2_min, x2_max, 100)
    x1, x2 = np.meshgrid(x1, x2)
    x1, x2 = x1.ravel(), x2.ravel()
    y_pred = clf.predict(np.concatenate([x1[:,None], x2[:,None]], axis=-1))
    plt.tricontourf(x1, x2, y_pred)
    sns.scatterplot(x_train[:,0], x_train[:,1], hue=y_train)

plt.figure()
plot_decision_boundary(clf, x, y)