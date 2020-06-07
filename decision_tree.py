# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:48:11 2020

@author: Gaurav
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_moons

#%%
def plot_decision_boundaries(X_train, y_train, clf):
    clf.fit(x,y)
    plt.figure()
    x1 = np.linspace(X_train[:,0].min(), X_train[:,0].max(), num=100)
    x2 = np.linspace(X_train[:,1].min(), X_train[:,1].max(), num=100)
    x1, x2 = np.meshgrid(x1, x2)
    x1 = x1.ravel()[:,None]
    x2 = x2.ravel()[:,None]
    y_pred = clf.predict(np.concatenate([x1, x2], axis=-1))
    plt.tricontourf(x1.ravel(), x2.ravel(), y_pred, cmap='Pastel1')
    plt.scatter(X_train[:,0], X_train[:,1], s=0.1, c=y_train, cmap='jet')


#x = np.array([[1,1], [1.1,1], [1.2,0.8],[2,1], [2,2], [2.1,2.1], [2,1.8]])
#y = np.array([0,0,0,0, 1,1,0])
x, y = make_moons(10000, noise=0.4, random_state=2)

#clf_dt = DecisionTreeClassifier(max_features=2, max_depth=None)
clf_rf = RandomForestClassifier(max_features=2, max_depth=None, min_samples_leaf=10)
#clf_svm = SVC(kernel='rbf')
#clf_v = VotingClassifier([('1', clf_rf), ('svm', clf_svm)] , voting='hard')
clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=500, learning_rate=1)
clf_g = GradientBoostingClassifier()
#plot_decision_boundaries(x, y, clf_dt)
plot_decision_boundaries(x, y, clf_rf)
#plot_decision_boundaries(x, y, clf_svm)
#plot_decision_boundaries(x, y, clf_v)
plot_decision_boundaries(x, y, clf_ada)
plot_decision_boundaries(x, y, clf_g)