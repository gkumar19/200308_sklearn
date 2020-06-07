# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:21:47 2020

@author: Gaurav
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits ,make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
scale = StandardScaler().fit(x_train)
x_train, x_test = scale.transform(x_train), scale.transform(x_test)

clf_log = LogisticRegression().fit(x_train, y_train)
clf_svm = SVC(probability=True, C=10).fit(x_train, y_train)
clf_rf = RandomForestClassifier().fit(x_train, y_train)

print(clf_log.score(x_test, y_test))
print(clf_svm.score(x_test, y_test))
print(clf_rf.score(x_test, y_test))


#voting
clf_vote = VotingClassifier([('log', clf_log),
                             ('rf', clf_rf)], 'hard').fit(x_train, y_train)
print(clf_vote.score(x_test, y_test))

#%%Bagging
x, y = make_moons(1000, noise=0.3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
scale = StandardScaler().fit(x_train)
x_train, x_test = scale.transform(x_train), scale.transform(x_test)
sns.scatterplot(x_train[:,0], x_train[:,1], hue = y_train)

clf_dt = DecisionTreeClassifier().fit(x_train, y_train)
print(clf_dt.score(x_test, y_test))

clf_bag = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=200,
                            max_samples=0.5).fit(x_train, y_train)
BaggingClassifier()
print(clf_bag.score(x_test, y_test))

clf_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1)).fit(x_train, y_train)
print(clf_ada.score(x_test, y_test))

clf_gra = GradientBoostingClassifier(n_estimators=200).fit(x_train, y_train)
print(clf_gra.score(x_test, y_test))
staged_pred = clf_gra.staged_predict(x_test)
staged_errors = [accuracy_score(y_test, y_pred) for y_pred in staged_pred]
plt.plot(staged_errors)

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

plot_decision_boundary(clf_dt, x_train, y_train)
plt.figure()
plot_decision_boundary(clf_bag, x_train, y_train)
plt.figure()
plot_decision_boundary(clf_ada, x_train, y_train)
plt.figure()
plot_decision_boundary(clf_gra, x_train, y_train)
