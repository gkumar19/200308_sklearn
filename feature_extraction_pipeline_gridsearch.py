# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:10:50 2020

@author: KGU2BAN
"""
#%% clean data and split into train and test
import pandas as pd
from sklearn.model_selection import train_test_split
x = pd.read_csv('train.csv')
x.isna().sum()
mean_age = x['Age'].mean()
x.fillna({'Age': mean_age}, inplace=True)
x.isna().sum()
x = x.loc[x['Embarked'].notna(),:]
x.isna().sum()
x.dtypes
x['PassengerId'] = x['PassengerId'].astype('float64')
x.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
y = x.pop('Survived')
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
cat_column = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
num_column = ['PassengerId', 'Age', 'Fare']


#%% check the preprocessing step , not used ahead
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ohe = OneHotEncoder(sparse=False).fit(x_train[cat_column])
oh_encoded = ohe.transform(x_train[cat_column])
ohe.categories_

scale = StandardScaler().fit(x_train[num_column])
scale_encoded = scale.transform(x_train[num_column])

#%% create column transformer, not used ahead and started fresh
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('ohe', OneHotEncoder(sparse=False), cat_column),
                        ('scale',StandardScaler(), num_column)], remainder='drop')
ct.fit(x_train)
encoded = ct.transform(x_train)

#%% create dataflow and model pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
ct = ColumnTransformer([('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_column),
                        ('scale',StandardScaler(), num_column)], remainder='drop')
pipeline = Pipeline([('column_transform', ct), ('model', SVC(C=1.5, kernel='rbf'))])

pipeline.fit(x_train, y_train)
pipeline.score(x_test, y_test)

cross_val_score(pipeline, x_train, y_train, cv=5)

#%% gridsearch for the parameters
from sklearn.model_selection import GridSearchCV
para_grid = {'model__C': [0.1, 1, 10, 100],
             'model__kernel': ['linear', 'rbf'],
             'model__gamma': ['auto', 'scale']}
grid = GridSearchCV(pipeline, param_grid=para_grid, cv=3).fit(x_train, y_train)
grid_result = pd.DataFrame(grid.cv_results_)
best_params = grid.best_params_
pipeline = Pipeline([('column_transform', ct), ('model', SVC(C=10, kernel='rbf', gamma='auto'))]).fit(x_train, y_train)

pipeline.score(x_test, y_test)


#%% Gradient Boosting and Extra tree regression 1% improvement
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
gb = GradientBoostingClassifier()
gb = ExtraTreesClassifier()
pipeline = Pipeline([('column_transform', ct), ('model', gb)]).fit(x_train, y_train)

para_grid = {'model__n_estimators': [10, 100, 1000],
             'model__min_samples_leaf': [1, 2],
             'model__max_depth': [3, 6]}
grid = GridSearchCV(pipeline, param_grid=para_grid, cv=3).fit(x_train, y_train)
grid_result = pd.DataFrame(grid.cv_results_)

grid.best_score_

#%% Adaboost: 82.7% accurate
from sklearn.ensemble import AdaBoostClassifier

ct = ColumnTransformer([('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_column),
                        ('scale',StandardScaler(), num_column)], remainder='drop')
pipeline = Pipeline([('column_transform', ct), ('model', SVC(C=1.5, kernel='rbf'))])

gb = AdaBoostClassifier(ExtraTreesClassifier(n_estimators=10, max_depth=6))

para_grid = {'model__n_estimators': [10, 100],
             'model__learning_rate': [0.1, 1, 10]}
grid = GridSearchCV(pipeline, param_grid=para_grid, cv=3).fit(x_train, y_train)
grid_result = pd.DataFrame(grid.cv_results_)

grid.best_score_

#%%kaggle submission:
kaggle_test = pd.read_csv('test.csv')
def extraxt_features(x):
    x.isna().sum()
    mean_age = x['Age'].mean()
    x.fillna({'Age': mean_age}, inplace=True)
    mean_fare = x['Fare'].mean()
    x.fillna({'Fare': mean_fare}, inplace=True)
    x.isna().sum()
    x = x.loc[x['Embarked'].notna(),:]
    x.isna().sum()
    x.dtypes
    x['PassengerId'] = x['PassengerId'].astype('float64')
    x.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    return x

kaggle_test = extraxt_features(kaggle_test)

kaggle_predict = grid.predict(kaggle_test)
kaggle_predict = pd.DataFrame(kaggle_predict, columns=['Survived'])
kaggle_submit = pd.concat([kaggle_test[['PassengerId']], kaggle_predict], axis=1).astype('int32')
kaggle_submit.to_csv('kaggle_submit.csv', index=False)
