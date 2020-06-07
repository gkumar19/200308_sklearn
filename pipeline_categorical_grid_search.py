# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 00:32:07 2020

@author: Gaurav
"""
import pandas as pd
import numpy as np
#%% create dataset
x = pd.read_csv('train.csv')
x.isna().sum()
x = x.loc[x['Embarked'].notna(), :]
x = x.drop(['Name', 'Ticket', 'Cabin','PassengerId' ], axis='columns')
mean_age = x['Age'].loc[x['Age'].notna()].mean()
x.fillna({'Age': mean_age}, inplace=True)
y = x.pop('Survived')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

#%%
categorical_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
numerical_columns = [i for i in list(x_train.columns) if i not in categorical_columns]
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ohe = OneHotEncoder(sparse=False)
encoded = ohe.fit_transform(x[categorical_columns])

from sklearn.compose import make_column_transformer

column_transform = make_column_transformer([('ohe',OneHotEncoder(sparse=False), categorical_columns),
                                            ('scale', StandardScaler(), numerical_columns)])


#%%
encoded = column_transform.fit_transform(x_train)
