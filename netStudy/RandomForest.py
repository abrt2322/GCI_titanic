import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../titanic/train.csv')
test = pd.read_csv('../titanic/test.csv')
sub = pd.read_csv('../titanic/gender_submission.csv')

data = pd.concat([train, test], sort=False)

delete_columns = ['Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], ['Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare'], inplace=True)
data['Title'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Rare'], [1, 2, 3, 4, 5], inplace=True)

delete_columns = ['Name', 'PassengerId']
data.drop(delete_columns, axis=1, inplace=True)

data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
data.loc[data['Age'] <= 16, 'Age'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[(data['Age'] > 64), 'Age'] = 4

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

delete_columns = ['Parch', 'SibSp', 'FamilySize']
data.drop(delete_columns, axis=1, inplace=True)

data['Age'] = data['Age'].astype(int)
data['Age*Class'] = data['Age'] * data['Pclass']

data['Embarked'].fillna('S', inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
data.loc[data['Fare'] > 31, 'Fare'] = 3
data['Fare'] = data['Fare'].astype(int)

# print(data.head())

train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)
y_train = train['Survived']

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)
sub['Survived'] = y_pred
sub.to_csv('../submitCsv/net_randomForest.csv', index=False)
