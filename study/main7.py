import optuna
from sklearn.metrics import log_loss
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../titanic/train.csv')
test = pd.read_csv('../titanic/test.csv')
sub = pd.read_csv('../titanic/gender_submission.csv')

data = pd.concat([train, test], sort=False)

# print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# g = sns.FacetGrid(train, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()

# grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()

grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

# plt.show()

#
#
delete_columns = ['Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# print(pd.crosstab(data['Title'], data['Sex']))

data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], ['Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare'], inplace=True)
data['Title'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Rare'], [1, 2, 3, 4, 5], inplace=True)
# print(pd.crosstab(data['Title'], data['Sex']))
# print(data.head())

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

# print(data.head())

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
# print(data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

data['IsAlone'] = 0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
print(data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

delete_columns = ['Parch', 'SibSp', 'FamilySize']
data.drop(delete_columns, axis=1, inplace=True)

# data['Age'] = data['Age'].astype(int)
data['Age*Class'] = data['Age'] * data['Pclass']

data['Embarked'].fillna('S', inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
data.loc[data['Fare'] > 31, 'Fare'] = 3
data['Fare'] = data['Fare'].astype(int)

print(data.head())


train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)
y_train = train['Survived']


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

def objective(trial):
    params = {
        'objective': 'binary',
        'max_bin': trial.suggest_int('max_bin', 255, 500),
        'learning_rate': 0.05,
        'num_leaves': trial.suggest_int('num_leaves', 32, 128),
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    score = log_loss(y_valid, y_pred_valid)
    return score

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=40)

params = {
    'objective': 'binary',
    'max_bin': study.best_params['max_bin'],
    'learning_rate': 0.05,
    'num_leaves': study.best_params['num_leaves'],
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

y_pred = (y_pred > 0.5).astype(int)
sub['Survived'] = y_pred
sub.to_csv('submission_net1.csv', index=False)