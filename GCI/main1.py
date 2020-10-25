import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import log_loss
import lightgbm as lgb
import numpy as np

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
sub = pd.read_csv('../data/gender_submission.csv')

tot = pd.concat([train, test], sort=False)

tot.Sex = tot.Sex.replace({"male": 0, "female": 1})
tot["Title"] = tot.Name.str.extract("([A-Za-z]+)\.")
tot['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Ms', 'Mme', 'Mlle'], ['Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare', 'Rare'], inplace=True)
tot['Title'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Rare'], [1, 2, 3, 4, 5], inplace=True)

tot.Embarked.fillna("S", inplace=True)
embarked = tot['Embarked']
embarked_ohe = pd.get_dummies(embarked)
tot = pd.concat([tot, embarked_ohe], axis=1)

age_avg = tot['Age'].mean()
age_std = tot['Age'].std()
tot['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
tot.loc[tot['Age'] <= 16, 'Age'] = 0
tot.loc[(tot['Age'] > 16) & (tot['Age'] <= 32), 'Age'] = 1
tot.loc[(tot['Age'] > 32) & (tot['Age'] <= 48), 'Age'] = 2
tot.loc[(tot['Age'] > 48) & (tot['Age'] <= 64), 'Age'] = 3
tot.loc[(tot['Age'] > 64), 'Age'] = 4

tot['FamilySize'] = tot['SibSp'] + tot['Parch'] + 1
tot['IsAlone'] = 0
tot.loc[tot['FamilySize'] == 1, 'IsAlone'] = 1

tot['Fare'].fillna(np.mean(tot['Fare']), inplace=True)
tot.loc[tot['Fare'] <= 7.91, 'Fare'] = 0
tot.loc[(tot['Fare'] > 7.91) & (tot['Fare'] <= 14.454), 'Fare'] = 1
tot.loc[(tot['Fare'] > 14.454) & (tot['Fare'] <= 31), 'Fare'] = 2
tot.loc[tot['Fare'] > 31, 'Fare'] = 3
tot['Fare'] = tot['Fare'].astype(int)


delete_columns = ['Parch', 'SibSp', 'FamilySize', 'Fare', 'Embarked', 'Name', 'PassengerId', 'Ticket', 'Cabin']
tot.drop(delete_columns, axis=1, inplace=True)

print(tot.head())

train = tot[:len(train)]
test = tot[len(train):]
X_train = train.drop('Perished', axis=1)
X_test = test.drop('Perished', axis=1)
y_train = train['Perished']

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
sub['Perished'] = y_pred
sub.to_csv('./submission_net1.csv', index=False)
