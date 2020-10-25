import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

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
print(data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

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

train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)
y_train = train['Survived']

y_preds = []
models = []
oof_train = np.zeros(len((X_train),))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

params = {
    'objective': 'binary',
    'max_bin': 300,
    'learning_rate': 0.05,
    'num_leaves': 40
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train.loc[train_index]
    y_val = y_train.loc[valid_index]

    print(f'fold_id: {fold_id}')
    print(f'y_tr y==1: {sum(y_tr)/len(y_tr)}')
    print(f'y_val y==1: {sum(y_val)/len(y_val)}')

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)
    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred > 0.5).astype(int)
    y_preds.append(y_pred)
    models.append(model)

y_pred_off = (oof_train > 0.5).astype(int)
len(y_preds)
y_sub = sum(y_preds) /len(y_preds)
y_sub = (y_sub > 0.5).astype(int)

sub['Survived'] = y_sub
sub.to_csv('../submitCsv/net_Sk.csv', index=False)
