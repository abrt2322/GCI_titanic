from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

train = pd.read_csv('../titanic/train.csv')
test = pd.read_csv('../titanic/test.csv')
gender_submission = pd.read_csv('../titanic/gender_submission.csv')

# gender_submission.head()

data = pd.concat([train, test], sort=False)
# data.isnull().sum()

data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].fillna('S', inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

age_avg = data['Age'].mean()
age_std = data['Age'].std()
data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)
y_train = train['Survived']
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

y_preds = []
models = []
oof_train = np.zeros(len((X_train),))
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = ['Embarked', 'Pclass', 'Sex']

params = {
    'objective': 'binary',
    'max_bin': 300,
    'learning_rate': 0.05,
    'num_leaves': 40
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train[train_index]
    y_val = y_train[valid_index]

    print(f'fold_id: {fold_id}')
    print(f'y_tr y==1: {sum(y_tr)/len(y_tr)}')
    print(f'y_val y==1: {sum(y_val)/len(y_val)}')

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_val, y_val,reference=lgb_train,  categorical_feature=categorical_features)

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)
    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_preds.append(y_pred)
    models.append(model)

pd.DataFrame(oof_train).to_csv('../csv/oof_train_kfold.csv', index=False)
scores = [m.best_score['valid_1']['binary_logloss'] for m in models]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)