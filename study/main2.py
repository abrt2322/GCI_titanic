import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)

# clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

sub = pd.read_csv('../titanic/gender_submission.csv')
# sub['Survived'] = list(map(int, y_pred))
# sub.to_csv('submission.csv', index=False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
params = {
    'objective': 'binary'
}
model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred > 0.5).astype(int)
sub['Survived'] = y_pred
sub.to_csv('submission_lightgbm_holdout.csv', index=False)



