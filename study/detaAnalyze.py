import pandas as pd
import pandas_profiling as pdf
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../titanic/train.csv')

# plt.xlabel('Age')
# plt.ylabel('count')

# plt.hist(train.loc[train['Survived'] == 1, 'Age'].dropna(), bins=30, alpha=0.5, label='1')
# plt.legend(title='Survived')

# sns.countplot(x='SibSp', hue='Survived', data=train)
# plt.legend(loc='upper right', title='Survived')

# sns.countplot(x='Parch', hue='Survived', data=train)
# plt.legend(loc='upper right', title='Survived')

# plt.hist(train.loc[train['Survived'] == 0, 'Fare'].dropna(), range=(0, 250), bins=25, alpha=0.5, label=0)
# plt.hist(train.loc[train['Survived'] == 1, 'Fare'].dropna(), range=(0, 25), bins=25, alpha=0.5, label=1)
# plt.xlabel('Fare')
# plt.ylabel('count')
# plt.legend(title='Survived')
# plt.xlim(-5, 250)

# sns.countplot(x='Pclass', hue='Survived', data=train)
# sns.countplot(x='Sex', hue='Survived', data=train)
# sns.countplot(x='Embarked', hue='Survived', data=train)

test = pd.read_csv('../titanic/test.csv')
data = pd.concat([train, test], sort=False)
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
train['FamilySize'] = data['FamilySize'][:len(train)]
test['FamilySize'] = data['FamilySize'][len(train):]
sns.countplot(x='FamilySize', hue='Survived', data=train)

plt.show()


