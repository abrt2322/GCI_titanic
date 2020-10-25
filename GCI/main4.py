import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from kernel import functions
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
train["n"] = 0
test["n"] = 1

tot = pd.concat([train, test], sort=False)

tot.Sex = tot.Sex.replace({"male": 0, "female": 1})
tot["Title"] = tot.Name.str.extract("([A-Za-z]+)\.")
tot["Title"].replace(['Lady', 'Countess', 'Sir', 'Jonkheer', 'Dona'], 'Upper', inplace=True)
tot["Title"].replace(['Don', 'Rev', 'Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
tot.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}, inplace=True)

temp = functions.check_survive2(tot, "Title").sort_values()
temp.iloc[:] = np.arange(len(temp))
dic = temp.to_dict()
tot.Title.replace(dic, inplace=True)

tot["FamilySize"] = tot.SibSp + tot.Parch + 1
tot.FamilySize = tot.FamilySize.map(lambda x: 2 if 1 < x < 5 else 1 if x == 1 else 0)
tot['IsAlone'] = 0
tot.loc[tot['FamilySize'] == 0, 'IsAlone'] = 1

tot.Embarked.fillna("S", inplace=True)
tot.Embarked = tot.Embarked.replace({"S": 0, "Q": 1, "C": 2})

tot.Fare.fillna(0, inplace=True)
tot.Fare[tot.Fare.notna()] = tot.Fare[tot.Fare.notna()].map(lambda x: functions.fareof(x))

tot.Age[tot.Age.notna()] = tot.Age[tot.Age.notna()].map(lambda x: functions.ageof(x))
dic = tot[tot.Age.notna()][["Age", "Title"]].groupby("Title").Age.mean().to_dict()
tot.Age[tot.Age.isna()] = tot.Title[tot.Age.isna()].replace(dic).map(lambda x: int(x))
tot['Age'] = tot['Age'].astype(int)
tot['Age*Class'] = tot['Age'] * tot['Pclass']

tot["RT"] = tot.Ticket.map(lambda x: x.split(" ")[0] if len(x.split(" "))==1 else x.split(" ")[1])
tot["LT"] = tot.Ticket.map(lambda x: str(x.split(" ")[0])[0] if len(x.split(" ")) > 1 else np.nan)
tot["Tlen"] = tot.RT.map(lambda x: len(x))
tot["RT"] = tot.RT.map(lambda x: str(x)[0])

lnames = tot.Name.map(lambda x: x.split(",")[0])
tot.Name = lnames
tnum = tot.Ticket.map(lambda x: x.split(" ")[-1])
tot.Ticket = tnum
tot["FamSize"] = tot.SibSp + tot.Parch
nlist = tot.Name.value_counts().index

tot["FamDeath"] = np.nan

for i in range(len(tot)):
    if tot.iloc[i, :].FamSize > 0:
        hisname = tot.iloc[i, :].Name
        hisfam = tot.iloc[i, :].FamSize
        temp = pd.concat([tot.iloc[:i, :], tot.iloc[i + 1:, :]])
        family = temp[(temp.Name == hisname) * (temp.FamSize == hisfam)]
        if len(family) == 0:
            continue
        tot.FamDeath.iloc[i] = family.Perished.mean()

tot.FamDeath.fillna(0.5, inplace=True)

del tot["Ticket"], tot["Cabin"], tot["RT"], tot["LT"]
del tot["FamSize"], tot["Name"]

print(pd.crosstab(tot['FamilySize'], tot['Sex']))
print(tot.head(10))
print(tot.columns)

dropped = ["n", "PassengerId", "Embarked", "Parch", "SibSp", "Tlen", 'IsAlone']

tot = tot.drop(dropped, axis=1)

print(tot.head())
