import numpy as np
import pandas as pd
from kernel import functions
from sklearn.ensemble import RandomForestClassifier
from chainer import training, iterators, optimizers, serializers, Chain
import chainer.functions as F
import chainer.links as L

from chainer.training import extensions
from chainer.datasets import tuple_dataset

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

dropped = ["PassengeId", "n", "Embarked", "Parch", "SibSp", "Tlen", 'IsAlone']

tot = tot.drop(dropped, axis=1)
print(tot.head())
tot.head()

train = tot[:len(train)]
test = tot[len(train):]
y_train = train['Perished']
X_train = train.drop('Perished', axis=1)
X_test = test.drop('Perished', axis=1)

#学習したモデルを出力するファイル名
resultFn = "sampleNN.model"

#入力層のノードの数
inputNum = len(tot.columns)-1

N = len(train)
train.iloc[:, :-1] /= train.iloc[:, :-1].max()

epoch = 400 #学習回数
batch = 1 #バッチサイズ
hiddens = [inputNum, 800, 400, len(train.iloc[:, inputNum].unique())] #各層のノード数

#学習、検証データの割合(単位：割)
trainSt = 0 #学習用データの開始位置 0割目から〜
trainPro = 8 #学習用データの終了位置　8割目まで
testPro = 10 #検証用データの終了位置 8割目から10割目まで

class MyChain(Chain):

    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(hiddens[0], hiddens[1]),
            l2=L.Linear(hiddens[1], hiddens[2]),
            l3=L.Linear(hiddens[2], hiddens[3]),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        o = self.l3(h2)
        return o


def learning(train):
    # 学習用データと検証用データに分ける
    train_df = train.iloc[0:int(N * trainPro / 10), :]
    test_df = train.iloc[int(N * trainPro / 10):int(N * testPro / 10), :]

    # データの目的変数を落としてnumpy配列にする。
    train_data = np.array(train_df.iloc[:, :-1].astype(np.float32))
    test_data = np.array(test_df.iloc[:, :-1].astype(np.float32))

    # 目的変数もnumpy配列にする。
    train_target = np.array(train_df.iloc[:, inputNum]).astype(np.int32)
    test_target = np.array(test_df.iloc[:, inputNum]).astype(np.int32)

    # ランダムにデータを抽出してバッチ学習する設定
    train = tuple_dataset.TupleDataset(train_data, train_target)
    test = tuple_dataset.TupleDataset(test_data, test_target)
    train_iter = iterators.SerialIterator(train, batch_size=batch, shuffle=True)
    test_iter = iterators.SerialIterator(test, batch_size=batch, repeat=False, shuffle=False)

    # モデルを使う準備。オブジェクトを生成
    model = L.Classifier(MyChain())

    # 最適化手法の設定。今回はAdamを使ったが他にAdaGradやSGDなどがある。
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # 学習データの割り当てを行う
    updater = training.StandardUpdater(train_iter, optimizer)

    # 学習回数を設定してtrainerの構築
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    # trainerの拡張をしておく
    trainer.extend(extensions.Evaluator(test_iter, model))  # 精度の確認
    trainer.extend(extensions.LogReport())  # レポートを残す
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))  # レポートの内容
    trainer.extend(extensions.ProgressBar())  # プログレスバーの表示
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))  # モデルの保存

    # 学習の実行
    trainer.run()

    # モデルの保存
    serializers.save_npz(resultFn, model)

if __name__ == "__main__":
    learning(train)
    print("write to " + str(resultFn))

