import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

net_log = pd.read_csv('../submitCsv/net_logisticRegression.csv')
net_sub = pd.read_csv('../submitCsv/net_randomForest.csv')
sub_ens = pd.read_csv('../submitCsv/net_SVC.csv')
df = pd.DataFrame({
    'sub_hold': net_log['Survived'],
    'sub_sk': net_sub['Survived'],
    'sub_reg': sub_ens['Survived'],
})

print(df.corr())

sub = pd.read_csv('../titanic/gender_submission.csv')

# sub['Survived'] = sub_hold['Survived'] + sub_sk['Survived'] + sub_reg['Survived'] + \
#                   sub_optuna['Survived'] + sub_rFo['Survived']

# sub['Survived'] = sub_hold['Survived'] + sub_sk['Survived'] + \
#                   sub_optuna['Survived'] + sub_rFo['Survived']

sub['Survived'] = net_log['Survived'] + \
                  net_sub['Survived'] + sub_ens['Survived']

sub['Survived'] = (sub['Survived'] >= 2).astype(int)
sub.to_csv('../submitCsv/net_ensemble2.csv', index=False)
