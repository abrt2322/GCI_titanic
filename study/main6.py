import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sub_hold = pd.read_csv('../submitCsv/submission_lightgbm_holdout.csv')
sub_sk = pd.read_csv('../submitCsv/submission_lightgbmSkfold.csv')
sub_reg = pd.read_csv('../submitCsv/submission_logistigRegression.csv')
sub_optuna = pd.read_csv('../submitCsv/submission_optuna.csv')
sub_rFo = pd.read_csv('../submitCsv/submission_randomForest.csv')

df = pd.DataFrame({
    'sub_hold': sub_hold['Survived'],
    'sub_sk': sub_sk['Survived'],
    'sub_reg': sub_reg['Survived'],
    'sub_optuna': sub_optuna['Survived'],
    'sub_rFo': sub_rFo['Survived']
})

print(df.corr())

sub = pd.read_csv('../titanic/gender_submission.csv')

# sub['Survived'] = sub_hold['Survived'] + sub_sk['Survived'] + sub_reg['Survived'] + \
#                   sub_optuna['Survived'] + sub_rFo['Survived']

# sub['Survived'] = sub_hold['Survived'] + sub_sk['Survived'] + \
#                   sub_optuna['Survived'] + sub_rFo['Survived']

sub['Survived'] = sub_sk['Survived'] + \
                  sub_optuna['Survived'] + sub_rFo['Survived']

sub['Survived'] = (sub['Survived'] >= 2).astype(int)
sub.to_csv('../submitCsv/submission_ensemble2.csv', index=False)
