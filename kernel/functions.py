import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns

def check_survive(tot, label):
    return tot[["Survived", label]].groupby(label).Survived.mean()

def check_survive2(tot, label):
    return tot[["Perished", label]].groupby(label).Perished.mean()

def ageof(x):
    if x < 10: return 0
    if x < 25 : return 1
    if x < 35 : return 2
    return 3

def fareof(x):
    if x < 7.5: return 0
    if x < 14 : return 1
    if x < 50 : return 2
    if x < 80 : return 3
    return 4
