import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('project/Autism_Prediction/train.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df['ethnicity'].value_counts())
print(df['relation'].value_counts())
df = df.replace({'yes':1, 'no':0, '?':'Others', 'others':'Others'})
most_frequent = df['ethnicity'].mode()[0]
df['ethnicity'] = df['ethnicity'].replace('?', most_frequent)
a=df['ethnicity'].head
print(a)
