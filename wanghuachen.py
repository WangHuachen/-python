# 王哥的编程
# 卡鲁帅的一
# 时间： 2020/12/3 21:51
import pandas as pd
from io import StringIO
import sys
from sklearn.impute import SimpleImputer as Imputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

df = pd.DataFrame([["XXL", 8, "black", "class 1", 22],
                   ["L", np.nan, "gray", "class 2", 20],
                   ["XL", 10, "blue", "class 2", 19],
                   ["M", np.nan, "orange", "class 1", 17],
                   ["M", 11, "green", "class 3", np.nan],
                   ["M", 7, "red", "class 1", 22]])
df.columns = ["size", "price", "color", "class", "boh"]
'''
print(df.values)
print(df.isnull())
print(df.isnull().any())
print(df.isnull().sum()) # 是否缺失 A 0 ,B 0 ,C 0 ,D 0  dtype:int64
df.dropna(axis=0) #删除缺失数据所在的axis=0行所有数据
df.dropna(axis=1) #删除缺失数据所在的axis=1列所有数据
df.dropna(how='all')  #删除行里所有数据缺失的行
df.dropna(thresh=4) # 删除行元素少于4个的行
df.dropna(subset=['C']) #行里C列缺失的行删除
'''

imr=Imputer(missing_values='NaN',strategy='mean') #allowed_strategies = ["mean", "median", "most_frequent", "constant"]
imr.axis=0
df["price"] = imr.fit_transform(df[["price"]])
df["boh"] = imr.fit_transform(df[["boh"]])
df
























