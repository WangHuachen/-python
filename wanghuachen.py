# 王哥的编程
# 卡鲁帅的一
# 时间： 2020/12/3 21:51
import pandas as pd
from sklearn.impute import SimpleImputer as Imputer


df = pd.DataFrame([["XXL", 8, "black", "class 1", 22],
                   ["L", np.nan, "gray", "class 2", 20],
                   ["XL", 10, "blue", "class 2", 19],
                   ["M", np.nan, "orange", "class 1", 17],
                   ["M", 11, "green", "class 3", np.nan],
                   ["M", 7, "red", "class 1", 22]])
df.columns = ["size", "price", "color", "class", "boh"]

imr=Imputer(missing_values='NaN',strategy='mean') #allowed_strategies = ["mean", "median", "most_frequent", "constant"]
imr.axis=0
df["price"] = imr.fit_transform(df[["price"]])
df["boh"] = imr.fit_transform(df[["boh"]])
df
























