import numpy as np
import pandas as pd
from feature_engineering import one_hot_encoding

# from warnings import ignore

dataset = pd.read_csv("melb_data.csv")

print(dataset.head())

cols_to_remove = ["Unnamed: 0","Address","Date"]

dataset.drop(cols_to_remove, axis=1, inplace=True)

print(dataset.head())

# dataset.to_csv("sample.csv")

### Handling Null Values

cols_to_fill_with_zero = ["Bedroom2","Bathroom","Car"]

dataset[cols_to_fill_with_zero] = dataset[cols_to_fill_with_zero].fillna(0)

# cols_to_fill_with_mean = ["Landsize","BuildingArea","YearBuilt","CouncilArea","Lattitude","Longtitude"]

dataset = dataset.fillna(dataset.mean())

dataset.drop("CouncilArea",axis=1,inplace=True)

# print(dataset.isnull().sum())
# print(dataset.shape)

##### One Hot Encoding

# one_hot_encoding(dataset)

# dataset.to_csv("sample.csv")

dataset = pd.read_csv("preprocessed_data.csv")

print(dataset.head())

# Remove Dummy Variables

# dataset = pd.get_dummies(dataset, drop_first=True)

# print(dataset.head())

X = dataset.drop("Price",axis=1)
y = dataset["Price"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=2)

# from sklearn.linear_model import LinearRegression

# reg = LinearRegression().fit(X_train, y_train)

# print(reg.score(X_train, y_train))
# print(reg.score(X_test, y_test))

# Lasso Regression
from sklearn import linear_model

# lasso_reg = linear_model.Lasso(alpha=50,max_iter=100,tol=0.1)
# lasso_reg.fit(X_train,y_train)

# print(lasso_reg.score(X_train, y_train))
# print(lasso_reg.score(X_test, y_test))

ridge_reg = linear_model.Ridge(alpha=0.1,max_iter=200,tol=0.1)
ridge_reg.fit(X_train,y_train)

print(ridge_reg.score(X_train, y_train))
print(ridge_reg.score(X_test, y_test))
