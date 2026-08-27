import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


#load the dataset
df=pd.read_csv("D:\\mlops_day1\\data\\data.csv", index_col=0)

#train-test-split
X,y = df.drop(columns=["sales"]), df["sales"]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=67)


#linear regression model
model= LinearRegression()
model.fit(xtrain, ytrain)

#Model dump
joblib.dump(model, "D:\\mlops_day1\\models\\linear_reg_model.pkl")
