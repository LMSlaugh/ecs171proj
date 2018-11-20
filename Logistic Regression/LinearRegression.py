import pandas as pd
import numpy as np
import matplotlib
import sklearn as sk
from sklearn import linear_model
import os

# Reading in data, setting index
os.chdir("C:\\Users\\lisam\\Desktop\\Repositories\\ecs171proj")
df = pd.read_csv('Dataset/scc_data_to_use.csv')
df.index = df.iloc[:,0]
df = df.drop(["Unnamed: 0"], axis=1)


# Splitting into training and testing sets
train = df.sample(frac=.7).copy()
test = df.drop(train.index).copy()


# Separating features from labels
col = df.columns
col = col[0:len(col)-1]

x_train = pd.DataFrame(index=train.index, data=train[col])
y_train= pd.DataFrame(index=train.index, data=train["Occupied"])

x_test = pd.DataFrame(index=test.index, data=test[col])
y_test= pd.DataFrame(index=test.index, data=test["Occupied"])


# Creating and training logistic regression model
mod = linear_model.LogisticRegression(solver="sag",max_iter=3000)
LR_model = mod.fit(x_train,y_train["Occupied"])


# Extracting accuracy
accuracy = LR_model.score(x_test,y_test)


# Converting accuracy to error
error = 1-accuracy
