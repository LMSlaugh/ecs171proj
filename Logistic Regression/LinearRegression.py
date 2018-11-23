import pandas as pd
import numpy as np
import PI_client as pc
import matplotlib
import sklearn as sk
from sklearn import linear_model

# Reading in data
df = pd.read_csv('data/scc_data_to_use_no_outliers.csv')

# Splitting into training and testing sets
train = df.sample(frac=.7).copy()
test = df.drop(train.index).copy()

# Separating features from labels
col = df.columns
col = col[0:len(col)-1]

x_train = pd.DataFrame(index=train.index, data=train[col])
y_train= pd.DataFrame(index=train.index, data=train["Occupancy"])

x_test = pd.DataFrame(index=test.index, data=test[col])
y_test= pd.DataFrame(index=test.index, data=test["Occupancy"])

# Creating and training logistic regression model
mod = linear_model.LogisticRegression(solver="sag",max_iter=3000)
LR_model = mod.fit(x_train,y_train["Occupancy"])

# Extracting accuracy
accuracy = LR_model.score(x_test,y_test)
accuracy

# Converting accuracy to error
error = 1-accuracy
error