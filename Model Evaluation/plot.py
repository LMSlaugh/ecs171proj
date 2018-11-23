import pandas as pd
import numpy as np
#import matplotlib
import sklearn as sk
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Reading in data, setting index
#df = pd.read_csv('Dataset/scc_data_to_use.csv')
df = pd.read_csv('Dataset/scc_data_to_use_normalized_features_removed.csv')
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

plt.style.use('ggplot')

y_predict_prob = LR_model.predict_proba(x_test)[:,1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_predict_prob)
auc_lr = auc(fpr_lr, tpr_lr)


plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()