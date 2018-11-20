
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import sklearn as sk

from sklearn.svm import SVC
from sklearn.feature_selection import RFE

data = pd.read_csv("../Dataset/scc_data_to_use.csv", header=0)
data = data.drop(data.columns[0], axis=1) # Remove time of day
data.sample(frac=1) # Shuffle data

column_names = data.columns.values[:-1]
featureData = data.iloc[:, :-1].values # Remove output labels
outputLabels = data.iloc[:,-1].values # Get only output labels

# Only use 3000 samples, otherwise takes too long
featureData = featureData[0:3000]
outputLabels = outputLabels[0:3000]

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, step=1, verbose=1, n_features_to_select=1)
rfe.fit(featureData, outputLabels)
ranking = rfe.ranking_

# Plot each feature rank
# Rank 1 means selected and is best
plot.bar(column_names, ranking)
plot.title("Ranking of features with RFE (lower is better)")
plot.xticks(rotation=90)
plot.savefig("linear_svc.pdf", bbox_inches='tight')
