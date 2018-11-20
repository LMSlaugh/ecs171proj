import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Importing data
data = pd.read_csv("Dataset/scc_data_to_use.csv", header=0)
data = data.drop(data.columns[0], axis=1) # Remove time of day

column_names = data.columns.values[:-1]
featureData = data.iloc[:, :-1].values # Remove output labels
outputLabels = data.iloc[:,-1].values # Get only output labels

data = data.values

# Isolation forest
clf = IsolationForest(contamination='auto', behaviour='new')

clf.fit(data)
detected_results_IS = clf.predict(data)

outliers_IS = []
for i in range(len(detected_results_IS)):
    if detected_results_IS[i] < 0:
        outliers_IS.append(i)

# LOF
clf = LocalOutlierFactor(n_neighbors=5, contamination='auto')

detected_results_LOF = clf.fit_predict(data)

outliers_LOF = []
for i in range(len(detected_results_LOF)):
    if detected_results_LOF[i] < 0:
        outliers_LOF.append(i)

# use LOF results to remove the outlieres
data = np.delete(data, (outliers_LOF), axis=0)

data = pd.DataFrame(data)

#output = data.to_csv('Dataset/scc_data_to_use_no_outliers.csv', index=False, header=None)
