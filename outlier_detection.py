import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# Importing data
data = pd.read_csv("Dataset/scc_data_to_use_normalized_features_removed.csv", header=0)
data = data.drop(data.columns[0], axis=1) # Remove time of day

column_names = data.columns.values[:-1]
featureData = data.iloc[:, :-1].values # Remove output labels
outputLabels = data.iloc[:,-1].values # Get only output labels

# Isolation forest
clf = IsolationForest(contamination='auto', behaviour='new')

clf.fit(featureData)
detected_results_IS = clf.predict(featureData)

outliers_IS = []
for i in range(len(detected_results_IS)):
    if detected_results_IS[i] < 0:
        outliers_IS.append(i)

# LOF
clf = LocalOutlierFactor(n_neighbors=5, contamination='auto')

detected_results_LOF = clf.fit_predict(featureData)

outliers_LOF = []
for i in range(len(detected_results_LOF)):
    if detected_results_LOF[i] < 0:
        outliers_LOF.append(i)

# use LOF results to remove the outlieres
featureData = np.delete(featureData, (outliers_LOF), axis=0)

featureData = pd.DataFrame(featureData)

output = data.to_csv('Dataset/scc_data_to_use_no_outliers.csv', index=False, header=None)
