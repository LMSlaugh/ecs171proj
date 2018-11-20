
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import sklearn as sk

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Negative data not valid so have to use normalized data
data = pd.read_csv("../Dataset/scc_data_to_use_normalized.csv", header=0)
data = data.drop(data.columns[0], axis=1) # Remove time of day
data.sample(frac=1) # Shuffle data

column_names = data.columns.values[:-1]
featureData = data.iloc[:, :-1].values # Remove output labels
outputLabels = data.iloc[:,-1].values # Get only output labels

kbest = SelectKBest(score_func=chi2)
kbest.fit(featureData, outputLabels)
ranking = kbest.scores_

# Plot the scores of each feature
plot.bar(column_names, ranking)
plot.title("Scores of features with Chi2 (higher is better)")
plot.xticks(rotation=90)
plot.savefig("chi2.pdf", bbox_inches='tight')
