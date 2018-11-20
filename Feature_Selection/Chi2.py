
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import sklearn as sk

from sklearn.feature_selection import chi2

# Negative data not valid so have to use normalized data
data = pd.read_csv("../Dataset/scc_data_to_use_normalized.csv", header=0)
data = data.drop(data.columns[0], axis=1) # Remove time of day
data.sample(frac=1) # Shuffle data

column_names = data.columns.values[:-1]
featureData = data.iloc[:, :-1].values # Remove output labels
outputLabels = data.iloc[:,-1].values # Get only output labels

chi2 = chi2(featureData, outputLabels)
ranking = chi2[0] # Chi2 statistics

print("p value of each feature")
for i in range(len(column_names)):
    print(column_names[i] + ": " + "%E" % chi2[1][i]) # p values

# Plot the scores of each feature
plot.bar(column_names, ranking)
plot.title("Scores of features with Chi2 (higher is better)")
plot.xticks(rotation=90)
plot.savefig("chi2.pdf", bbox_inches='tight')
