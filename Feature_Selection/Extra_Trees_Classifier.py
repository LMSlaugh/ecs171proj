
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import sklearn as sk

from sklearn.ensemble import ExtraTreesClassifier

for i in range(2):
    if i == 0:
        dataset = "../Dataset/scc_data_to_use.csv"
        output_name = "ExtraTreesClassifier (original data).pdf"
    if i == 1:
        dataset = "../Dataset/scc_data_to_use_normalized.csv"
        output_name = "ExtraTreesClassifier (normalized data).pdf"

    data = pd.read_csv(dataset, header=0)
    data = data.drop(data.columns[0], axis=1) # Remove time of day
    data.sample(frac=1) # Shuffle data

    column_names = data.columns.values[:-1]
    featureData = data.iloc[:, :-1].values # Remove output labels
    outputLabels = data.iloc[:,-1].values # Get only output labels

    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(featureData, outputLabels)
    ranking = model.feature_importances_

    # Plot the scores of each feature
    plot.bar(column_names, ranking)
    plot.title("Scores of features with ExtraTreesClassifier (higher is better)")
    plot.xticks(rotation=90)
    plot.savefig(output_name, bbox_inches='tight')
    plot.clf()
