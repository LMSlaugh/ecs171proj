import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Importing data
data = pd.read_csv("Dataset/scc_data_to_use_normalized_features_removed.csv", header=0)
data = data.drop(data.columns[0], axis=1) # Remove time of day

column_names = data.columns.values[:-1]
featureData = data.iloc[:, :-1].values # Remove output labels
outputLabels = data.iloc[:,-1].values # Get only output labels

# Isolation forest
IF = IsolationForest(contamination='auto', behaviour='new')

detected_results_IS = IF.fit_predict(featureData)

outliers_IS = []
for i in range(len(detected_results_IS)):
    if detected_results_IS[i] < 0:
        outliers_IS.append(i)

IS_data = np.delete(featureData, (outliers_IS), axis=0)
output_IS = np.delete(outputLabels, (outliers_IS), axis=0)

# LOF
LOF = LocalOutlierFactor(n_neighbors=20, contamination='auto')

detected_results_LOF = LOF.fit_predict(featureData)

outliers_LOF = []
for i in range(len(detected_results_LOF)):
    if detected_results_LOF[i] < 0:
        outliers_LOF.append(i)

LOF_data = np.delete(featureData, (outliers_LOF), axis=0)
output_LOF = np.delete(outputLabels, (outliers_LOF), axis=0)

# SVM
SVM = svm.OneClassSVM(gamma='auto')

detected_results_SVM = SVM.fit_predict(featureData)

outliers_SVM = []
for i in range(len(detected_results_SVM)):
    if detected_results_SVM[i] < 0:
        outliers_SVM.append(i)

SVM_data = np.delete(featureData, (outliers_SVM), axis=0)
output_SVM = np.delete(outputLabels, (outliers_SVM), axis=0)

# Elliptic envelope
EE = EllipticEnvelope()

detected_results_EE = EE.fit_predict(featureData)

outliers_EE = []
for i in range(len(detected_results_EE)):
    if detected_results_EE[i] < 0:
        outliers_EE.append(i)

EE_data = np.delete(featureData, (outliers_EE), axis=0)
output_EE = np.delete(outputLabels, (outliers_EE), axis=0)

# Compare the outlier results
accuracy_scores = {}

model0 = svm.SVC()
model0.fit(IS_data[0:int(len(IS_data)/2)], output_IS[0:int(len(IS_data)/2)])
predicted_labels0 = model0.predict(IS_data[int(len(IS_data)/2):])
accuracy_scores["outliers_IS"] = accuracy_score(output_IS[int(len(IS_data)/2):], predicted_labels0)

model1 = svm.SVC()
model1.fit(LOF_data[0:int(len(LOF_data)/2)], output_LOF[0:int(len(LOF_data)/2)])
predicted_labels1 = model1.predict(LOF_data[int(len(LOF_data)/2):])
accuracy_scores["outliers_LOF"] = accuracy_score(output_LOF[int(len(LOF_data)/2):], predicted_labels1)

model2 = svm.SVC()
model2.fit(SVM_data[0:int(len(SVM_data)/2)], output_SVM[0:int(len(SVM_data)/2)])
predicted_labels2 = model2.predict(SVM_data[int(len(SVM_data)/2):])
accuracy_scores["outliers_SVM"] = accuracy_score(output_SVM[int(len(SVM_data)/2):], predicted_labels2)

model3 = svm.SVC()
model3.fit(EE_data[0:int(len(EE_data)/2)], output_EE[0:int(len(EE_data)/2)])
predicted_labels3 = model3.predict(EE_data[int(len(EE_data)/2):])
accuracy_scores["outliers_EE"] = accuracy_score(output_EE[int(len(EE_data)/2):], predicted_labels3)

result = max(accuracy_scores, key=accuracy_scores.get)

# remove outliers
featureData = np.delete(featureData, eval(result), axis=0)

featureData = pd.DataFrame(featureData, columns=[column_names])

output = featureData.to_csv('Dataset/scc_data_to_use_no_outliers.csv', index=False)
