from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import pandas
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np


#reads data from file
def read_data():
    #return pandas.read_csv('scc_data_to_use_normalized.csv', index_col = 0, parse_dates=True, infer_datetime_format=True)
    return pandas.read_csv('Dataset/scc_data_to_use_no_outliers.csv', index_col = 0, parse_dates=True, infer_datetime_format=True)


#collect all option permutations to use easily later on
class ANN_Options:
    def __init__(self, activation_function, num_nodes_per_hidden, num_hidden_layers, batch_size, learning_rate):
        self.activation_func = activation_function
        self.nodes_per_hidden = num_nodes_per_hidden
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    def __str__(self):
        return str(self.activation_func) + " | hid nodes: " + str(self.nodes_per_hidden) + " | hid layers: " + str(self.num_hidden_layers) + " | batch size: " + str(self.batch_size) + " | learning rate: " + str(self.learning_rate)
    def writeToFile(self):
        return "ANN_Options(" + str(self.activation_func) + "|" + str(self.nodes_per_hidden) + "|" + str(self.num_hidden_layers) + "|" + str(self.batch_size) + "|" + str(self.learning_rate) + ")"

"""
    main function to build the ann
    X_TRAIN, Y_TRAIN = testing data
    X_TEST, Y_TEST   = testing data
    ann_options      = an instance of ANN_Options to specify the options for the ann
"""
def BuildAnn(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, ann_options, verbose=True):
    model = Sequential()
    #input layer?
    model.add(Dense(ann_options.nodes_per_hidden, input_dim=X_TRAIN.shape[1], activation=ann_options.activation_func, use_bias=True))
    #hidden layers
    for n in range(0, ann_options.num_hidden_layers):
        model.add(Dense(ann_options.nodes_per_hidden, activation=ann_options.activation_func, use_bias=True))
    #output layer
    model.add(Dense(1,activation='sigmoid'))
    #stochastic gradient descent:
    optimizerSGD = SGD(lr=ann_options.learning_rate, momentum=0.9, decay=0.0, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=optimizerSGD, metrics=['accuracy'])
    #model.compile(optimizer= 'adam', loss ="sparse_categorical_crossentropy", metrics = ['accuracy'])
    num_epochs = 30
    if (ann_options.learning_rate < 0.005):
        num_epochs = 100

    hist = model.fit(X_TRAIN, Y_TRAIN,
              epochs = num_epochs,
              batch_size = ann_options.batch_size,
              validation_data = (X_TEST, Y_TEST),
              verbose = verbose)    #verbose 0 silent, verbose 1 = progress bar

    #evalaute or predict
    training_res = model.evaluate(X_TRAIN, Y_TRAIN, batch_size=ann_options.batch_size, verbose=verbose)
    testing_res  = model.evaluate(X_TEST , Y_TEST , batch_size=ann_options.batch_size, verbose=verbose)
    #training_res = model.predict(X_TRAIN, Y_TRAIN, batch_size=ann_options.batch_size)
    #testing_res = model.predict(X_TEST, Y_TEST, batch_size=ann_options.batch_size)

    # need to return the TRAINING LOSS, TRAINING ACCURACY, TESTING LOSS, TESTING ACCURACY, the HISTORY
    return (training_res[0], training_res[1], testing_res[0], testing_res[1], hist, model)


#collect all the data
data_csv = read_data()
data = data_csv.values

####################
#shuffle data
np.random.seed(123456)
np.random.shuffle(data)

#split into testing / training
#currently 70 - 30 split
split = 0.7
training_set_size = int(len(data) * split)
data_TRAINING = data[:training_set_size,:]
data_TESTING  = data[training_set_size:,:]
#split into features / classes
last_column_index = len(data[0]) - 1
#TRAINGING
X_TRAINING = data_TRAINING[:, :last_column_index]   #all features
Y_TRAINING = data_TRAINING[:, last_column_index]    #occupied (0 or 1)
#TESTING
X_TESTING  = data_TESTING [:, :last_column_index]   #all features
Y_TESTING  = data_TESTING [:, last_column_index]    #classes

#---------- Start Model Evaluation & Comparision ----------
#Train ANN, LR, and RF
#testing_options = ANN_Options('elu', 12, 3, 32, 0.1) #Old dataset settings
#testing_options = ANN_Options('relu', 6, 3, 32, 0.05) #New dataset settings
testing_options = ANN_Options('elu', 9, 3, 32, 0.01)
print("Training: " + testing_options.writeToFile())
results = BuildAnn(X_TRAINING, Y_TRAINING, X_TESTING, Y_TESTING, testing_options, verbose=True)
model = results[5]

print("")
print("------Keras-----")
print("Training loss: ", results[0])
print("Training accuracy: ", results[1])
print("Testing loss: ", results[2])
print("Testing accuracy: ", results[3])
#Train Logistic Regression
mod = linear_model.LogisticRegression(solver="sag",max_iter=3000)
LR_model = mod.fit(X_TRAINING,Y_TRAINING)
print("")
print("-----LR-----")
print("Testing Accuracy: ", LR_model.score(X_TESTING, Y_TESTING))
#Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=10, n_estimators=100)
rf.fit(X_TRAINING, np.ravel(Y_TRAINING))
print("")
print("-----RF-----")
print("Testing Accuracy: ", rf.score(X_TESTING, Y_TESTING))
#Plot ROC and PR curves
plt.style.use('ggplot')
#Determine TP and FP rate and AUC for each model
y_predict_prob = LR_model.predict_proba(X_TESTING)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(Y_TESTING, y_predict_prob)
auc_lr = auc(fpr_lr, tpr_lr)
#Determine Precision and Recall for each model
prec_lr, rec_lr, _ = precision_recall_curve(Y_TESTING, y_predict_prob)
auc_lr_pr = auc(rec_lr, prec_lr)

y_pred = model.predict(X_TESTING).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_TESTING, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)

prec_keras, rec_keras, _ = precision_recall_curve(Y_TESTING, y_pred)
auc_keras_pr = auc(rec_keras, prec_keras)

y_pred_rf = rf.predict_proba(X_TESTING)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y_TESTING, y_pred_rf)
auc_rf = auc(fpr_rf, tpr_rf)

prec_rf, rec_rf, _ = precision_recall_curve(Y_TESTING, y_pred_rf)
auc_rf_pr = auc(rec_rf, prec_rf)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.3f})'.format(auc_lr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('Model Evaluation/ROC.png')
plt.show()

plt.figure(2)
plt.xlim(-0.01, 0.25)
plt.ylim(0.75, 1.01)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.plot(fpr_lr, tpr_lr, label='LR (area = {:.3f})'.format(auc_lr))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.savefig('Model Evaluation/ROC_zoom.png')
plt.show()

plt.figure(3)
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.plot([0, 1], [0.5, 0.5], 'k--')
plt.plot(rec_keras, prec_keras, label='Keras (area = {:.3f})'.format(auc_keras_pr))
plt.plot(rec_rf, prec_rf, label='RF (area = {:.3f})'.format(auc_rf_pr))
plt.plot(rec_lr, prec_lr, label='LR (area = {:.3f})'.format(auc_lr_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve')
plt.legend(loc='best')
plt.savefig('Model Evaluation/PR.png')
plt.show()

plt.figure(4)
plt.xlim(0.6, 1.01)
plt.ylim(0.6, 1.01)
plt.plot([0, 1], [0.5, 0.5], 'k--')
plt.plot(rec_keras, prec_keras, label='Keras (area = {:.3f})'.format(auc_keras_pr))
plt.plot(rec_rf, prec_rf, label='RF (area = {:.3f})'.format(auc_rf_pr))
plt.plot(rec_lr, prec_lr, label='LR (area = {:.3f})'.format(auc_lr_pr))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve (zoomed in at top right)')
plt.legend(loc='best')
plt.savefig('Model Evaluation/PR_zoom.png')
plt.show()