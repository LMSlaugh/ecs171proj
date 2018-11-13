import tensorflow as tf
import readFile as rf
import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
import keras
import keras.utils

NUM_CLASSES = 10
#columns dictionary
colMap = { 0: "mcg", 1: "gvh", 2: "alm", 3: "mit", 4: "erl", 5: "pox", 6: "vac", 7: "nuc", 8: "CLASS" }
#class names dictionary, and reversed dictionary
classes = {    0: "CYT", 1: "NUC", 2: "MIT", 3: "ME3", 4: "ME2", 5: "ME1", 6: "EXC", 7: "VAC", 8: "POX", 9: "ERL" }
classesRev = { "CYT": 0, "NUC": 1, "MIT": 2, "ME3": 3, "ME2": 4, "ME1": 5, "EXC": 6, "VAC": 7, "POX": 8, "ERL": 9 }

def isolationForest(inp):
    clf = IsolationForest(max_samples=len(inp), behaviour="new", contamination="auto")
    clf.fit(inp)
    y_pred_clf = clf.predict(inp)
    return y_pred_clf

def oneClassSVM(inp):
    svm = OneClassSVM(nu=0.2, gamma="auto")
    svm.fit(inp)
    y_pred_svm = svm.predict(inp)
    return y_pred_svm

def localOutlierFactor(inp):
    lof = LocalOutlierFactor(novelty=True, contamination="auto")
    lof.fit(inp)
    y_pred_lof = lof.predict(inp)
    return y_pred_lof

# Removes all occurences of elm from the list l
def removeAll(l, elm):
    newList = []
    for e in l:
        if e != elm:
            newList.append(e)
    return newList

#read data from file (see readFile.py if you want to see code)
data = rf.read_file("yeast.data", [str, float, float, float, float, float, float, float, float, str])
allFeatures = []    #only the features, not the class
dataClass = []      #only class column


#remove the first column of strings, and convert the class string into int
newData = []
for dataLine in data:
    newData.append(dataLine[1:])
    newData[-1][len(newData[-1])- 1] = classesRev[newData[-1][len(newData[-1])- 1]]
data = newData
#take separate features and class column
for dat in data:
    allFeatures.append(dat[:-1])
    dataClass.append(dat[-1])

#count number of items in each class, to verify the class distribution given in yeast.names
classCount = [0] * NUM_CLASSES
for dat in data:
    classCount[dat[-1]] = classCount[dat[-1]] + 1
print("Actual Class Distribution: ")
for i in range(0, NUM_CLASSES):
    print(classes[i] + " :\t" + str(classCount[i]))


#find outliers via three methods
pred_ISO = isolationForest(allFeatures)
pred_SVM = oneClassSVM(allFeatures)
pred_LOF = localOutlierFactor(allFeatures)

count_clf = 0
count_svm = 0
count_lof = 0
for n in pred_ISO:
    if n == -1:
        count_clf = count_clf + 1
for n in pred_SVM:
    if n == -1:
        count_svm = count_svm + 1
for n in pred_LOF:
    if n == -1:
        count_lof = count_lof + 1

print("\nOutliers:")
print("Isolation Forest Outliers: " + str(count_clf))
print("SVM Outliers: " + str(count_svm))
print("Local Outlier Factor Outliers: " + str(count_lof))

#find the outliers that are similar in different methods
inFL = 0
inFS = 0
inSL = 0
inAll = 0
for i in range(0, len(pred_ISO)):
    if pred_ISO[i] == -1 and pred_LOF[i] == -1:
        inFL = inFL + 1
    if pred_ISO[i] == -1 and pred_SVM[i] == -1:
        inFS = inFS + 1
    if pred_SVM[i] == -1 and pred_LOF[i] == -1:
        inSL = inSL + 1
    if pred_ISO[i] == -1 and pred_SVM[i] == -1 and pred_LOF[i] == -1:
        inAll = inAll + 1
print("Outliers found in multiple methods:")
print("F AND L: " + str(inFL))
print("F AND S: " + str(inFS))
print("S AND L: " + str(inSL))
print("ALL: " + str(inAll))

#take out the outliers from random forest from the dataset
for i in range(0, len(pred_ISO)):
    if pred_ISO[i] == -1:
        data[i] = []
        allFeatures[i] = []
        dataClass[i] = []
data = removeAll(data, [])
allFeatures = removeAll(allFeatures, [])
dataClass = removeAll(dataClass, [])

#class distribution after taking out outliers
classCount = [0] * NUM_CLASSES
for dat in data:
    classCount[dat[-1]] = classCount[dat[-1]] + 1
print("Class Distribution after no outliers: ")
for i in range(0, NUM_CLASSES):
    print(classes.get(i) + " :\t" + str(classCount[i]))

#start problem 2
#shuffle data first
#NOTE IM NOT USING allFeatures OR classes anymore, I WILL USE X AND Y INSTEAD
data = numpy.array(data)
numpy.random.shuffle(data)

#split into 70% 30% testing training
training_set_size = int(len(data) * 0.7)
testing_set_size = len(data) - training_set_size    #dont need this
data_TRAINING = data[:training_set_size,:]
data_TESTING  = data[training_set_size:,:]
#split into features / classes
X = data_TRAINING[:, :8]     #all features
Y = data_TRAINING[:, 8]      #classes
X_TESTING = data_TESTING[:, :8]     #all features
Y_TESTING = data_TESTING[:, 8]      #classes
Y_TESTING = keras.utils.to_categorical(Y_TESTING)
Y = keras.utils.to_categorical(Y)   #one-hot encoding
print(Y)









#creating ANN
num_hidden = [1,2,3]
num_nodes = [3,6,9,12]
errors = numpy.empty([3,4])
NUM_EPOCHS = 100
for y in range(0, len(num_hidden)):
    for x in range(0, len(num_nodes)):
        print("Model with " + str(num_hidden[y]) + " hidden layers, " + str(num_nodes[x]) + " nodes per layer")
        model = Sequential()
        #input
        model.add(Dense(input_dim=8, output_dim=3, activation='sigmoid'))

        #hidden layers
        for i in range(0, num_hidden[y]):
            model.add(Dense(output_dim=num_nodes[x] , activation='sigmoid'))

        #output layer
        output = Dense(output_dim=9, activation='softmax')
        model.add(output)

        optimizerSGD = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)  #change learning rate
        model.compile(loss='categorical_crossentropy', optimizer=optimizerSGD, metrics=['accuracy'])
        history = model.fit(X, Y,
                            epochs=NUM_EPOCHS,
                            batch_size=32,
                            validation_data=(X_TESTING, Y_TESTING),
                            verbose=0)

        #find accuracy
        #scores = model.evaluate(X, Y)
        #print("TRAINGING ACCURACY:")
        #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        #scores = model.evaluate(X_TESTING, Y_TESTING)
        #print("TESTING ACCURACY")
        #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        #print final losses
        #print("Final training loss: " + str(history.history.get('loss')[-1]))
        #print("Final testing loss: " + str(history.history.get('val_loss')[-1]))

        TESTING_LOSS = history.history.get('val_loss')[-1]
        errors[y][x] = TESTING_LOSS
        print(errors)




