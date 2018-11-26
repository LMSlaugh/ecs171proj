from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas
import matplotlib.pyplot as plt

"""
The purpose of this file is to create and train ANN models in a grid search
The grid search results are saved to a csv file called gridSearch.csv
The saved data inculdes testing/training loss and accuracy
Specific options for each ANN model are specified in ANN_OPTIONS
"""

#reads data from file using pandas
def read_data():
    return pandas.read_csv('scc_data_to_use_normalized.csv', index_col = 0, parse_dates=True, infer_datetime_format=True)


#collect all option permutations to use easily later on
class ANN_Options:
    def __init__(self, activation_function, num_nodes_per_hidden, num_hidden_layers, batch_size, learning_rate):
        self.activation_func = activation_function
        self.nodes_per_hidden = num_nodes_per_hidden
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    #used to print out this object to file or print stream
    def __str__(self):
        return str(self.activation_func) + " | hid nodes: " + str(self.nodes_per_hidden) + " | hid layers: " + str(self.num_hidden_layers) + " | batch size: " + str(self.batch_size) + " | learning rate: " + str(self.learning_rate)
    
    #write to the file in a different format
    def writeToFile(self):
        return "ANN_Options(" + str(self.activation_func) + "|" + str(self.nodes_per_hidden) + "|" + str(self.num_hidden_layers) + "|" + str(self.batch_size) + "|" + str(self.learning_rate) + ")"

"""
    main function to build and train the ANN
    X_TRAIN, Y_TRAIN = testing data
    X_TEST, Y_TEST   = testing data
    ann_options      = an instance of ANN_Options to specify the options for the ann
    verbose          = boolean whether you want a lot of details printed
"""
def BuildAnn(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, ann_options, verbose=True):
    model = Sequential()
    #input layer
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
    #train the ann
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
    return (training_res[0], training_res[1], testing_res[0], testing_res[1], hist)


#collect all the data
data_csv = read_data()
print("Data csv type: " + str(type(data_csv)))
#convert to numpy array
data = data_csv.values
print("Data type: " + str(type(data)))

#all permutations of these for grid search:
#activation functions:
activation_functions = ['tanh', 'sigmoid', 'elu', 'softplus', 'relu']
number_of_nodes_per_layer = [3,6,9,12]
number_of_hidden_layers   = range(1,4)  # 1 to 3 inclusive
batch_sizes = [32]
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]

print("\nNumber of permutations for grid search: ", end="")
print(len(activation_functions) * len(number_of_nodes_per_layer) * len(number_of_hidden_layers) * len(batch_sizes) * len(learning_rates))

#collect all the permutations in one array for ease of access
option_permutations = []
for act in activation_functions:
    for num_nodes in number_of_nodes_per_layer:
        for num_layers in number_of_hidden_layers:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    option_permutations.append(ANN_Options(act, num_nodes, num_layers, batch_size, learning_rate))

####################
#shuffle data
numpy.random.shuffle(data)

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

#print(X_TRAINING[0])


#run a specific model, and show the history graphically
#comment out the block comments to run this part.
"""
testing_options = ANN_Options('sigmoid', 12, 2, 32, 0.01)
print("Training: " + testing_options.writeToFile())
results = BuildAnn(X_TRAINING, Y_TRAINING, X_TESTING, Y_TESTING, testing_options, verbose=True)
history = results[4]    #gathers loss and accuracy over the training process

print("")
print("Training loss: ", results[0])
print("Training accuracy: ", results[1])
print("Testing loss: ", results[2])
print("Testing accuracy: ", results[3])

print(history.history.keys())

#print graph of loss / accuracy to make sure its actually working
epochs_X = numpy.linspace(1, len(history.history.get('loss')), len(history.history.get('loss')))
plt.plot(epochs_X, history.history.get('loss'), label="LOSS")
plt.plot(epochs_X, history.history.get('val_loss'), label="VAL_LOSS")
plt.plot(epochs_X, history.history.get('acc'), label="ACC")
plt.plot(epochs_X, history.history.get('val_acc'), label="VAL_ACC")
plt.legend()
plt.show()
"""


#perform the grid search over all of the option permutations
#note the grid search will always append to the file, and never delete the file.
for i in range(len(option_permutations)):
    options = option_permutations[i]
    print(str(i) + " / " + str(len(option_permutations)))
    print("Training: " + str(options))
    results = BuildAnn(X_TRAINING, Y_TRAINING, X_TESTING, Y_TESTING, options, verbose=False)
    #save results (accuracies and losses) to file
    with open("gridSearch.csv", "a") as gridSearchFile:
        gridSearchFile.write(options.writeToFile() + "," + str(results[0]) + "," + str(results[1]) + "," + str(results[2]) + "," + str(results[3]) + "\n")
    print("")

print("\n###GRID SEARCH DONE!\n###")
