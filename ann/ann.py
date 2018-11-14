import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas


num_epochs = 50


#reads data from file
def read_data():
    data_csv = pandas.read_csv('scc_data_to_use.csv', index_col = 0, parse_dates=True, infer_datetime_format=True)
    return data_csv


#collect all option permutations to use easily later on
class ANN_Options:
    def __init__(self, activation_function, num_nodes_per_hidden, num_hidden_layers, batch_size, learning_rate):
        self.activation_func = activation_function
        self.nodes_per_hidden = num_nodes_per_hidden
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    def __str__(self):
        return "ANN_Options: (Act func: " + str(self.activation_func) + " | hid nodes: " + str(self.nodes_per_hidden) + " | hid layers: " + str(self.num_hidden_layers) + " | batch size: " + str(self.batch_size) + " | learning rate: " + str(self.learning_rate)


"""
    main function to build the ann
    X_TRAIN, Y_TRAIN = testing data
    X_TEST, Y_TEST   = testing data
    ann_options      = an instance of ANN_Options to specify the options for the ann
"""
def BuildAnn(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, ann_options):
    model = Sequential()
    #input layer?
    model.add(Dense(ann_options.nodes_per_hidden,input_dim=X_TRAIN.shape[1],activation=ann_options.activation_func))
    #hidden layers
    for n in range(0, ann_options.num_hidden_layers):
        model.add(Dense(ann_options.nodes_per_hidden,activation=ann_options.activation_func))
    #output layer
    model.add(Dense(1,activation='softmax'))
    #stochastic gradient descent:
    optimizerSGD = keras.optimizers.SGD(lr=ann_options.learning_rate, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(loss='binary_crossentropy', optimizer=optimizerSGD, metrics=['accuracy'])
    model.fit(X_TRAIN, Y_TRAIN,
              epochs = num_epochs,
              batch_size = ann_options.batch_size,
              validation_data = (X_TEST, Y_TEST),
              verbose = 1)    #verbose 0 silent, verbose 1 = progress bar

    training_score = model.evaluate(X_TRAIN, Y_TRAIN, batch_size=ann_options.batch_size)
    testing_score  = model.evaluate(X_TEST , Y_TEST , batch_size=ann_options.batch_size)
    # need to return the TRAINING ERROR, TESTING ERROR, TRAINING ACCURACY, TESTING ACCURACY
    return (training_score, testing_score)


#collect all the data
data_csv = read_data()
print("Data csv type: " + str(type(data_csv)))
data = data_csv.values
print("Data type: " + str(type(data)))
#print(data)


#all permutations of these for grid search:
#activation functions:
activation_functions = ['tanh', 'sigmoid', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'hard_sigmoid']
number_of_nodes_per_layer = range(1,9)  # 1 to 8 inclusive
number_of_hidden_layers   = range(1,5)  # 1 to 4 inclusive
batch_sizes = [1,2,4,8,16,32,64]
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

print("\nNumber of permutations for grid search: ", end="")
print(len(activation_functions) * len(number_of_nodes_per_layer) * len(number_of_hidden_layers) * len(batch_sizes) * len(learning_rates))

option_permutations = []
for act in activation_functions:
    for num_nodes in number_of_nodes_per_layer:
        for num_layers in number_of_hidden_layers:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    option_permutations.append(ANN_Options(act, num_nodes, num_layers, batch_size, learning_rate))
"""
print("Activation function\tnum_nodes\tnum_layers\tbatch_size\tlearning_rate")
for options in option_permutations:
    print(options)
"""

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
#TRAINGING
X_TRAINING = data_TRAINING[:, :8]   #all features
Y_TRAINING = data_TRAINING[:, 8]    #occupied (0 or 1)
#TESTING
X_TESTING  = data_TESTING [:, :8]   #all features
Y_TESTING  = data_TESTING [:, 8]    #classes

#run a few options, testing to see if the ann works well or not