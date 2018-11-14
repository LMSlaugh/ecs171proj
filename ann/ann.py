from keras.models import Sequential
from keras.layers import Dense
import pandas
import data_parse

def BuildAnn(xTrain,yTrain,hiddenLayerSize):
    model = Sequential()
    model.add(Dense(hiddenLayerSize,input_dim=xTrain.shape[1],activation='sigmoid'))
    model.add(Dense(hiddenLayerSize,activation='sigmoid'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.fit(xTrain,yTrain,epochs=20,batch_size=32)
    score = model.evaluate(xTrain,yTrain,batch_size=32)


data_csv = data_parse.read_data()
print("Data csv type: " + str(type(data_csv)))
data = data_csv.values
print("Data type: " + str(type(data)))
print(data)

#all permutations of these for grid search:
#activation functions:
activation_functions = ['tanh', 'sigmoid', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'hard_sigmoid']
number_of_nodes_per_layer = range(1,9)  # 1 to 8 inclusive
number_of_hidden_layers   = range(1,5)  # 1 to 4 inclusive
batch_sizes = [1,2,4,8,16,32,64]
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

print("\nNumber of permutations for grid search: ", end="")
print(len(activation_functions) * len(number_of_nodes_per_layer) * len(number_of_hidden_layers) * len(batch_sizes) * len(learning_rates))