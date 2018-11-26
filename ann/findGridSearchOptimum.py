import pandas as pd
#sort the csv file and output csv files of the sorted columns.
# Read a csv data file
names = ['params','training_loss','training_acc','testing_loss','testing_acc']
data = pd.read_csv('gridSearch.csv', names=names)
data_training_loss = data.sort_values('training_loss',ascending=False)
data_training_acc = data.sort_values('training_acc',ascending=False)
data_testing_loss = data.sort_values('testing_loss',ascending=False)
data_testing_acc = data.sort_values('testing_acc',ascending=False)
data_training_loss.to_csv('gridSearch_training_loss.csv')
data_training_acc.to_csv('gridSearch_training_acc.csv')
data_testing_loss.to_csv('gridSearch_testing_loss.csv')
data_testing_acc.to_csv('gridSearch_testing_acc.csv')

print("SORTED GRID SEARCH AND SAVED SORTED CSV FILES")
