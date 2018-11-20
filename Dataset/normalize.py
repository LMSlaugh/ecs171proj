import pandas
import numpy

def read_data():
    return pandas.read_csv('scc_data_to_use.csv', index_col = 0, parse_dates=True, infer_datetime_format=True)

data = read_data()
print(data)
print("Data csv type: " + str(type(data)))
#convert to numpy array
data_numpy = data.values
print("Data type: " + str(type(data_numpy)))
#print(data)


for column in range(0, len(data_numpy[0])):
    minNum = data_numpy[0][column]
    maxNum = data_numpy[0][column]

    for y in range(0, len(data_numpy)):
        n = data_numpy[y][column]
        minNum = min(minNum, n)
        maxNum = max(maxNum, n)

    print("Column: " + str(column) + "\tmin: " + str(minNum) + ", max: " + str(maxNum))

    #normalize the column to range [0, 1]
    for y in range(0, len(data_numpy)):
        n = data_numpy[y][column]
        normalizedN = (n - minNum) * 1.0 / (maxNum - minNum)
        data_numpy[y][column] = normalizedN

print("###")

#set all features to be floating point except for the last one (occupied vs unoccupied)
#data = data.astype(numpy.float64)
#print(data.dtypes)
data['AP Connection Count'] = data['AP Connection Count'].astype(numpy.float64)
data['Day of Week'] = data['Day of Week'].astype(numpy.float64)
#print(data.dtypes)

#copy to pandas
for column in range(0, len(data_numpy[0])):
    for y in range(0, len(data_numpy)):
        data.iat[y, column] =  data_numpy[y][column]

#output
output = data.to_csv('scc_data_to_use_normalized.csv', date_format='%m/%d/%Y %H:%M')
print("Output type: " +  str(type(output)))
if output == None:
    print("Output saved successfully")