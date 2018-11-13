from keras.models import Sequential
from keras.layers import Dense

def BuildAnn(xTrain,yTrain,hiddenLayerSize):
    model = Sequential()
    model.add(Dense(hiddenLayerSize,input_dim=xTrain.shape[1],activation='sigmoid'))
    model.add(Dense(hiddenLayerSize,activation='sigmoid'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.fit(xTrain,yTrain,epochs=20,batch_size=32)
    score = model.evaluate(xTrain,yTrain,batch_size=32)
    