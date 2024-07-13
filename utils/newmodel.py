# a new CNN model for Fracture detection

from keras.models import Sequential
from keras.layers import Dense,Conv2D, Input, Reshape, MaxPooling2D, Flatten, Dropout, BatchNormalization


def newmodel(input_shape=(32,32,3)):
    model=Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128))
    # model.add(Dense(2, activation='softmax')) 
    model.add(Dense(1, activation='sigmoid'))
    
    return model


