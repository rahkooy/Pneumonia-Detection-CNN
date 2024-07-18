#
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Sequential

# Using Sequential to stack built-in DenseNet169 on only one fully connected layer
def densenet(input_shape: tuple = (224, 224, 3)) -> Sequential:
    model=Sequential()
    model.add(Input(shape=input_shape)),
    model.add(DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)),
    model.add(GlobalAveragePooling2D()),
    model.add(Dense(128, activation='relu')),
    model.add(Dense(1, activation='sigmoid'))
    #])
    return model