# InceptionV3 Model Built-In in Keras

from tensorflow import keras
from keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, Sequential


def Inception_V3(input_shape=(299,299,3)):
    model = Sequential([
    # Buil-In Inception in Keras
    InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape),

    # Adding a fully connected layer
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    # one output neuron(not two as we do binary classification) 
    # with sigmoid instead of softmax
    Dense(1, activation='sigmoid')
    ])
    return model

# Without using Sequential and 
# Freezing pre-trained models via trainable = False
def inception_V3():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False 
    
    # Adding a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(20, activation='softmax')(x)

    model = Model(inputs = base_model.input, outputs = predictions)
    
    return model


