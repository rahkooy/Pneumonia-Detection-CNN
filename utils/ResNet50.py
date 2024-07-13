# ResNet50 Model

from tensorflow import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as res_preprocess
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization
from keras.models import Model, Sequential


# Built-in base model nontrainable with several normalisation layers
def resnet50_nontrain_normalised():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    # Flatten, normalise, add dense layers and dropout
    x = base_model.output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs = base_model.input, outputs = predictions)
    
    return model

# Built-in base model nontrainable with only one added fully connected layer
def resnet50_nontrain():
    base_model = ResNet50(weights='imagenet', include_top=False, 
                          input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    # Adding fully connected layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs = base_model.input, outputs = predictions)
    
    return model


# Using sequeltial to stack built-in ResNet50 on only one fully connected layer
def resnet50_seq(input_shape=(224,224,3)):
    resnet_model = Sequential([
        ResNet50(weights='imagenet', include_top=False, input_shape=input_shape),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(1,activation='sigmoid')    
    ])            
    return resnet_model