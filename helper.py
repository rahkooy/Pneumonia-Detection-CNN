# Import necessary libraries
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay

# Data generators via ImageDataGenerator
datagen_train = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen_test = ImageDataGenerator(
    rescale=1./255
)

# Train Data Generator
def datagen_train_dir(dir, in_shape):
    generator = datagen_train.flow_from_directory(
        dir,
        target_size=(in_shape[0], in_shape[1]),  # target_size=(224, 224),
        batch_size=32,
        class_mode='binary',  # Since class_mode is binary, labels will be 0 or 1
        classes=['NORMAL', 'PNEUMONIA']  # Specify the class labels explicitly
    )
    return generator

# Test Data Generator
def datagen_test_dir(dir, in_shape):
    generator = datagen_test.flow_from_directory(
        dir,
        target_size=(in_shape[0], in_shape[1]),  # target_size=(224, 224),
        batch_size=32,
        class_mode='binary',  # Since class_mode is binary, labels will be 0 or 1
        classes=['NORMAL', 'PNEUMONIA']  # Specify the class labels explicitly
    )
    return generator

# Plot accuracy vs val_accuracy and Loss vs val_loss
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
    return plt.gcf()

def plot_confusion_matrix(cm, model_name, classes=['NORMAL', 'PNEUMONIA']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()   
    return plt.gcf()

# Calculate and plot ROC curve and AUC
def plot_roc_auc(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    return plt.gcf()
