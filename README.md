# Pneumonia Detection Project

## Project Purpose
Detecting Pneumonia from X-Ray images using Convolutional Neural Networks

## Dataset
[Labeled Chest X-Ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images)

## Requirements
- `numpy`
- `seaborn`
- `pyplot` from `matplotlib`
- `tensorflow`
- `keras`
  - Optimizer: `Adam`
  - Models: `InceptionV3`, `ResNet50`, `DenseNet`
- `sklearn`
  - Metrics: `confusion_matrix`, `accuracy_score`

## Utils Directory
The `utils` directory includes Python files, each of which contains a CNN model and possibly some variants of it. 

### helper.py includes the following auxiliary functions:
- `datagen_train_dir`
- `datagen_test_dir`
- `plot_history`
- `plot_confusion_matrix`
- `plot_roc_au`

## Models Considered and Their Performance
1. **LeNet**: 
   - Quite old but produces satisfiable results
   - 91% accuracy
   
2. **AlexNet**: 
   - Also quite old but produces satisfiable results
   - 91% accuracy

3. **InceptionV3**: 
   - Better results than the previous models
   - 94% accuracy

4. **ResNet** with the following variants:
   - **4.1 With One Fully Connected Layer**:
     - Best possible result among all models
     - 95% accuracy
   - **4.2 Nontrainable ResNet50 with One Fully Connected Layer**:
     - Accuracy and loss values drop drastically
     - 83% accuracy
   - **4.3 Nontrainable ResNet50 with Several Normalisation Layers**:
     - Improves accuracy and loss but still worse than the trainable base model (4.1)
     - 92% accuracy
   - **4.4 Nontrainable ResNet50 with Several Normalisation Layers and Further Data Preprocessing (e.g., rotation)**:
     - Makes the gap between data accuracy/loss and validation accuracy/loss much less
     - 86% accuracy

5. **DenseNet**: 
   - 95% accuracy

6. **Our Model**: 
   - Has very few layers of normalisation and dropout
   - Shows similar accuracy and loss as the best pre-trained models
   - 95% accuracy
