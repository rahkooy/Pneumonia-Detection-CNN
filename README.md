# Pneumonia Detection Project

## Project Purpose
Detecting Pneumonia from X-Ray images using Convolutional Neural Networks

## Dataset
The dataset contains 5,856 Chest X-Ray images. The images are split into a training set (train directory) and a testing set (test directory) of independent patients. There are two classes of images with labels NORMAL and PNEUMONIA. The dataset  is publicly available at 
[Labeled Chest X-Ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images),
which is itself taken from https://data.mendeley.com/datasets/rscbjbr9sj/3.
under Licence: CC BY 4.0, and the main paper is https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5


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
The `utils` directory includes Python files, each of which contains a CNN model and possibly some variants of it. It also contains helper.py, which includes the following auxiliary functions:
- `datagen_train_dir`
- `datagen_test_dir`
- `plot_history`
- `plot_confusion_matrix`
- `plot_roc_au`

## Models Considered and Their Performance
1. **Our Model**: 
   - Has only seven layers, including two concolutional layers and two maxpooling layers with only ~300,000 parameters
   - Shows the best accuracy and loss, the same as ResNet with 130 million parameters and other large pre-trained models
   - 95% accuracy

2. **LeNet**: 
   - Quite old but produces satisfiable results
   - 91% accuracy
   
3. **AlexNet**: 
   - Also quite old but produces satisfiable results
   - 91% accuracy

4. **InceptionV3**: 
   - Better results than the previous models
   - 94% accuracy

5. **ResNet** with the following variants:
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

6. **DenseNet**: 
   - 95% accuracy