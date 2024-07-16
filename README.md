Project Purpose: Detecting Pneumonia from X-Ray images using Convolutional Neural Networks

Dataset:
https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images

Requirement:

 numpy, seaborn, pyplot from matplotlib, 
 
 tensorflow, keras, optimizer Adam, InceptionV3, ResNet50, DenseNet,

from sklearn metrics,  confusion_matrix, accuracy_score


utils directory includes python files each of which contains a CNN model and possibly some variants of it.
helper.py includes the following auxiliary functions:
-datagen_train_dir
-datagen_test_dir
-plot_history
-plot_confusion_matrix
-plot_roc_au

Models Considered and their prefrmance:
1-LeNet: quite old however, satisfiable results (91% accuracy)
2- AlexNet: quite old, however, satisfiable results (91% accuracy)
3-InceptionV3: better results than the previous models (94% accuracy)
4-ResNet with the following variants:
4.1-  with one Fully Connected Layer: has the best possible result among all
	models (95% accuracy) 
4.2- nontrainable ResNet50 with one Fully Connected Layer: 	the accuracy and loss values drop down drastically (0.83% accuracy)
4.3- Nontrainable Resnet50 with several normalisation layers: improves accuracy and loss, howeever, it is still worse that the trainable basemodel as in 4.1. 	(92% accuracy)
4.4- Nontrainable Resnet50 with several normalisation layers; further data preprocessing via rotation, etc. makes the gap between data accuracy/loss and val accuracy/loss much 
less. The accuracy and loss has not much improved in	comparison with 4.2. (86% accuracy)
5- DenseNet 	(95% accuracy)
6- Our Model: has very few layers of normalisation and dropout; shows competitive accuracy and loss as the best pretrained models (95% accuracy)

