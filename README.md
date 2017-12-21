# ML_Project2

Repository for the second project of Machine Learning (CS-???)

## Requirements
- Only tested on Linux with a kernel 4.13, but should work on any platform
- Keras 2.1.11
- Matplotlib
- Scikit-learn
- Numpy


## How to use
The model make use of a pre-trained weight of VGG16 that is available in [Keras v2.1.11](https://keras.io/)

There is options in the run.py to change the usage of the completely pre-trained model (using pre-trained weights also given) or recomputing the model from the pre-trained weight of VGG16. There is also options to change the directory of the training and test datasets.

## File Structure
### Model files
Files that contains the four different models that were implemented. There is dummy model that randomly predicts the output, a logistic model that uses logistic regression, a CNN model that uses neural network and a sliding window and finally a fcn model that uses a fully convolution network with pre-trained weights on vgg16.
### helpers.py
File that contains helpers ranging from file loading and predictions to utilities for one-hot encoding and cropping.
### cross_validation.py
Wrapper around Sci-kit learn k-fold to be able to use it with the form of our models
### Notebooks
There are two notebooks. One contains the basic data exploration that we did, the other contains the full restul of the cross-validation.

## Time estimation
