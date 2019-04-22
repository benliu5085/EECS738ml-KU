This the source code (python 2.7) for project 3 of EECS 738 machine learning
by Dr. Martin Kuehnhausen
From (author) Ben Liu
================================================================================

dependency:
mlxtend - to load mnist data from local file

Including files:
mnist.py -- source code to train and test on mnist data set.

t10k-images-idx3-ubyte

t10k-labels-idx1-ubyte

train-images-idx3-ubyte

train-labels-idx1-ubyte

         -- mnist data set, unzip them first
         
         
abalone.py -- source code to train and test on abalone data set.

abalone.csv -- balone data set.


---------------------------

covLayer.py -- define class to initialize, forward propagate for CNN. Implemented img2col.

================================================================================

dataset1: mnist

Mnist dataset contains 60,000 grey-scale images, each of size 28x28 pixels, associating 
with labels {0, 1, 2, ... , 9}. Obviously, this is a multi-classification problem. I am 
going to train a multi-layer perceptron to finish the job. 

To be more specific, the networks have 4 layers of neurons, one input layer of 784 units,
one output layer of 10 units, and 2 hidden layers, each of 20 units. Activation of hidden
units is biased ReLU. The output units are biased softmax.

For training, I use cross-entropy as the loss function. Set learning rate to be 0.001,
update weights and biases by miniBatch strategy, set batch size to be 100, training on 10 
epochs. Regularization factor is 0.01, only for weights.

Initialize all parameter as samples of standard distribution, divided by factor 
sqrt(2/(L1xL2)), where L1 is the number of units in previous layer and L2 is the number of
units in current layer.

The Trained networks achieve 0.9585 accuracy on testing set of 10,000 images.

dataset2: abalone
abalone dataset consists of 4277 records of 8 features and 1 label, of which one feature is 
3-state(sex: 'F', 'M', 'I'), so I extend feature sex to be length 3 one-hot vector. Split 
the data set to be 70% training data, and 30% testing data.

This networks have 4 layers of neurons, one input layer of 10 units, with non-biased sigmoid
as activation (to scale input value), one output layer of 10 units, and 2 hidden layers, each
of 20 units. Activation of hidden units is biased ReLU. The output units are biased sugmoid.

For training, I use MSE as the loss function. Set learning rate to be 0.001, update weights 
and biases by miniBatch strategy, set batch size to be 32, training on 50 epochs. 
Regularization factor is 0.01, only for weights.

The Trained networks achieve accumulated square error 10.33 on testing set.

================================================================================

some discussion
(1) In order to squeeze out all performance, one need to compare training error and test error to 
find out the optimal hyperparameters, including networks structure, batch size and number of 
epochs, etc. One might also want to use cross valiadation to confirm training properly.
(2) I tried to implement convolutional neural networks, but the backward propagation stage is too tricky:
pooling layer has no parameter and it change the size of the tensor, so I am not sure how to compute the 
gradient using BP algorithm. In fact, I read something about upsample and only propagate the error to those
filter whose output is the maxima, but the weights is a matrix, not a vector like MLP, so I am not sure if I 
am doing vectorization correctly.




