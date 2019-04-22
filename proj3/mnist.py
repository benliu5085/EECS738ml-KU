import numpy as np

from mlxtend.data import loadlocal_mnist

def rrelu(x):
    return 1*(x > 0)

def relu(x):
    return x * ( x > 0)

def forward(x, W, b):
    z = np.dot(x,W)+b
    a = relu(z)
    return (z,a)

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a), axis=1, keepdims=True)

train_X, train_Y = loadlocal_mnist(
        images_path='train-images-idx3-ubyte',
        labels_path='train-labels-idx1-ubyte')
test_X, test_Y = loadlocal_mnist(
        images_path='t10k-images-idx3-ubyte',
        labels_path='t10k-labels-idx1-ubyte')
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
# Hyper parameters
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001
shape = [train_X.shape[1], 20, 20, num_classes]
reg_lambda = 0.01

# 60000x785
XX = train_X / 255.
# 60000x1
# YY = np.squeeze(np.eye(num_classes)[train_Y.reshape(-1)])
YY = train_Y

## initialization
np.random.seed(2)
weights_scale = np.sqrt(shape[0]*shape[1]/2)
W1 = np.random.randn(shape[0],shape[1]) / weights_scale
B1 = np.random.randn(shape[1]) / weights_scale
weights_scale = np.sqrt(shape[1]*shape[2]/2)
W2 = np.random.randn(shape[1],shape[2]) / weights_scale
B2 = np.random.randn(shape[2]) / weights_scale
weights_scale = np.sqrt(shape[2]*shape[3]/2)
W3 = np.random.randn(shape[2],shape[3]) / weights_scale
B3 = np.random.randn(shape[3]) / weights_scale

## training
inx = np.arange(0,XX.shape[0])
for cnt_epoch in range(0, num_epochs):
    np.random.seed(42)
    np.random.shuffle(inx)
    for i in range(0,XX.shape[0],batch_size):
        # forward
        a0 = XX[inx[i:i+batch_size] , :]
        (z1, a1) = forward(a0, W1, B1)
        (z2, a2) = forward(a1, W2, B2)
        (z3, a3) = forward(a2, W3, B3)
        P = softmax(z3)
        y = YY[inx[i:i+batch_size]]

        # compute loss
        corect_logprobs = -np.log(P[range(batch_size), y])
        data_loss = np.sum(corect_logprobs)
        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        loss = 1./batch_size * data_loss
        # backward
        delta4 = P
        delta4[range(batch_size), y] -= 1
        db3 = np.sum(delta4,axis=0)
        dw3 = np.dot(a2.T, delta4)

        delta3 = np.dot(delta4, W3.T) * rrelu(z2)
        db2 = np.sum(delta3,axis=0)
        dw2 = np.dot(a1.T, delta3)

        delta2 = np.dot(delta3, W2.T) * rrelu(z1)
        db1 = np.sum(delta2,axis=0)
        dw1 = np.dot(a0.T, delta2)

        dw3 += reg_lambda * W3
        dw2 += reg_lambda * W2
        dw1 += reg_lambda * W1

        W1 += -learning_rate * dw1
        B1 += -learning_rate * db1
        W2 += -learning_rate * dw2
        B2 += -learning_rate * db2
        W3 += -learning_rate * dw3
        B3 += -learning_rate * db3


        if i % 10000 == 0:
            print("[Epoch ["+str(cnt_epoch+1)+"/"+str(num_epochs)+"], step ["+str(i/100)+"/600], loss: " + str(loss))

        ##


## testing
# 60000x785
XX = test_X  / 255.
# 60000x1
# YY = np.squeeze(np.eye(num_classes)[train_Y.reshape(-1)])
YY = test_Y
(z1, a1) = forward(XX, W1, B1)
(z2, a2) = forward(a1, W2, B2)
(z3, a3) = forward(a2, W3, B3)
P = softmax(z3)
pred = np.argmax(P,axis=1)
acc = np.sum(pred==YY)
print("test accuracy: " + str(float(acc) / YY.shape[0]) )
