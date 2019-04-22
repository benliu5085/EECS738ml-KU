import numpy as np
import pandas

def rrelu(x):
    return 1*(x > 0)

def relu(x):
    return x * ( x > 0)

def sigmoid(x):
    return np.exp(x)/(np.exp(x) + 1)

def rsigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def forward(x, W, b):
    z = np.dot(x,W)+b
    a = relu(z)
    return (z,a)

alphabet = {'M':[1,0,0], 'F':[0,1,0], 'I':[0,0,1]}
df = pandas.read_csv("abalone.csv")
df = df.dropna(0)
d = np.array(df)
X = np.zeros((d.shape[0],d.shape[1]+1))
Y = np.zeros(d.shape[0])
for i in range(d.shape[0]):
    X[i,:3] = alphabet[d[i,0]]
    X[i,3:] = d[i,1:-1]
    Y[i] = d[i,-1]

S = np.max(Y)
Y = Y / S
idx = np.arange(X.shape[0])
np.random.seed(37)
np.random.shuffle(idx)

train_id = idx[:int(0.7*X.shape[0])]
test_id = idx[int(0.7*X.shape[0]):]

# Hyper parameters
num_epochs = 50
batch_size = 32
learning_rate = 0.001
shape = [X.shape[1], 20, 20, 1]
reg_lambda = 0.01

XX = X[train_id]
YY = Y[train_id]

## initialization
np.random.seed(7)
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
    np.random.seed(59)
    np.random.shuffle(inx)
    for i in range(0, XX.shape[0], batch_size):
        # forward
        if i+batch_size < XX.shape[0]:
            __m = batch_size
        else:
            __m = XX.shape[0] - i
        z0 = XX[inx[i:i+__m] , :]
        a0 = sigmoid(z0)
        (z1, a1) = forward(a0, W1, B1)
        (z2, a2) = forward(a1, W2, B2)
        (z3, a3) = forward(a2, W3, B3)
        P = sigmoid(z3)

        y = YY[inx[i:i+__m]].reshape((__m,1))

        # compute loss
        loss = np.sum((P-y)*(P-y))
        loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        loss = 1./__m * loss

        # backward
        delta4 = (P-y)/__m
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


        if i % 256 == 0:
            print("[Epoch ["+str(cnt_epoch+1)+"/"+str(num_epochs)+"], step ["+str(i)+"/"+str(XX.shape[0])+"], loss: " + str(loss))


## testing
XX = X[test_id]
YY = Y[test_id].reshape((XX.shape[0],1))
z0 = XX
a0 = sigmoid(z0)
(z1, a1) = forward(a0, W1, B1)
(z2, a2) = forward(a1, W2, B2)
(z3, a3) = forward(a2, W3, B3)
P = sigmoid(z3)

acc = np.sum((P-YY)*(P-YY))*S*S
print("test accuracy: " + str(acc / YY.shape[0]) )
