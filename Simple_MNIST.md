NEURAL NETWORK USING NUMPY AND MATHS


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

```


```python
train_data = pd.read_csv('digit-recognizer/train.csv')
```


```python
train_data.head()
```


```python
data = np.array(train_data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape
```


```python
Y_train
```


```python
def init_params():
    W1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    W2 = np.random.rand(64, 128) - 0.5
    b2 = np.random.rand(64, 1) - 0.5
    W3 = np.random.rand(10, 64) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2,W3,b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3=W3.dot(A2)+b3
    A3=softmax(Z3)
    return Z1, A1, Z2,A2,Z3,A3

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2,Z3,A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1=W2.T.dot(dZ2)*ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2,dW3,db3

def update_params(W1, b1, W2, b2,W3,b3, dW1, db1, dW2, db2,dW3,db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    W3 = W3 - alpha * dW3  
    b3 = b3 - alpha * db3 
    return W1, b1, W2, b2,W3,b3
```


```python
def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2,W3,b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2,Z3,A3 = forward_prop(W1, b1, W2, b2,W3,b3, X)
        dW1, db1, dW2, db2,dW3,db3 = backward_prop(Z1, A1, Z2, A2,Z3, A3, W1, W2,W3, X, Y)
        W1, b1, W2, b2,W3,b3 = update_params(W1, b1, W2, b2,W3,b3, dW1, db1, dW2, db2,dW3,db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2,W3,b3
    
```


```python
W1, b1, W2, b2 ,W3,b3= gradient_descent(X_train, Y_train, 0.10, 500)
```


```python
def make_predictions(X, W1, b1, W2, b2,W3,b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2,W3,b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_prediction(index, W1, b1, W2, b2,W3,b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2,W3,b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
```


```python
test_prediction(0, W1, b1, W2, b2,W3,b3)
test_prediction(1, W1, b1, W2, b2,W3,b3)
test_prediction(2, W1, b1, W2, b2,W3,b3)
test_prediction(3, W1, b1, W2, b2,W3,b3)
```


```python
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2,W3,b3)
get_accuracy(dev_predictions, Y_dev)
```


```python

```


```python

```
