import jax.numpy as np
from jax import grad


#testing some basic linear regression

#X and Y are input output pairs, w, b are weight/bias vectors
X = np.array([
    [0.],
    [2.],
    [4.],
    [-4.],
    [10.],
    [-10.]
])

Y = np.array([
    [1.],
    [3.],
    [4.],
    [-5.],
    [12.],
    [-8.]
])

w = np.array([[-4.]])
b = -3.

#cost functions
def lsCost(X, Y, w, b):
    y_pred = X.dot(w) + b
    return ((Y - y_pred)**2).mean()

#defining important preloop things
cost = lsCost(X, Y, w, b)
lr = 0.01
debug = False


if (debug):
    print("initial cost" + str(cost))

#gradient descent loop
for i in range(100):
    w -= lr*grad(lsCost, argnums=2)(X, Y, w, b)
    b -= lr*grad(lsCost, argnums=3)(X, Y, w, b)
    if(i % 20 == 0 and debug):
        print(lsCost(X, Y, w, b))

print("w is: " + str(w))
print("b is: " + str(b))




