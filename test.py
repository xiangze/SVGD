import tensorflow as tf
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from steinKL import steinKLpq
from edward.models import Normal

# RBF kernel
def rbf(a,b,r="0.3"):
    return tf.exp( -tf.div((a-b)*(a-b),tf.constant(r)))

def drbf(a,b,r="0.3"):
    return -tf.div((a-b),tf.constant(r/2)) * tf.exp( -tf.div((a-b)*(a-b),tf.constant(r)))


def build_toy_dataset(N, noise_std=0.5):
  X = np.concatenate([np.linspace(0, 2, num=N / 2),
                      np.linspace(6, 8, num=N / 2)])
  y = 2.0 * X + 10 * np.random.normal(0, noise_std, size=N)
  X = X.reshape((N, 1))
  return X, y

ed.set_seed(42)

N = 40  # number of data points
D = 1  # number of features

# DATA
X_train, y_train = build_toy_dataset(N)
X_test, y_test = build_toy_dataset(N)

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w) + b, scale=tf.ones(N))

# INFERENCE
qw = Normal(loc=tf.get_variable("qw/loc", [D]),
            scale=tf.nn.softplus(tf.get_variable("qw/scale", [D])))
qb = Normal(loc=tf.get_variable("qb/loc", [1]),
            scale=tf.nn.softplus(tf.get_variable("qb/scale", [1])))

inference = steinKLpq(rbf,drbf,{w: qw, b: qb}, data={X: X_train, y: y_train})
inference.run()

# CRITICISM
y_post = ed.copy(y, {w: qw, b: qb})
print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X: X_test, y_post: y_test}))

print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X: X_test, y_post: y_test}))
