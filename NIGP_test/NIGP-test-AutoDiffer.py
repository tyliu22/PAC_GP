import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
# from matplotlib import animation, cm
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

"""
Test auto diff
"""


def kernel_Tensor(X1):
    return tf.sin(X1)
    # return X1

if __name__ == "__main__":
    noise_x = 0.5
    noise_y = 0.1

    X_train = (np.random.random((150, 1)) * 20.0 - 10.0).reshape(-1, 1) # generate 150 data points from interval [-10, 10]

    Y_train = np.sin(X_train) + noise_y * np.random.randn(*X_train.shape)
    X_train += noise_x * np.random.randn(*X_train.shape)

    # noise_y = pow(noise_y, 2)
    # noise_x = pow(noise_x, 2)
    # create data by constant interval
    # X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
    # Y_test = np.sin(X_test[:, 0])

    VAR_X_train = tf.placeholder(tf.float64, shape=X_train.shape)
    # K = kernel_Tensor(X1=VAR_X_train) + sigma_y ** 2 * tf.eye(len(X_train), dtype=tf.float64)
    # mu_s_old = tf.matmul(tf.matmul(K, K_inv, transpose_a=True), Y_train)

    mu_s_old = kernel_Tensor(VAR_X_train)

    var_grad = tf.gradients(mu_s_old, VAR_X_train)
    with tf.Session() as sess:
        grad_posterior_mean = sess.run(var_grad, feed_dict={VAR_X_train: X_train})

    print(grad_posterior_mean[0])


