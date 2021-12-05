# -*- coding: utf-8 -*-
"""
    Function:
        NIGP_f_grad_mean():
            calculate the derivative of the mean function at TRAINING data points

        NIGP_reg_matrix():
            calculate the regularization item in NIGP, including square of mean
            slops multiply input noise covariance


    Setting:
        GP_f_mean:    X_*: test data;    X: train data;    Sigma_y: output noise
            f_bar = m(X_*) + K(X_*,X) * inv[ K(X, X) + Sigma_y*I_N ] * [ y - m(X) ] --- m(X) \= 0
            f_bar = K(X_*,X) * inv[ K(X, X) + Sigma_y*I_N ] * y                     --- m(X)  = 0

        GP_f_covariance:   X_*: test data;   X: train data;   Sigma_y: output noise
            f_cov = K(X_*,X_*) - K(X_*,X) * inv[ K(X, X) + Sigma_y*I_N ] * K(X,X_*)

        NIGP_f_grad_mean:
            the partial_differential value [ f_bar / X_* ] on X_* = X

"""


import numpy as np
import tensorflow as tf
from NIGP_test.NIGP_reg_standard.kernels import kernel_Tensor



# calculate the derivative of the mean function at TRAINING data points
def NIGP_f_grad_mean(X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    # Eq:  f_mean = m(X) + K * inv[K + Sigma_y*I_N] * [y - m(X)]  -- mean func of f at TRAINING points
    # Eq:  f_var  = K - K * inv[K + Sigma_y*I_N] * K              -- covariance func of f

    # VAR_X_train = tf.placeholder(tf.float64, shape=X_train.shape)
    # K = kernel_Tensor(X1=VAR_X_train, l=l, sigma_f=sigma_f)
    # K_reg = K + sigma_y ** 2 * tf.eye(len(X_train), dtype=tf.float64)
    # K_inv = tf.linalg.inv(K_reg)
    #
    # mean_f = tf.matmul(tf.matmul(K, K_inv), Y_train)
    # cov_f = tf.matmul(tf.matmul(K, K_inv), Y_train)
    #
    # f_grad = tf.gradients(mean_f, VAR_X_train)
    # with tf.Session() as sess:
    #     f_grad_mean = sess.run(f_grad,  feed_dict={VAR_X_train: X_train})
    # return f_grad_mean[0]

    VAR_X_train = tf.placeholder(tf.float64, shape=X_train.shape)
    VAR_X_test = tf.placeholder(tf.float64, shape=X_train.shape)
    # VAR_Y_train = tf.placeholder(tf.float64, shape=Y_train.shape)

    K_X_star = kernel_Tensor(X1=VAR_X_test, X2=VAR_X_train, l=l, sigma_f=sigma_f)
    K = kernel_Tensor(X1=VAR_X_train, l=l, sigma_f=sigma_f)
    K_reg = K + sigma_y ** 2 * tf.eye(len(X_train), dtype=tf.float64)
    K_inv = tf.linalg.inv(K_reg)

    #
    mean_f = tf.matmul(tf.matmul(K_X_star, K_inv), Y_train)

    f_grad = tf.gradients(mean_f, VAR_X_test)
    with tf.Session() as sess:
        f_grad_mean = sess.run(f_grad, feed_dict={VAR_X_train: X_train, VAR_X_test: X_train})
    return f_grad_mean[0]

# Calculate slide(K)= k+SIGMA_Y+Reg, including NIGP regularization items
def NIGP_reg_matrix(X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1e-8):

    # calculate the regularization item of posterior mean function

    f_grad_mean = NIGP_f_grad_mean(X_train, Y_train, l, sigma_f, sigma_y)
    SIGMA_x = sigma_x**2 * np.eye(len(X_train[1]))

    return f_grad_mean.T.dot(SIGMA_x).dot(f_grad_mean)