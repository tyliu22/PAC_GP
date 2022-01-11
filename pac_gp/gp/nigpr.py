# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

# import tensorflow as tf
import tensorflow as tf

import numpy as np
from gp.mean_functions import Zero
from gp.conditionals import feature_conditional


class NIGPR:
    """
    Noisy Input Gaussian Process Regression.

    Implementation of full GP regression following GPflow implementation
    """
    def __init__(self, X, Y, sn2, kern, mean_function=None, noise_input_variance_tf=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        noise_x: input noises first-order approximation regularization item (matrix)
        """
        self.X = X
        self.Y = Y

        # Here, sn2 is the same as the defined parameter (tf.placeholder) self.sn2_tf
        self.sn2 = sn2
        # self.noise_x = noise_x

        self.kern = kern
        self.mean_function = mean_function or Zero()

        # noise_input_variance_tf is the same as self.noise_input_variance_tf
        # self.noise_input_variance = noise_input_variance_tf

        self.N = tf.shape(X)[0]   # Number of data points
        self.D = tf.shape(X)[1]   # Input dimensionality
        self.R = tf.shape(Y)[1]   # Output dimensionality
        self.jitter = 1e-06

    # def _build_predict_f(self, Xnew, full_cov=True):
    #     """
    #     Compute the mean and variance of the latent function at some new points
    #     Xnew.
    #     jitter = 1e-06 : to prevent zero matrix
    #     """
    #     N = tf.shape(self.X)[0]
    #
    #     Kx = self.kern.K(self.X, Xnew) # if no Xnew, then Kx=K : k_N
    #     K = self.kern.K(self.X) # K_NN
    #
    #     # For the standard GP, (K_NN + sigma_n * I)
    #     # For the NIGP, (K_NN + sigma_n * I + noise_x_reg); noise_x_reg represents first-order approximation regularization item
    #     # K += tf.eye(N, dtype=tf.float64) * (self.sn2 + self.jitter)
    #     K += tf.eye(N, dtype=tf.float64) * (self.sn2 + self.jitter) + tf.matmul(self.noise_x, tf.transpose(self.noise_x))
    #
    #     L = tf.cholesky(K)  # L = chol(K_NN + sigma_n * I);
    #     A = tf.matrix_triangular_solve(L, Kx, lower=True) # inv(L) * k_N(x)
    #     V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X)) # inv(L) * (y_N-m_N)
    #     # NIGP f_mean = m(x) + k_N(x) * inv(K_NN + sigma_n * I) * (y_N-m_N)
    #     fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
    #     if full_cov:
    #         # NIGP f_var = K(x,x_new) + k_N(x) * inklv(K_NN + sigma_n * I) * k_N(x_new)
    #         fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True) #
    #         shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
    #         fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
    #     else:
    #         fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
    #         fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
    #     return fmean, fvar

    def _build_predict_NIGP_reg_item(self, feed):
        """
        Calculate the derivative of GP mean function
        mean_function is assumed as zero() function
        """

        N = tf.shape(self.X)[0]

        K_X_star = self.kern.K(self.VAR_X_test, self.VAR_X_train)
        K = self.kern.K(self.VAR_X_test)
        K += tf.eye(N, dtype=tf.float64) * (self.sn2 + self.jitter)
        K_inv = tf.linalg.inv(K)

        mean_f = tf.matmul(tf.matmul(K_X_star, K_inv), self.Y)
        f_grad = tf.gradients(mean_f, self.VAR_X_test)
        with tf.Session() as sess:
            grad_posterior_mean = sess.run(f_grad, feed_dict=feed)
            # grad_posterior_mean = sess.run(f_grad, feed_dict={VAR_X_train: self.X, VAR_X_test: self.X})

        # MUST be the diagnal matrix,
        f_grad_mean = np.diag(np.diag(grad_posterior_mean.dot(self.noise_input_variance_tf).dot(grad_posterior_mean.T)))
        f_grad_mean_tf = tf.convert_to_tensor(f_grad_mean, tf.float64)
        # regu_item = tf.matmul( tf.matmul(grad_posterior_mean, self.input_noise_variance), tf.transpose(grad_posterior_mean))
        # regu_diag_item = tf.matrix_diag(tf.matrix_diag_part(regu_item))

        return f_grad_mean, f_grad_mean_tf


    # Here the "self" means the
    def _build_predict_f(self, Xnew, full_cov=True, grad_posterior_mean=None):
        """
            # noise_input_variance=None
        Compute the mean and variance of the latent function at some new points
        Xnew.
        jitter = 1e-06 : to prevent zero matrix
        For standard GP, (K_NN + sigma_n * I)
        For NIGP, (K_NN + sigma_n * I + noise_x_reg); noise_x_reg represents first-order approximation regularization item

        """

        N = tf.shape(self.X)[0]

        Kx = self.kern.K(self.X, Xnew) # if no Xnew, then Kx=K : k_N
        K = self.kern.K(self.X) # K_NN

        # K += tf.eye(N, dtype=tf.float64) * (self.sn2 + self.jitter)
        # K += tf.eye(N, dtype=tf.float64) * (self.sn2 + self.jitter) + self.f_grad_mean_tf
        K += tf.eye(N, dtype=tf.float64) * (self.sn2 + self.jitter) + grad_posterior_mean

        L = tf.cholesky(K)  # L = chol(K_NN + sigma_n * I);
        A = tf.matrix_triangular_solve(L, Kx, lower=True) # inv(L) * k_N(x)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X)) # inv(L) * (y_N-m_N)
        # NIGP f_mean = m(x) + k_N(x) * inv(K_NN + sigma_n * I) * (y_N-m_N)
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)

        if full_cov:
            # NIGP f_var = K(x,x_new) + k_N(x) * inklv(K_NN + sigma_n * I) * k_N(x_new)
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

    def _build_predict_y(self, Xnew, full_cov=True, grad_posterior_mean=None):
        """
        Compute the mean and variance of the observations at some new points
        Xnew.
        """
        mean, var = self._build_predict_f(Xnew, full_cov, grad_posterior_mean)

        # if full_cov is True:
        #     noise = self.sn2 * tf.eye(tf.shape(Xnew)[0], dtype=tf.float64)
        #     var = var + noise[:, :, None]
        # else:
        #     var = var + self.sn2

        return mean, var



