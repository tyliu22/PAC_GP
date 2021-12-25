# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""
import abc
import numpy as np
import tensorflow as tf
from utils.transformations import Log1pe

"""
The following snippets are derived from GPFlow V 1.0
  (https://github.com/GPflow/GPflow)
Copyright 2017 st--, Mark van der Wilk, licensed under the Apache
License, Version 2.0, cf. 3rd-party-licenses.txt file in the root directory
of this source tree.
"""


class Kernel(object):
    '''
    Generic Kernel class
    '''
    __metaclass__ = abc.ABCMeta

    _jitter = 1e-10

    def __init__(self, jitter=1e-10):
        self._jitter = jitter

    @abc.abstractmethod
    def kernel(self, X, X2=None):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError("Subclass should implement this.")

    @classmethod
    def jitter(self):
        return self._jitter

class RBF:
    def __init__(self, input_dim, variance=1.0, lengthscale=None, ARD=True):
        with tf.name_scope('kern'):
            # convert input to at least 1-dimensional arrays
            self.variance = np.atleast_1d(variance)

            if lengthscale is not None:
                # If lengthscale is given check:
                # 1) individual lengthscale for each dimension
                # or 2) one lengthscale for all dimensions
                lengthscale = np.asarray(lengthscale, dtype=np.float64)
                # atleast_1d(): Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
                lengthscale = np.atleast_1d(lengthscale)

                assert_msg = 'Bad number of lengthscale dimensions'
                assert lengthscale.ndim == 1, assert_msg
                assert_msg = 'Bad number of lengthscales'
                assert lengthscale.size in [1, input_dim], assert_msg

                if ARD is True and lengthscale.size != input_dim:
                    lengthscale = np.ones(input_dim)*lengthscale

            else:
                # Default lengthscale if nothing is given
                if ARD is True:
                    # Independent lengthscalea for each dimension
                    lengthscale = np.ones(input_dim)
                    shape = (input_dim, )
                else:
                    # One lengthscale for all dimensions
                    lengthscale = np.ones(1)
                    shape = (1,)

            self.variance_unc_tf = tf.compat.v1.placeholder(dtype=tf.float64,
                                                  shape=(1, ),
                                                  name='variance_unc')

            lengthscales_unc_tf = tf.compat.v1.placeholder(dtype=tf.float64,
                                                 shape=shape,
                                                 name='lengthscales_unc')

        self.lengthscales_unc_tf = lengthscales_unc_tf
        self.lengthscale = lengthscale
        self.input_dim = input_dim
        self.trans = Log1pe()
        # softplus function: log( 1+exp(x) )
        self.variance_tf = self.trans.forward_tensor(self.variance_unc_tf)
        self.lengthscales_tf = self.trans.forward_tensor(lengthscales_unc_tf)
        self.ARD = ARD

    def square_dist(self, X, X2):
        X = X / self.lengthscales_tf
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales_tf
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))


    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X):
        N = tf.shape(X)[0]
        return tf.fill(tf.stack([N]), tf.squeeze(self.variance_tf))

    def K(self, X, X2=None):
        return self.variance_tf * tf.exp(-self.square_dist(X, X2) / 2)


class RBF_ARD(Kernel):
    """
    The radial basis function (RBF) or squared exponential kernel
    """

    def __init__(self, lengthscales, sigma, jitter=1e-3):

        super(RBF_ARD, self).__init__(jitter)

        self.lengthscales = tf.Variable(lengthscales, dtype=tf.float32)
        # self.log_sigma0 = tf.Variable([log_sigma0], dtype=tf.float32)
        self.sigma = tf.Variable([sigma], dtype=tf.float32)
        self.jitter = tf.constant([jitter], dtype=tf.float32)

    def kernel(self, X, X2=None):

        """
        This function computes the covariance matrix for the GP
        """

        if X2 is None:
            X2 = X
            white_noise = 0.0
            # white_noise = (self.jitter + tf.exp(self.log_sigma0)) * tf.eye(tf.shape(X)[0], dtype=tf.float32)
        else:
            white_noise = 0.0

        X = X / tf.sqrt(self.lengthscales)
        X2 = X2 / tf.sqrt(self.lengthscales)

        value = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
        value2 = tf.expand_dims(tf.reduce_sum(tf.square(X2), 1), 1)
        distance = value - 2 * tf.matmul(X, tf.transpose(X2)) + tf.transpose(value2)

        return tf.exp(self.log_sigma) * tf.exp(-0.5 * distance) + white_noise

    def get_params(self):
        return [self.lengthscales, self.sigma]

    def get_sigma(self):
        return self.sigma

    def get_var_points(self, data_points):
        return tf.ones([tf.shape(data_points)[0]]) * tf.exp(self.sigma) + (self.jitter)




