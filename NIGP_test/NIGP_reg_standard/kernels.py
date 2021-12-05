# -*- coding: utf-8 -*-
"""
    Function:
        RBF_ARD():
            para:
                lengthscales:
                sigma       :
            Input:
                X : training data
                X2: test data
    Setting:

    Modified based on @author: carlos's code in Noisy input Gaussian processing classification project

"""
import abc
import numpy as np
import tensorflow as tf
from utils.transformations import Log1pe



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



class RBF_ARD():
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def __init__(self, lengthscales, sigma, jitter=1e-3):
        # super(RBF_ARD, self).__init__(jitter)
        # self.lengthscales = lengthscales
        # self.sigma = sigma

        self.lengthscales = tf.Variable(lengthscales, dtype=tf.float64)
        self.sigma = tf.Variable(sigma, dtype=tf.float64)

        self.jitter = tf.constant([jitter], dtype=tf.float64)

    def kernel(self, X, X2=None):
        """
        This function computes the covariance matrix for the GP
            tf.square(): calculate the square of each elements
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

        return tf.exp(self.sigma) * tf.exp(-0.5 * distance) + white_noise

    def get_params(self):
        return [self.lengthscales, self.sigma]

    def get_sigma(self):
        return self.sigma

    def get_var_points(self, data_points):
        return tf.ones([tf.shape(data_points)[0]]) * tf.exp(self.sigma) + (self.jitter)




# calculate the kernel function TENSOR
# one dimension tensor
def kernel_Tensor(X1, X2=None, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """

    X1 = X1 / l
    Xs = tf.reduce_sum(tf.square(X1), 1)
    if X2 is None:
        temp = tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1)) + \
               -2 * tf.matmul(X1, X1, transpose_b=True)
    else:
        X2 = X2 / l
        X2s = tf.reduce_sum(tf.square(X2), 1)
        temp = -2 * tf.matmul(X1, X2, transpose_b=True) + \
               tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    return sigma_f**2 * tf.exp(-temp / 2)



