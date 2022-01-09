# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import GPy

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.optimize import minimize
import tensorflow.compat.v1 as tf

from gp.mean_functions import Zero
from gp.kerns import RBF
from gp.pac_gp import PAC_INDUCING_HYP_GP, PAC_HYP_GP
from gp.pac_nigp import NIGP_PAC_HYP_GP

from utils.data_generator import generate_sin_data

# %% Configuration

# Number of data points
N_train = 50
N_test = 100

# Number of inducing inputs
M = 10
# Input space dimension
D = 1

x_min = -3
x_max = 3
dx = (x_max - x_min) / 6.0

epsilon_np = 0.1
delta_np = 0.001

# %% Generate data
x_data, y_data = generate_sin_data(N_train, x_min+dx, x_max-dx,
                                   0.4**2, random_order=True)
x_true, y_true = generate_sin_data(N_test, x_min, x_max, None,
                                   random_order=False)

# Add noise input, obs input: x' = x + eps_x
noise_input_std = 0.5 # input noise std
noise_input_variance = noise_input_std**2 # input noise variance

noise_eps = np.random.normal(loc=0.0, scale=np.sqrt(noise_input_std),
                       size=x_data.shape)
x_data = x_data + noise_eps
# (x_data, y_data) (x_true, y_true), noise_input_variance



"""
    NIGP for comparison
"""
# calculate the kernel function NUMPY
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) \
             - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

# calculate the kernel function TENSOR
def kernel_Tensor(X1, X2=None, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """

    # sqdist = X1**2 + X2**2 - 2 * tf.matmul(X1, tf.transpose(X2))
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

# calculate the derivative of the mean function at TRAINING data points
def NIGP_f_grad_mean(X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
        Eq:  f_mean = m(X) + K * inv[K + Sigma_y*I_N] * [y - m(X)]  -- mean func of f at TRAINING points
        Eq:  f_var  = K - K * inv[K + Sigma_y*I_N] * K              -- covariance func of f
        only related X_train, y_train, and kernel para
    """

    VAR_X_train = tf.placeholder(tf.float64, shape=X_train.shape)
    VAR_X_test = tf.placeholder(tf.float64, shape=X_train.shape)
    # VAR_Y_train = tf.placeholder(tf.float64, shape=Y_train.shape)

    K_X_star = kernel_Tensor(X1=VAR_X_test, X2=VAR_X_train, l=l, sigma_f=sigma_f)
    K = kernel_Tensor(X1=VAR_X_train, l=l, sigma_f=sigma_f)
    K_reg = K + sigma_y ** 2 * tf.eye(len(X_train), dtype=tf.float64)
    K_inv = tf.linalg.inv(K_reg)

    mean_f = tf.matmul(tf.matmul(K_X_star, K_inv), Y_train)

    f_grad = tf.gradients(mean_f, VAR_X_test)
    with tf.Session() as sess:
        f_grad_mean = sess.run(f_grad, feed_dict={VAR_X_train: X_train, VAR_X_test: X_train})
    return f_grad_mean[0]

# Calculate slide(K)= k+SIGMA_Y+Reg, including NIGP regularization items
def NIGP_K_grad_mean(X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1e-8):
    """
        calculate the derivative of the mean function at training data points
        f_bar = m(X) + K * inv[ K + Sigma_y*I_N ] * [ y - m(X) ]  -----  m(X) \= 0
        f_bar = K * inv[ K + Sigma_y*I_N ] * y                    -----  m(X)  = 0
    """

    # calculate the mean function of posterior distribution
    f_grad_mean = NIGP_f_grad_mean(X_train, Y_train, l, sigma_f, sigma_y)
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    SIGMA_x = sigma_x**2 * np.eye(len(X_train[1]))
    K_nigp = K + np.diag(np.diag(f_grad_mean.dot(SIGMA_x).dot(f_grad_mean.T)))
    # K_nigp = K + f_grad_mean.dot(SIGMA_x).dot(f_grad_mean.T)

    return K_nigp

def NIGP_posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1e-8):
    """
    Computes the sufficient statistics of the posterior distribution with NIGP
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """

    # K_Xtr_Xtr = kernel(X_train, X_train, l=l, sigma_f=sigma_f)
    K_Xs_Xs   = kernel(X_s,     X_s,     l=l, sigma_f=sigma_f)
    K_Xs_Xtr  = kernel(X_s,     X_train, l=l, sigma_f=sigma_f)
    K_Xtr_Xs  = kernel(X_train, X_s,     l=l, sigma_f=sigma_f)

    K_NIGP = NIGP_K_grad_mean(X_train, Y_train, l=l, sigma_f=sigma_f, sigma_y=sigma_y, sigma_x=sigma_x)
    K_nigp_inv = inv(K_NIGP)

    # NIGP posterior distribution mean and covariance
    mu_s_nigp = K_Xs_Xtr.dot(K_nigp_inv).dot(Y_train)
    cov_s_nigp = K_Xs_Xs - K_Xs_Xtr.dot(K_nigp_inv).dot(K_Xtr_Xs)

    return mu_s_nigp, cov_s_nigp

def prediction(X_test, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1e-8):
    mu_nigp_pred, cov_nigp_pred = NIGP_posterior(X_test, X_train, Y_train, l=l, sigma_f=sigma_f,
                                                 sigma_y=sigma_y, sigma_x=sigma_x)
    return mu_nigp_pred, cov_nigp_pred

def nll_fn_nigp(X_train, Y_train, noise_y, NIGP_matrix_init_para):
    """
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of Eq. (11), if
               False use a numerically more stable implementation.

    Returns:
        Minimization objective.
    """

    Y_train = Y_train.ravel()

    def nll_naive(theta):

        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise_y ** 2 * np.eye(len(X_train))
        K_nigp = K + NIGP_matrix_init_para
        return 0.5 * np.log(det(K_nigp)) + \
               0.5 * Y_train.dot(inv(K_nigp).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2 * np.pi)
    return nll_naive

# calculate the posterior of NIGP
def f_nigp_posterior(X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1.0):
    """
        posterior mean function of NIGPR, including a regularization item
    """

    f_grad_mean = NIGP_f_grad_mean(X_train, Y_train, l, sigma_f, sigma_y)

    K = kernel(X_train, X_train, l, sigma_f)

    K_NIGP = NIGP_K_grad_mean(X_train, Y_train, l=l, sigma_f=sigma_f, sigma_y=sigma_y, sigma_x=sigma_x)
    K_inv = inv(K_NIGP)

    mu_s = K.T.dot(K_inv).dot(Y_train)
    cov_s = K - K.T.dot(K_inv).dot(K)

    return mu_s, cov_s




if __name__ == "__main__":

    np.random.seed(9527)
    # Fig_path = '/Users/tianyuliu/PycharmProjects/PAC_GP/Results_fig/'
    Fig_path = None

    # ****************************************** #
    # Data preparing                             #
    # ****************************************** #
    X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
    Y_test = 2.0 * np.sin(X_test)

    # std: covariance of noisy_X and noisy_y
    noise_y = 0.4  # Output noisy std
    noise_x = 0.4  # Iutput noisy std

    # New noisy training data
    # X_train = np.random.random((150, 1)) * 20.0 - 10.0 # generate 150 data points from interval [-10, 10]
    X_train = np.linspace(-10, 10, 150).reshape(-1, 1)
    Y_train = 2.0 * np.sin(X_train) + noise_y * np.random.randn(*X_train.shape)
    X_train_obs = X_train + noise_x * np.random.randn(*X_train.shape)
    X_train = X_train_obs

    # ---- intialization parameter ---- #
    l       = 1.0
    sigma_f = 1.0
    sigma_x = noise_x # std, variance= sigma_x**2
    sigma_y = noise_y # std, variance= sigma_y**2

    # (x_data, y_data) (x_true, y_true), noise_input_variance

    f_grad_mean = NIGP_f_grad_mean(X_train, Y_train, l, sigma_f, sigma_y)
    SIGMA_x = sigma_x ** 2 * np.eye(len(X_train[1]))
    NIGP_matrix_init_para = np.diag(np.diag(f_grad_mean.dot(SIGMA_x).dot(f_grad_mean.T)))

    # ****************************************** #
    # Parameters optimization                    #
    # ****************************************** #

    # Optimize the NIGP nll loss (used in NIGPR), with regularization item, slope of mean func
    res = minimize(nll_fn_nigp(X_train, Y_train, noise_y, NIGP_matrix_init_para), [1, 1],
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B',
                   tol=1e-12,
                   options={'disp': False, 'eps': 0.001}
                   )
    l_opt, sigma_f_opt = res.x

    mu_s_nigp_fit, cov_s_nigp_fit = NIGP_posterior(X_test, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt,
                                                   sigma_y=noise_y, sigma_x=noise_x)


















"""
    PAC-NIGP for comparison
"""

# pac_gp = PAC_INDUCING_HYP_GP(X=x_data, Y=y_data, Z=z_gpy,
#                              sn2=sn2_gpy,
#                              kernel=kern, mean_function=mean,
#                              epsilon=epsilon_np, delta=delta_np,
#                              verbosity=0,
#                              method='bkl', loss='01_loss')

# pac_gp = PAC_HYP_GP(X=x_data, Y=y_data,
#                     sn2=sn2_gpy,
#                     kernel=kern, mean_function=mean,
#                     epsilon=epsilon_np, delta=delta_np,
#                     verbosity=0,
#                     method='bkl', loss='01_loss')


# %% Set up and train PAC-GP model


kern = RBF(D)
mean = Zero() # mean function: zwro()
noise_input_variance_np = np.array([[noise_input_variance]], dtype=np.float64)
# noise_input_variance_np = np.ndarray(shape=(1,1), dtype=np.float64)
# To create an array with the same size of sn2_gpy,which is the original one
sn2_nigpy = np.array([[0.01]], dtype=np.float64).reshape(1,)

pac_nigp = NIGP_PAC_HYP_GP(X=x_data, Y=y_data,
                         sn2=sn2_nigpy,
                         kernel=kern, mean_function=mean,
                         epsilon=epsilon_np, delta=delta_np,
                         verbosity=0,
                         method='bkl', loss='01_loss',
                         noise_input_variance=noise_input_variance_np)

# Parameter
# (self, X, Y, sn2, kernel=None, mean_function=None,
#                  epsilon=0.2, delta=0.01, verbosity=0, method='bkl',
#                  loss='01_loss', noise_input_variance=None)
pac_nigp.optimize()
# Z_opt = pac_gp.Z
# pac_nigp_



"""
# %% Set up and train GPy model for comparison
"""
kernel = GPy.kern.RBF(input_dim=D, ARD=True)

full_gpy = GPy.models.GPRegression(x_data, y_data, kernel=kernel)
full_gpy.optimize()

sparse_gpy = GPy.models.SparseGPRegression(x_data, y_data,
                                           kernel=kernel, num_inducing=M)
sparse_gpy.optimize()


# Initialize GP parameters from optimized sparse GP model (GPy)
sf2_gpy = sparse_gpy.rbf.variance.values
sn2_gpy = sparse_gpy.Gaussian_noise.variance.values
lengthscales_gpy = sparse_gpy.rbf.lengthscale.values
z_gpy = sparse_gpy.inducing_inputs.values



"""
    Predict on test data for comparison
"""
y_mean_full_gpy, y_var_full_gpy = full_gpy.predict(x_true)
y_mean_sparse_gpy, y_var_sparse_gpy = sparse_gpy.predict(x_true)
y_mean_pac_nigp, y_var_pac_nigp = pac_nigp.predict(Xnew=x_true, full_cov=False)



"""
    Estimation accuracy analysis
"""
# # Random Forest Regression
# regr = RandomForestRegressor(max_depth=4, random_state=0)
# regr.fit(x_data, y_data)
# y_fit_RF = regr.predict(x_true)
#
# # SVM Regression
# svr = SVR().fit(x_data, y_data)
# y_fit_SVR = svr.predict(x_true)

# print("Random Forest error: " + str(mean_squared_error(y_true, y_fit_RF)))
# print("SVR error: " + str(mean_squared_error(y_true, y_fit_SVR)))

print("GP error: " + str(mean_squared_error(y_true, y_mean_full_gpy)))
print("sparse GP error: " + str(mean_squared_error(y_true, y_mean_sparse_gpy)))
# print("NIGP error: " + str(mean_squared_error(y_true, y_mean_nigpy)))
print("PAC-NIGP error: " + str(mean_squared_error(y_true, y_mean_pac_nigp)))




# %% Plot data and GPy/GP tf predictions including -*- coding: utf-8 -*-
# """
# Copyright (c) 2018 Robert Bosch GmbH
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# @author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
# """
#
# import GPy
#
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
#
# from gp.mean_functions import Zero
# from gp.kerns import RBF
# from gp.pac_gp import PAC_INDUCING_HYP_GP, PAC_HYP_GP
# from gp.pac_nigp import NIGP_PAC_HYP_GP
#
# from utils.data_generator import generate_sin_data
#
# # %% Configuration
#
# # Number of data points
# N_train = 50
# N_test = 100
#
# # Number of inducing inputs
# M = 10
#
# # Input space dimension
# D = 1
#
# x_min = -3
# x_max = 3
# dx = (x_max - x_min) / 6.0
#
# epsilon_np = 0.1
# delta_np = 0.001
#
# # %% Generate data
#
# x_data, y_data = generate_sin_data(N_train, x_min+dx, x_max-dx,
#                                    0.1**2, random_order=True)
# x_true, y_true = generate_sin_data(N_test, x_min, x_max, None,
#                                    random_order=False)
#
# # Add noise input, obs input: x' = x + eps_x
# noise_input_std = 0.25 # input noise std
# noise_input_variance = noise_input_std**2 # input noise variance
#
# noise_eps = np.random.normal(loc=0.0, scale=np.sqrt(noise_input_std),
#                        size=x_data.shape)
# x_data = x_data + noise_eps
#
# # GPy.models.gplvm
#
# # %% Set up and train GPy model for comparison
# kernel = GPy.kern.RBF(input_dim=D, ARD=True)
#
# full_gpy = GPy.models.GPRegression(x_data, y_data, kernel=kernel)
# full_gpy.optimize()
#
# sparse_gpy = GPy.models.SparseGPRegression(x_data, y_data,
#                                            kernel=kernel, num_inducing=M)
# sparse_gpy.optimize()
#
# # Initialize GP parameters from optimized sparse GP model (GPy)
# sf2_gpy = sparse_gpy.rbf.variance.values
# sn2_gpy = sparse_gpy.Gaussian_noise.variance.values
# lengthscales_gpy = sparse_gpy.rbf.lengthscale.values
# z_gpy = sparse_gpy.inducing_inputs.values
#
#
# # pac_gp = PAC_INDUCING_HYP_GP(X=x_data, Y=y_data, Z=z_gpy,
# #                              sn2=sn2_gpy,
# #                              kernel=kern, mean_function=mean,
# #                              epsilon=epsilon_np, delta=delta_np,
# #                              verbosity=0,
# #                              method='bkl', loss='01_loss')
#
# # pac_gp = PAC_HYP_GP(X=x_data, Y=y_data,
# #                     sn2=sn2_gpy,
# #                     kernel=kern, mean_function=mean,
# #                     epsilon=epsilon_np, delta=delta_np,
# #                     verbosity=0,
# #                     method='bkl', loss='01_loss')
#
#
# # %% Set up and train PAC-GP model
#
#
#
# kern = RBF(D)
# mean = Zero() # mean function: zwro()
# noise_input_variance_np = np.array([[noise_input_variance]], dtype=np.float64)
# # noise_input_variance_np = np.ndarray(shape=(1,1), dtype=np.float64)
# # To create an array with the same size of sn2_gpy,which is the original one
# sn2_nigpy = np.array([[0.01]], dtype=np.float64).reshape(1,)
#
# pac_nigp = NIGP_PAC_HYP_GP(X=x_data, Y=y_data,
#                          sn2=sn2_nigpy,
#                          kernel=kern, mean_function=mean,
#                          epsilon=epsilon_np, delta=delta_np,
#                          verbosity=0,
#                          method='bkl', loss='01_loss',
#                          noise_input_variance=noise_input_variance_np)
#
# # Parameter
# # (self, X, Y, sn2, kernel=None, mean_function=None,
# #                  epsilon=0.2, delta=0.01, verbosity=0, method='bkl',
# #                  loss='01_loss', noise_input_variance=None)
# pac_nigp.optimize()
# # Z_opt = pac_gp.Z
#
# # %% Predict on test data
# y_mean_full_gpy, y_var_full_gpy = full_gpy.predict(x_true)
# y_mean_sparse_gpy, y_var_sparse_gpy = sparse_gpy.predict(x_true)
# y_mean_pac_nigp, y_var_pac_nigp = pac_nigp.predict(Xnew=x_true, full_cov=False)
#
#
#
# """
#     Estimation accuracy analysis
# """
# # Random Forest Regression
# regr = RandomForestRegressor(max_depth=4, random_state=0)
# regr.fit(x_data, y_data)
# y_fit_RF = regr.predict(x_true)
#
# # SVM Regression
# svr = SVR().fit(x_data, y_data)
# y_fit_SVR = svr.predict(x_true)
#
# print("Random Forest error" + mean_squared_error(y_true, y_fit_RF))
# print("SVR error" + mean_squared_error(y_true, y_fit_SVR))
# print("GP error" + mean_squared_error(y_true, y_mean_full_gpy))
# print("PAC-NIGP error" + mean_squared_error(y_true, y_mean_pac_nigp))
#
#
# # %% Plot data and GPy/GP tf predictions including PAC GP
# plt.figure('Data and GPy/GPtf/PAC GP predictions')
# plt.clf()
#
#
# plt.subplot(1, 3, 1)
# plt.title('Full GP (GPy)')
# plt.plot(x_data, y_data, '+', label='data points')
# plt.plot(x_true, y_true, '-', label='true function')
#
# y = np.squeeze(y_mean_full_gpy)
# error = np.squeeze(2 * np.sqrt(y_var_full_gpy))
# plt.plot(x_true, y, '-', color='C2', label='full GP (GPy)')
# plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)
# plt.xlabel('input $x$')
# plt.ylabel('output $y$')
# plt.grid()
# plt.legend(loc=2)
# plt.ylim([-1.5, 1.5])
# plt.xlim([-3, 3])
#
# plt.subplot(1, 3, 2)
# plt.title('Sparse GP (GPy)')
# plt.plot(x_data, y_data, '+', label='data points')
# plt.plot(x_true, y_true, '-', label='true function')
#
# y = np.squeeze(y_mean_sparse_gpy)
# error = np.squeeze(2 * np.sqrt(y_var_sparse_gpy))
# plt.plot(x_true, y, '-', color='C2', label='sparse GP (GPy)')
# plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)
#
# plt.plot(np.squeeze(sparse_gpy.inducing_inputs), -1.5 * np.ones((M, )), 'o',
#          color='C3', label='GPy inducing inputs')
# plt.xlabel('input $x$')
# plt.ylabel('output $y$')
# plt.grid()
# plt.legend(loc=2)
# plt.ylim([-1.5, 1.5])
# plt.xlim([-3, 3])
#
#
# plt.subplot(1, 3, 3)
# plt.title('PAC GP')
# plt.plot(x_data, y_data, '+', label='data points')
# plt.plot(x_true, y_true, '-', label='true function')
#
# y = np.squeeze(y_mean_pac_gp)
# error = np.squeeze(2 * np.sqrt(y_var_pac_gp))
# plt.plot(x_true, y, '-', color='C2', label='sparse GP (tf)')
# plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)
#
# plt.plot(np.squeeze(z_gpy), -1.5 * np.ones((M, )), 'o',
#          color='C3', label='GPy inducing inputs')
# # plt.plot(np.squeeze(Z_opt), -1.5 * np.ones((M, )), 'o',
# #          color='C4', label='PAC GP inducing inputs')
# plt.xlabel('input $x$')
# plt.ylabel('output $y$')
# plt.grid()
# plt.legend(loc=2)
# plt.ylim([-1.5, 1.5])
# plt.xlim([-3, 3])
# plt.show()ng PAC GP
# plt.figure('Data and GPy/GPtf/PAC GP predictions')
# plt.clf()


plt.subplot(2, 2, 1)
plt.title('Full GP')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_full_gpy)
error = np.squeeze(2 * np.sqrt(y_var_full_gpy))
plt.plot(x_true, y, '-', color='C2', label='full GP')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)
# plt.xlabel('input $x$')
# plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-2, 2])
plt.xlim([-3, 3])

plt.subplot(2, 2, 2)
plt.title('Sparse GP')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_sparse_gpy)
error = np.squeeze(2 * np.sqrt(y_var_sparse_gpy))
plt.plot(x_true, y, '-', color='C2', label='sparse GP')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)

plt.plot(np.squeeze(sparse_gpy.inducing_inputs), -1.5 * np.ones((M, )), 'o',
         color='C3', label='inducing inputs')
# plt.xlabel('input $x$')
# plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-2, 2])
plt.xlim([-3, 3])


# plt.subplot(2, 2, 3)
# plt.title('NIGP')
# plt.plot(x_data, y_data, '+', label='data points')
# plt.plot(x_true, y_true, '-', label='true function')
#
# y = np.squeeze(y_mean_nigpy)
# error = np.squeeze(2 * np.sqrt(y_var_nigpy))
# plt.plot(x_true, y, '-', color='C2', label='NIGP')
# plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)
#
# plt.grid()
# plt.legend(loc=2)
# plt.ylim([-2, 2])
# plt.xlim([-3, 3])




plt.subplot(2, 2, 4)
plt.title('PAC GP')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_pac_nigp)
error = np.squeeze(2 * np.sqrt(y_var_pac_nigp))
plt.plot(x_true, y, '-', color='C2', label='sparse GP')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)

# plt.plot(np.squeeze(z_gpy), -1.5 * np.ones((M, )), 'o',
#          color='C3', label='GPy inducing inputs')
# plt.plot(np.squeeze(Z_opt), -1.5 * np.ones((M, )), 'o',
#          color='C4', label='PAC GP inducing inputs')
# plt.xlabel('input $x$')
# plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-2, 2])
plt.xlim([-3, 3])
plt.show()
