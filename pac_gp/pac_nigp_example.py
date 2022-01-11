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

# import torch

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.optimize import minimize
import tensorflow as tf
# import tensorflow.compat.v1 as tf

from gp.mean_functions import Zero
from gp.kerns import RBF
from gp.pac_nigp import NIGP_PAC_HYP_GP

from utils.data_generator import generate_sin_data
from tensorflow.contrib.distributions.python.ops.mvn_full_covariance import MultivariateNormalFullCovariance as MVN
from tensorflow.python.ops.distributions.kullback_leibler import kl_divergence as KL



# %% Configuration
np.random.seed(11)
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
    K_reg = K + sigma_y ** 2 *  tf.eye(len(X_train), dtype=tf.float64)
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

    K_Xtr_Xtr = kernel(X_train, X_train, l=l, sigma_f=sigma_f)
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




Fig_path = None

noise_y = 0.4  # Output noisy std
noise_x = 0.5  # Iutput noisy std

# ---- intialization parameter ---- #
l       = 1.0
sigma_f = 1.0
sigma_x = 0.4 # std, variance= sigma_x**2
sigma_y = 0.5 # std, variance= sigma_y**2
# (x_data, y_data) (x_true, y_true), noise_input_variance
f_grad_mean = NIGP_f_grad_mean(x_data, y_data, l, sigma_f, sigma_y)
SIGMA_x = sigma_x ** 2 * np.eye(len(x_data[1]))
NIGP_matrix_init_para = np.diag(np.diag(f_grad_mean.dot(SIGMA_x).dot(f_grad_mean.T)))

P_mean_nigp = np.zeros(x_data.shape)
P_cov_nigp = kernel(x_data, x_data, l=l, sigma_f=sigma_f)

# Optimize the NIGP nll loss (used in NIGPR), with regularization item, slope of mean func
res = minimize(nll_fn_nigp(x_data, y_data, noise_y, NIGP_matrix_init_para), [1, 1],
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B',
                   tol=1e-12,
                   options={'disp': False, 'eps': 0.001}
                   )
l_opt, sigma_f_opt = res.x

mu_s_nigp_fit, cov_s_nigp_fit = NIGP_posterior(x_true, x_data, y_data, l=l_opt, sigma_f=sigma_f_opt,
                                                   sigma_y=noise_y, sigma_x=noise_x)
Q_mean_nigp, Q_cov_nigp = NIGP_posterior(x_data, x_data, y_data, l=l_opt, sigma_f=sigma_f_opt,
                                                   sigma_y=noise_y, sigma_x=noise_x)


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.


    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term = np.trace(iS1.dot(S0))
    det_term = np.log(np.linalg.det(S1) / np.linalg.det(S0))  # np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T.dot(np.linalg.inv(S1)).dot(diff)  # np.sum( (diff*diff) * iS1, axis=1)
    # print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N)

# KL_divergence_nigp = kl_mvn(P_mean_nigp, P_cov_nigp, Q_mean_nigp, Q_cov_nigp)


def cal_KL_divergence(P_mean_, P_cov_, Q_mean_, Q_cov_):
    P_mean = tf.placeholder(tf.float32, shape=P_mean_.shape)
    P_cov = tf.placeholder(tf.float32, shape=P_cov_.shape)
    Q_mean = tf.placeholder(tf.float32, shape=Q_mean_.shape)
    Q_cov = tf.placeholder(tf.float32, shape=Q_cov_.shape)

    P_dist = MVN(loc=P_mean,
                 covariance_matrix=P_cov,
                 name='prior')
    Q_dist = MVN(loc=Q_mean,
                 covariance_matrix=Q_cov,
                 name='posterior')

    # KL(Q_dist||P_dist)
    KL_divergence = KL(Q_dist, P_dist)

    with tf.Session() as sess:
        KL_divergence = sess.run(KL_divergence, feed_dict={P_mean:P_mean_,
                                                         P_cov:P_cov_,
                                                         Q_mean:Q_mean_,
                                                         Q_cov:Q_cov_})

    return KL_divergence[0]


def compute_kl(P_mean_nigp, P_cov_nigp, Q_mean_nigp, Q_cov_nigp):
    # Compute KL divergence between two Gaussians (self and other)
    # (refer to the paper)
    # b is the variance of priors
    b1 = torch.pow(self.sigma, 2)
    b0 = torch.pow(other.sigma, 2)

    term1 = torch.log(torch.div(b0, b1))
    term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
    term3 = torch.div(b1, b0)
    kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
    return kl_div

# KL_divergence_nigp = cal_KL_divergence(P_mean_nigp, P_cov_nigp, Q_mean_nigp, Q_cov_nigp)


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
pac_risk  = pac_nigp.empirical_risk
pac_kl    = pac_nigp.KL_divergence
pac_bound = pac_nigp.upper_bound




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

print("GP error: " + str(mean_squared_error(y_true, y_mean_full_gpy)))
print("sparse GP error: " + str(mean_squared_error(y_true, y_mean_sparse_gpy)))
print("NIGP error: " + str(mean_squared_error(y_true, mu_s_nigp_fit)))
print("PAC-NIGP error: " + str(mean_squared_error(y_true, y_mean_pac_nigp)))



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


plt.subplot(2, 2, 3)
plt.title('NIGP')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(mu_s_nigp_fit)
error = np.squeeze(2 * np.sqrt(np.diag(cov_s_nigp_fit)))
plt.plot(x_true, y, '-', color='C2', label='NIGP')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)

plt.grid()
plt.legend(loc=2)
plt.ylim([-2, 2])
plt.xlim([-3, 3])




plt.subplot(2, 2, 4)
plt.title('PAC-NIGP')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_pac_nigp)
error = np.squeeze(2 * np.sqrt(y_var_pac_nigp))
plt.plot(x_true, y, '-', color='C2', label='PAC-NIGP')
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
plt.savefig('PAC_NIGP_single_dim_COMP.pdf')
plt.show()

