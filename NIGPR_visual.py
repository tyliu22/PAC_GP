"""
    This code is copy from:
    https://github.com/krasserm/bayesian-machine-learning/blob/dev/gaussian-processes/gaussian_processes.ipynb
    Theory: http://krasserm.github.io/2018/03/19/gaussian-processes/
fgh
    Setting:
        mean function of GPR is selected as X, which is the same as
"""

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

# scikit learn package
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# GPy
import GPy

def plot_gp(mu, cov, X, X_train=None, Y_train=None, titles='GPR plot'):
    # X: test data:  X_train and y_train: training data and labels
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    # for i, sample in enumerate(samples):
    #     plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.title(titles)
    plt.show()


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    # temp1 = np.sum(X1**2, 1).reshape(-1, 1)
    # temp2 = np.sum(X2**2, 1)
    # temp15 = temp1+temp2
    # temp3 = 2 * np.dot(X1, X2.T)
    # temp = temp1 + temp2 - temp3
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


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
        temp = -2 * tf.matmul(X1, X1, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
    else:
        X2 = X2 / l
        X2s = tf.reduce_sum(tf.square(X2), 1)
        temp = -2 * tf.matmul(X1, X2, transpose_b=True) + \
               tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    return sigma_f**2 * tf.exp(-temp / 2)


def NIGP_K_grad_mean(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1e-8):
    # calculate the derivative of the mean function with \tlide{x}
    VAR_X_train = tf.placeholder(tf.float64, shape=X_train.shape)
    K = kernel_Tensor(X1=VAR_X_train, X2=VAR_X_train, l=l, sigma_f=sigma_f) + sigma_y ** 2 * tf.eye(len(X_train), dtype=tf.float64)
    X_s_Tensor = tf.convert_to_tensor(X_s, dtype=tf.float64)
    K_s = kernel_Tensor(X1=VAR_X_train, X2=X_s_Tensor, l=l, sigma_f=sigma_f)
    # K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = tf.linalg.inv(K)
    # mu_s_old = K_s.T.dot(K_inv).dot(Y_train)
    mu_s_old = tf.matmul(tf.matmul(K_s, K_inv, transpose_a=True), Y_train)
    var_grad = tf.gradients(mu_s_old, VAR_X_train)
    with tf.Session() as sess:
        grad_posterior_mean = sess.run(var_grad, feed_dict={VAR_X_train: X_train})

    # calculate the mean function of posterior distribution
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (7): posterior distribution mean
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    SIGMA_x = sigma_x**2 * np.eye(len(X_train))
    K_nigp = K + grad_posterior_mean[0].T.dot(SIGMA_x).dot(grad_posterior_mean[0])

    return K_nigp


def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, Non_zero_mu=True):
    """
    Computes the suffifient statistics of the posterior distribution
    from m training data X_train and Y_train and n new inputs X_s.
 -
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

    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    if Non_zero_mu==True:
        mu_s = X_s + K_s.T.dot(K_inv).dot(Y_train - X_train)
    else:
        mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def NIGP_posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution
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

    # calculate the derivative of the mean function with \tlide{x}
    VAR_X_train = tf.placeholder(tf.float64, shape=X_train.shape)
    VAR_X_s = tf.placeholder(tf.float64, shape=X_s.shape)

    K = kernel_Tensor(X1=VAR_X_train, X2=VAR_X_train, l=l, sigma_f=sigma_f) \
        + sigma_y ** 2 * tf.eye(len(X_train), dtype=tf.float64)
    X_s_Tensor = tf.convert_to_tensor(X_s, dtype=tf.float64)
    K_s = kernel_Tensor(X1=VAR_X_train, X2=X_s_Tensor, l=l, sigma_f=sigma_f)
    # K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = tf.linalg.inv(K)
    # mu_s_old = K_s.T.dot(K_inv).dot(Y_train)
    mu_s_old = tf.matmul(tf.matmul(K_s, K_inv, transpose_a=True), Y_train)
    var_grad = tf.gradients(mu_s_old, VAR_X_train)
    with tf.Session() as sess:
        grad_posterior_mean = sess.run(var_grad, feed_dict={VAR_X_train: X_train})

    # calculate the mean function of posterior distribution
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (7): posterior distribution mean
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # SIGMA_x = sigma_x**2 * np.eye(len(X_train)) # Here  dimension should be D = 0
    SIGMA_x = sigma_x ** 2
    reg_grad = grad_posterior_mean[0].dot(SIGMA_x).dot(grad_posterior_mean[0].T)
    # reg_grad_diag = tf.linalg.band_part(reg_grad, len(X_train)-2, 0)
    K_nigp = K + reg_grad
    K_nigp_inv = inv(K_nigp)

    # NIGP posterior distribution mean and covariance
    mu_s_nigp = K_s.T.dot(K_nigp_inv).dot(Y_train)
    cov_s_nigp = K_ss - K_s.T.dot(K_nigp_inv).dot(K_s)

    return mu_s_nigp, cov_s_nigp


def nll_fn(X_train, Y_train, noise, naive=True):
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
        # Naive implementation of Eq. (11). Works well for the examples
        # in this article but is numerically less stable compared to
        # the implementation in nll_stable below.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise ** 2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * (Y_train-X_train[0]).dot(inv(K).dot( (Y_train-X_train[0]) )) + \
               0.5 * len(X_train) * np.log(2 * np.pi)

    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.

        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise ** 2 * np.eye(len(X_train))
        L = cholesky(K)

        S1 = solve_triangular(L, Y_train, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)

        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.dot(S2) + \
               0.5 * len(X_train) * np.log(2 * np.pi)

    if naive:
        return nll_naive
    else:
        return nll_stable


def nll_fn_nigp(X_train, Y_train, noise):

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
        # Naive implementation of Eq. (11). Works well for the examples
        # in this article but is numerically less stable compared to
        # the implementation in nll_stable below.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise ** 2 * np.eye(len(X_train))
        # K_nigp =
        return 0.5 * np.log(det(K)) + \
               0.5 * Y_train.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2 * np.pi)
    return nll_naive


def prediction(X_test, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1e-8):
    mu_nigp_pred, cov_nigp_pred = NIGP_posterior(X_test, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, sigma_x=1e-8)
    return mu_nigp_pred, cov_nigp_pred

# def mean_func(x, I=np.identity(x.shape[0])):
#     return I.dot(x)

# ********************************************************************************* #
#                                                                                   #
# ********************************************************************************* #
"""
    tf.Gradient() example test 
"""
# def mul(x, y):
#     return np.sum(x+y) + 2*x*x*y
#
# x_con = [4.0, 1.0]
# x = tf.placeholder(tf.float32)
# y = 2.0
# result = mul(x,y)
# var_grad = tf.gradients(result,x)
# with tf.Session() as sess:
#     var_grad_val = sess.run(var_grad, feed_dict={x:x_con})
# print(var_grad_val)
"""
    Prediction from noisy training data
"""
Non_zero_mu = True

# Finite number of points
X = np.arange(-5, 5, 0.2).reshape(-1, 1)

# PRIOR DISTRIBUTION: Mean and covariance of the prior
if Non_zero_mu == True:
    # mu_prior = np.identity(X.shape[0]).dot(X)
    mu_prior = X
else:
    mu_prior = np.zeros(X.shape)
cov_prior = kernel(X, X)


# std: covariance of noisy_X and noisy_y
noise_y = 0.4
noise_x = 0.2

# Noisy training data
X_train = np.arange(-3, 4, 0.5).reshape(-1, 1)
Y_train = np.sin(X_train) + noise_y * np.random.randn(*X_train.shape)
X_train_obs = X_train + noise_x * np.random.randn(*X_train.shape)
X_train = X_train_obs

# samples = np.random.multivariate_normal(mu_prior.ravel(), cov_prior, 3)
plot_gp(mu_prior, cov_prior, X, X_train=X_train, Y_train=Y_train, titles='GPR prior dis init_paras')
# Compute mean and covariance of the posterior distribution
# mu_s, cov_s = posterior(X, X_train, Y_train, sigma_y=noise_y)
# GPR posterior distribution
mu_post_GPR, cov_post_GPR = posterior(X, X_train, Y_train, sigma_y=noise_y)
# mu_post_NIGP, cov_post_NIGP = NIGP_posterior(X, X_train, Y_train, sigma_y=noise_y, sigma_x=noise_x)

# plot: posterior dis of training dataset, in terms  o f GPR and -NIGPR
# samples = np.random.multivariate_normal(mu_post_GPR.ravel(), cov_post_GPR, 3)
plot_gp(mu_post_GPR, cov_post_GPR, X, X_train=X_train, Y_train=Y_train, titles='GPR post init_paras')

# samples = np.random.multivariate_normal(mu_post_NIGP.ravel(), cov_post_NIGP, 3)
# plot_gp(mu_post_NIGP, cov_post_NIGP, X, X_train=X_train, Y_train=Y_train, samples=samples, titles='NIGPR post init_paras')


# NIGP optimization
# Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
# We should actually run the minimization several times with different
# initializations to avoid local minima but this is skipped here for
# simplicity. x0 = [1,1]
res = minimize(nll_fn(X_train, Y_train, noise_y), [1, 1],
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')

# Store the optimization results in global variables so that we can
# compare it later with the results from other implementations.
# hyperparameters: lengthscale - l_opt and sigma_f - sigma_f_opt
l_opt, sigma_f_opt = res.x

# Compute posterior mean and covariance with optimized kernel parameters and plot the results
mu_post_GPR_fit, cov_post_GPR_fit = posterior(X, X_train, Y_train, l=l_opt,
                                              sigma_f=sigma_f_opt, sigma_y=noise_y)
plot_gp(mu_post_GPR_fit, cov_post_GPR_fit, X, X_train=X_train, Y_train=Y_train, titles='GPR post fit paras')


mu_post_NIGP_fit, cov_post_NIGP_fit = NIGP_posterior(X, X_train, Y_train, l=l_opt,
                                                     sigma_f=sigma_f_opt, sigma_y=noise_y, sigma_x=noise_x)
plot_gp(mu_post_NIGP_fit, cov_post_NIGP_fit, X, X_train=X_train, Y_train=Y_train, titles='NIGPR post fit paras')

print('End')









