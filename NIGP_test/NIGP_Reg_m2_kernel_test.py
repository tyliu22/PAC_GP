
"""
    Function:
        Other implementation approach of Noisy Input Gaussian Processing (NIGP):
        two steps:
            1) Calculate the standard GP mean function and the slope of mean funciton;
            2) calculate the NIGP (GP with a regularization item )

    Setting:
        Noisy input and noisy output
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import det
from scipy.optimize import minimize
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib

from NIGP_test.NIGP_reg_standard.kernels import RBF


# plot the mean and covariance of GPR with testing data
def plot_gp(mu, cov, X, X_train=None, Y_train=None, titles='GPR plot', Fig_path=None):
    # X: test data:  X_train and y_train: training data and labels

    plt.figure(figsize=(6, 4))
    plt.style.use('bmh')
    fontsize = 11
    fontfamily = 'Times New Roman'
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['font.family'] = fontfamily

    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.3)
    plt.plot(X, mu, label='Estimation')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx', label='Real data')

    plt.legend(loc='best')
    plt.xlabel('Test input')
    plt.ylabel('Output')
    if Fig_path is not None:
        plt.savefig(fname=Fig_path+titles+'.pdf', format="pdf")
    plt.show()


def plot_gp_f(mu, cov, X_train=None, Y_train=None, titles='GPR function f'):
    """
        Plot the GP function f, calculated only use training data without testing data
    """
    #X_train and y_train: training data and labels
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.scatter(X_train.ravel(), mu)
    plt.scatter(X_train.ravel(), mu + uncertainty)
    plt.scatter(X_train.ravel(), mu - uncertainty)
    # plt.fill_between(X_train.ravel(), mu + uncertainty, mu - uncertainty, alpha=0.1)
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx', label='TrainingData')
    # plt.plot(X_train.ravel(), mu, label='Data2f_mean')
    # plt.plot(X_train.ravel(), mu, 'bo', label='Data2f_mean')
    plt.legend()
    plt.title(titles)
    plt.show()


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
def kernel_Tensor(X1, X2=None, l=1.0, sigma_f=1.0, input_dim=None):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """

    if input_dim is not None:
        l = np.ones(input_dim)
        shape = (input_dim, )


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



def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8, Non_zero_mu=False):
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




# ********************************************************************************* #
# Start: loading data                                                #
# ********************************************************************************* #
if __name__ == "__main__":

    np.random.seed(9527)
    Fig_path = '/Users/tianyuliu/PycharmProjects/PAC_GP/Results_fig/'

    # ****************************************** #
    # Data preparing                             #
    # ****************************************** #
    X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
    Y_test = np.sin(X_test)


    # std: covariance of noisy_X and noisy_y
    noise_y = 0.1  # Output noisy std
    noise_x = 0.4  # Iutput noisy std

    # New noisy training data
    X_train = np.linspace(-10, 10, 150).reshape(-1, 1)
    Y_train = np.sin(X_train) + noise_y * np.random.randn(*X_train.shape)
    X_train_obs = X_train + noise_x * np.random.randn(*X_train.shape)
    X_train = X_train_obs

    # ---- intialization parameter ---- #
    l       = 1.0
    sigma_f = 1.0
    sigma_x = noise_x
    sigma_y = noise_y

    # ***************************** GPR with fitted paras **************************** #

    # GPR or NIGPR posterior
    # mu_post_GPR_init, cov_post_GPR_init = posterior(X_test, X_train, Y_train, sigma_y=sigma_y)
    # plot_gp(mu_post_GPR_init, cov_post_GPR_init, X_test, X_train=X_train, Y_train=Y_train,
    #         titles='Reproduction_GPR_post_init_paras', Fig_path=Fig_path)

    mu_s_nigp_init, cov_s_nigp_init = NIGP_posterior(X_test, X_train, Y_train, sigma_y=noise_y, sigma_x=noise_x)
    plot_gp(mu_s_nigp_init, cov_s_nigp_init, X_test, X_train=X_train, Y_train=Y_train,
            titles='Reproduction_NIGPR_post_init_paras', Fig_path=Fig_path)

    # Performance analysis
    # MSE_GPR_init   = 0.5 * np.sum((mu_post_GPR_init - Y_test)**2)
    MSE_NIGPR_init = 0.5 * np.sum((mu_s_nigp_init   - Y_test)**2)
    # print('Init para MSE error:  GPR ', MSE_GPR_init, 'NIGPR  ', MSE_NIGPR_init)

    f_grad_mean = NIGP_f_grad_mean(X_train, Y_train, l, sigma_f, sigma_y)
    SIGMA_x = sigma_x ** 2 * np.eye(len(X_train[1]))
    NIGP_matrix_init_para = np.diag(np.diag(f_grad_mean.dot(SIGMA_x).dot(f_grad_mean.T)))

    # ****************************************** #
    # Parameters optimization                    #
    # ****************************************** #

    # Optimize the nll loss (used in GPR), without regularization item, slope of mean func
    # res = minimize(nll_fn(X_train, Y_train, noise_y), [1, 1],
    #                bounds=((1e-5, None), (1e-5, None)),
    #                method='L-BFGS-B',
    #                tol=1e-12,
    #                options={'disp': False, 'eps': 0.001}
    #                )
    # l_opt, sigma_f_opt = res.x

    # Optimize the NIGP nll loss (used in NIGPR), with regularization item, slope of mean func
    res = minimize(nll_fn_nigp(X_train, Y_train, noise_y, NIGP_matrix_init_para), [1, 1],
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B',
                   tol=1e-12,
                   options={'disp': False, 'eps': 0.001}
                   )
    l_opt, sigma_f_opt = res.x

    # ****************************************** #
    # Posterior distribution                     #
    # ****************************************** #
    # mu_post_GPR_fit, cov_post_GPR_fit = posterior(X_test, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=sigma_y)
    # plot_gp(mu_post_GPR_fit, cov_post_GPR_fit, X_test, X_train=X_train, Y_train=Y_train,
    #         titles='Reproduction_GPR_post_fit_paras', Fig_path=Fig_path)

    mu_s_nigp_fit, cov_s_nigp_fit = NIGP_posterior(X_test, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_y, sigma_x=noise_x)
    plot_gp(mu_s_nigp_fit, cov_s_nigp_fit, X_test, X_train=X_train, Y_train=Y_train,
            titles='Reproduction_NIGPR_post_fit_paras', Fig_path=Fig_path)

    # Performance analysis
    # MSE_GPR_fit   = 0.5 * np.sum((mu_post_GPR_fit - Y_test)**2)
    MSE_NIGPR_fit = 0.5 * np.sum((mu_s_nigp_fit   - Y_test)**2)
    # print('Fitted para MSE error:  GPR ', MSE_GPR_fit, 'NIGPR  ', MSE_NIGPR_fit)


    print('End')









