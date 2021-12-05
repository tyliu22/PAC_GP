
"""
    Function:
        motivation, why should we cansider the Noisy Input in Gaussian Processing (NIGP):

        The influence of noisy input on standard GP algorithms

    Setting:
        Noisy input with OR without?
        Running very slowly
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
import matplotlib

# scikit learn package
from sklearn.metrics import mean_squared_error as mse


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
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[], title='None'):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.title(title)
    plt.show()


def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
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
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


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
               0.5 * Y_train.dot(inv(K).dot(Y_train)) + \
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




if __name__ == "__main__":

    np.random.seed(0)
    Fig_path = '/Users/tianyuliu/PycharmProjects/PAC_GP/Results_fig/'

    # Noise free training data
    X_train_ori = np.arange(-3, 4, 0.1).reshape(-1, 1)
    Y_train_ori = np.sin(X_train_ori)

    X = np.arange(-5, 5, 0.2).reshape(-1, 1)

    noise_y = 0.2
    noise_x = 0.3
    Y_train_out = Y_train_ori + noise_y * np.random.randn(*X_train_ori.shape)
    X_train_in = X_train_ori + noise_x * np.random.randn(*X_train_ori.shape)


    noise_x_set = [0.0, 0.5]
    noise_y_set = [0.1, 0.2, 0.3, 0.4, 0.5]

    """
        Prediction from output noise training data
    """
    print('Start GPR with output noisy')

    Ana_output_noise = np.zeros([2, 2, 5]) # mean and std

    for temp_noise_x in range(2):

        noise_x = noise_x_set[temp_noise_x]
        print(noise_x)
        X_train_in = X_train_ori + noise_x * np.random.randn(*X_train_ori.shape)

        for temp_noise_y in range(5):

            noise_y = noise_y_set[temp_noise_y]
            error_input_0 = np.zeros([5])

            for i in range(5):
                Y_train_out = Y_train_ori + noise_y * np.random.randn(*X_train_ori.shape)

                res = minimize(nll_fn(X_train_in, Y_train_out, noise_y), [1, 1],
                               bounds=((1e-5, None), (1e-5, None)),
                               method='L-BFGS-B') # maxiter=20

                l_opt, sigma_f_opt = res.x
                print(i)
                # Compute posterior mean and covariance with optimized kernel parameters and plot the results
                mu_s, cov_s = posterior(X, X_train_ori, Y_train_out, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_y)
                # plot_gp(mu_s, cov_s, X, X_train=X_train_ori, Y_train=Y_train_ori, title='GPR with output noisy')
                error_input_0[i] = (mse(mu_s, np.sin(X)))

            Ana_output_noise[temp_noise_x, :, temp_noise_y] = [np.mean(error_input_0), np.std(error_input_0)]


    plt.figure(figsize=(6, 4))
    plt.style.use('bmh')
    fontsize = 11
    fontfamily = 'Times New Roman'
    matplotlib.rcParams['font.size'] = fontsize
    matplotlib.rcParams['font.family'] = fontfamily
    plt.plot(noise_y_set, Ana_output_noise[0, 0, :], label='Without input noisy')
    plt.fill_between(noise_y_set, Ana_output_noise[0, 0, :] - Ana_output_noise[0, 1, :],
                     Ana_output_noise[0, 0, :] + Ana_output_noise[0, 1, :],
                     color='gray', alpha=0.2)
    plt.plot(noise_y_set, Ana_output_noise[1, 0, :], label='With input noisy')
    plt.fill_between(noise_y_set, Ana_output_noise[1, 0, :] - Ana_output_noise[1, 1, :],
                     Ana_output_noise[1, 0, :] + Ana_output_noise[1, 1, :],
                     color='gray', alpha=0.2)
    # plt.legend(['With input noisy', 'Without input noisy'], loc='best')
    plt.legend(loc='best')
    plt.xlabel('Output noisy level (std)')
    plt.ylabel('MSE error')

    plt.savefig(fname="Motivation_NIGPR.pdf",format="pdf")
    plt.show()

    print('GPR with output noisy')

    # Output results
    # noise_x_set = [0.0, 0.5]
    # noise_y_set = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Ana_output_noise = np.array([[[0.01402519, 0.06972807, 0.05729557, 0.09332338, 0.11905276],
    #                      [0.01039965, 0.05913818, 0.05114749, 0.04139018, 0.02466474]],
    #                     [[0.18171676, 0.08312128, 0.10532905, 0.12828164, 0.15591512],
    #                     [0.04728107, 0.02522427, 0.04025637, 0.05323896, 0.02952605]]])
    #

    print('End')

    # """
    #     Prediction from input and output noisy training data
    # """
    # # ************************ GPR with input and output noisy *************************** #
    #
    # res = minimize(nll_fn(X_train_in, Y_train_out, noise_y), [1, 1],
    #                bounds=((1e-5, None), (1e-5, None)),
    #                method='L-BFGS-B')
    # l_opt, sigma_f_opt = res.x
    #
    # # Compute posterior mean and covariance with optimized kernel parameters and plot the results
    # mu_s, cov_s = posterior(X, X_train_in, Y_train_out, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_y)
    # plot_gp(mu_s, cov_s, X, X_train=X_train_ori, Y_train=Y_train_ori, title='GPR with input and output noisy')





#
# # *************************** Higher dimensions ******************88 #
# """
#     Higher dimensions:
#         The above implementation can also be used for higher input data dimensions. Here, a GP is used to fit noisy
#         samples from a sine wave originating at  0  and expanding in the x-y plane.
# """
# def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
#     ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
#     ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
#     ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
#     ax.set_title(title)
#     plt.show()
#
# noise_2D = 0.1
#
# rx, ry = np.arange(-5, 5, 0.3), np.arange(-5, 5, 0.3)
# gx, gy = np.meshgrid(rx, rx)
#
# X_2D = np.c_[gx.ravel(), gy.ravel()]
#
# X_2D_train = np.random.uniform(-4, 4, (100, 2))
# Y_2D_train = np.sin(0.5 * np.linalg.norm(X_2D_train, axis=1)) + \
#              noise_2D * np.random.randn(len(X_2D_train))
#
# plt.figure(figsize=(14,7))
#
# mu_s, _ = posterior(X_2D, X_2D_train, Y_2D_train, sigma_y=noise_2D)
# plot_gp_2D(gx, gy, mu_s, X_2D_train, Y_2D_train,
#            f'Before parameter optimization: l={1.00} sigma_f={1.00}', 1)
#
# res = minimize(nll_fn(X_2D_train, Y_2D_train, noise_2D), [1, 1],
#                bounds=((1e-5, None), (1e-5, None)),
#                method='L-BFGS-B')
#
# mu_s, _ = posterior(X_2D, X_2D_train, Y_2D_train, *res.x, sigma_y=noise_2D)
# plot_gp_2D(gx, gy, mu_s, X_2D_train, Y_2D_train,
#            f'After parameter optimization: l={res.x[0]:.2f} sigma_f={res.x[1]:.2f}', 2)
#
#
# """
#     **************************************** Libraries that implement GPs *****************************************
# """
#     # *************************************** Scikit-learn **************************************** #
#
#
#
# rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
# gpr = GaussianProcessRegressor(kernel=rbf, alpha=noise**2)
#
# # Reuse training data from previous 1D example
# gpr.fit(X_train, Y_train)
#
# # Compute posterior mean and covariance
# mu_s, cov_s = gpr.predict(X, return_cov=True)
#
# # Obtain optimized kernel parameters
# l = gpr.kernel_.k2.get_params()['length_scale']
# sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])
#
# # Compare with previous results
# assert(np.isclose(l_opt, l))
# assert(np.isclose(sigma_f_opt, sigma_f))
#
# # Plot the results
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
#
#
#
#
#
# # ****************************************************** GPy ********************************************************* #
#
# rbf = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
# gpr = GPy.models.GPRegression(X_train, Y_train, rbf)
#
# # Fix the noise variance to known value
# gpr.Gaussian_noise.variance = noise**2
# gpr.Gaussian_noise.variance.fix()
#
# # Run optimization
# gpr.optimize();
#
# # Display optimized parameter values
# # display(gpr)
#
#
# # Obtain optimized kernel parameters
# l = gpr.rbf.lengthscale.values[0]
# sigma_f = np.sqrt(gpr.rbf.variance.values[0])
#
# # Compare with previous results
# assert(np.isclose(l_opt, l))
# assert(np.isclose(sigma_f_opt, sigma_f))
#
# # Plot the results with the built-in plot function
# gpr.plot()
#
#
#
# mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise)
#
# # Include noise into predictions using Equation (9).
# cov_s = cov_s + noise**2 * np.eye(*cov_s.shape)
#
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
#
# print("End")
#
#
#
#
