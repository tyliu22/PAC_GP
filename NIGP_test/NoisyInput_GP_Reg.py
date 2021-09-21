# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:21:25 2016
@author: Aleksander

Paper implementation : "Gaussian Process Training with Input Noise"
Parameters:
    var_x: input noise variance
    var_y: output noise variance
    varf : kernal hyperparameter sigma_f
    l    : kernal hyperparameter lengthscale   or   parameter set
    K    :
    k1   :
    k2   :

"""

import numpy as np
import matplotlib.pyplot as plt

import math
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize

def plot_gp(mu, std, X, X_train=None, Y_train=None, title='NIGP'): # samples=[]
    X = X.ravel()
    mu = mu.ravel()
    # uncertainty = 1.96 * np.sqrt(np.diag(cov))
    uncertainty = 1.96 * std

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Estimation')
    # for i, sample in enumerate(samples):
    #     plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if X_train is not None:
        plt.plot(Xcv, sincsig(Xcv), 'rx', label='Real data')
    plt.legend()
    plt.title(title)
    plt.show()

class GPRegressor:

    # l - Correlation length. Scalar or array of length num_features in X
    # varf - Signal variance.
    # normalize_y - Subtract mean of y from y if True. Often a good idea since
    #               we are modeling a zero-mean GP.
    # num_restarts_hyper - Number of randomly initialized optimizations of the hyperparameters,
    #                   that should be performed. More than one is usually a good idea since
    #                   the log_marinal_likelihood often has local maxima.
    #
    def __init__(self, l=1.0, varf=1.0, normalize_y=True, num_restarts_hyper=1):

        self.varf = varf
        self.normalize_y = normalize_y
        self.num_restarts = num_restarts_hyper
        self.l = np.array(l)

    # X - Input prediction data
    # return_std - Returns GP std.dev. if True
    # np.zeros_like(M): create zero-matrix with the same shape as M
    def predict(self, X, return_std=False):
        l = self.l
        # [K2]: k(x, X);     [self.pred_vec]: K(X,X)+sigma_y*I + sigma_x_regularization
        self.K2 = self.kernel(X, self.X_train, np.zeros_like(X), self.var_x, l, self.varf)
        # k(x,X) * inv( K(X,X)+sigma_y*I + sigma_x_regul ) * y
        mean = np.dot(self.K2, self.pred_vec) + self.mu
        if return_std:
            # See equation (7) in paper NIGP:
            std2 = np.sqrt(np.diag(
                self.autokernel(X, np.zeros_like(X), l, self.varf)
                - np.dot(self.K2, np.dot(self.pred_fac, self.K2.T)) ))
            return mean, std2
        else:
            return mean, 0.0

    def kernel(self, X1, X2, var_x1, var_x2, l, varf): # K(x_star,X)  kernal func
        # squared exponential kernel with Automatic Relevance Determination
        tmp = 0.0
        tmp2 = 1.0
        l = l * np.ones(len(self.X_train[0, :]))
        for i in range(0, len(X1[0, :])):
            l2 = l[i] * l[i]  # !
            d1 = cdist(X1[:, i].reshape(-1, 1), X2[:, i].reshape(-1, 1), metric='sqeuclidean')
            d2 = cdist(var_x1[:, i].reshape(-1, 1), -var_x2[:, i].reshape(-1, 1), metric='euclidean')
            tmp += d1 / (l2 + d2)
            tmp2 *= (1.0 + d2 / l2)
        return varf * np.power(tmp2, -0.5) * np.exp(-0.5 * tmp)

    def autokernel(self, X, var_x, l, varf): # (self.X_train, self.var_x, l, varf)
        # Equation 9 in NIGP
        tmp = 0.0
        tmp2 = 1.0
        l = l * np.ones(len(self.X_train[0, :])) # diagonal matrix of the kernal
        for i in range(0, len(X[0, :])):
            l2 = l[i] * l[i]  # Lambda : square of lengthscale of kernal
            d1 = cdist(X[:, i].reshape(-1, 1), X[:, i].reshape(-1, 1), metric='sqeuclidean') # ||x-x'||^2_2  (N*N)
            d2 = cdist(var_x[:, i].reshape(-1, 1), -var_x[:, i].reshape(-1, 1), metric='euclidean') # (N*N) Sigma_x variance of input
            tmp += d1 / (l2 + d2) # (N*N)
            tmp2 *= (1.0 + d2 / l2) # (N*N) sigma_f^2 * sqrt[Sigma_x in v(Lambda) + I] * exp(-0.5*(x_i-x_j))
        return varf * np.power(tmp2, -0.5) * np.exp(-0.5 * tmp)

        # X_train - Input data (num_samples, num_features)

    # y_train - Output data (num_samples)
    # var_x - Variance in input points x
    # var_y - Variance in output points y
    # l_bounds - Bounds for the hyperparameters.
    #          Array size (3x2) or ((2+num_features)x2).
    #          l[0] is varf(signal variance), l[1] is noise_variance
    #          and l[2::] is correlation length(s)  <--->  res here？
    #          None means using the supplied hyperparameters.
    def fit(self, X_train, y_train, var_x, var_y, l_bounds=None):

        if self.normalize_y:
            self.mu = np.mean(y_train, 0)
        else:
            self.mu = 0.0
        self.X_train = X_train
        self.y_train = (y_train - self.mu)
        #
        if np.iterable(var_x):
            self.var_x = var_x
        else:
            self.var_x = var_x * np.ones_like(X_train)

        self.var_y = var_y

        # Fit hyperparameters by maximizing log marginal likelihood.
        if l_bounds is not None:
            bounds = []
            for i in range(0, len(l_bounds)):
                bounds.append(l_bounds[i])
            best_f = 1e6
            for j in range(0, self.num_restarts):
                loglb = np.log10(l_bounds[:, 0])
                loghb = np.log10(l_bounds[:, 1])
                l0 = loglb + (loghb - loglb) * np.random.random(size=loglb.shape)
                l0 = 10.0 ** l0

                res = minimize(self.neg_log_marginal_likelihood, l0, method='l-bfgs-b', bounds=bounds, tol=1e-12,
                               options={'disp': False, 'eps': 0.001})
                # res: final optimized parameters; l0 is the initialize parameter
                if res['fun'] < best_f:
                    self.varf = res['x'][0]    # output variance
                    self.alpha = res['x'][1]   # varn in nlml function
                    self.l = res['x'][2::]     # lengthscale parameter for kernal
                    self.opt_params = res['x'] #
                # print("iter: " + str(j) + ". params: " + "Output variance：" + str(self.varf) +
                      # ", " + str(self.alpha) + ", " + str(self.l))
            self.var_y += self.alpha
        # Calculate factors needed for prediction.
        # K1 = K(X,X)
        self.K1 = self.autokernel(self.X_train, self.var_x, self.l, self.varf)
        # self.pred_fac:
        self.pred_fac = np.linalg.pinv(self.K1 + np.identity(len(self.K1[:, 0])) * self.var_y)
        # self.pred_vec:
        self.pred_vec = np.dot(self.pred_fac, self.y_train)

    def neg_log_marginal_likelihood(self, l):
        # This nlml three hyperparameters: SEE (2.31) in book GPML
        varf = l[0] # output variance
        varn = l[1] # input noise introduced variance, total variance
        l = l[2::]  # lengthscale parameter for kernal
        # calculate the kernal matrix: K(X,X) + sigma^2 * I + varn'

        # varn or l[1]: (slope of mean func)^T * Sigma_x * (slope of mean func)
        K = self.autokernel(self.X_train, self.var_x, l, varf) \
            + np.identity(len(self.X_train[:, 0])) * (self.var_y + varn)
        Kinv = np.linalg.pinv(K)  # Calculate the pseudo-inverse of a matrix
        # print('varf:', varf, 'varn:', varn, 'lenghtscale:', l)
        return 0.5 * np.dot(self.y_train, np.dot(Kinv, self.y_train)) + 0.5 * np.log(np.linalg.det(K)) \
               + 0.5 * len(K[:, 0]) * np.log(2 * math.pi)


## Example insipired from 'Learning Gaussian Process Models from Uncertain Data', Dallaire.
if __name__ == "__main__":
    # def sincsig(x):
    #     return (x >= 0) * np.sinc(x / math.pi) + (x < 0) * (0.5 * (1 + np.exp(-10 * x - 5)) ** (-1) + 0.5)

    def sincsig(x):
        return np.sin(x)

    X_train = np.random.random((150, 1)) * 20.0 - 10.0 # generate 150 data points from interval [-10, 10]
    y_train = sincsig(X_train[:, 0])
    # y_train = sincsig(X_train[:, 0])

    X_std = np.random.random(X_train.shape) * 2.0 + 0.5
    y_std = 0.1 * np.ones_like(y_train)
    y_train += np.random.normal(0.0, y_std)
    X_train += np.random.normal(0.0, X_std)
    # y = sin(x) + noise_y
    # x_train = x + noise_x

    # create data by constant interval
    Xcv = np.linspace(-10, 10, 100).reshape(-1, 1)
    ycv = sincsig(Xcv[:, 0])

    print('Start standard GP')
    l_bounds = np.array([[0.01, 0.3], [0.02, 0.2], [0.1, 5.0]])
    gp = GPRegressor(1, 1, num_restarts_hyper=10)
    gp.fit(X_train, y_train, 0.0, 0.0, l_bounds=l_bounds)
    yp, std = gp.predict(Xcv, True)
    print('End standard GP')
    # print
    # np.sqrt(np.average((yp - ycv) ** 2))

    plot_gp(yp, std, Xcv, X_train=X_train, Y_train=y_train, title='GP')
    # plt.figure()
    # plt.errorbar(Xcv[:, 0], yp, yerr=2 * std)
    # plt.plot(Xcv[:, 0], sincsig(Xcv))
    # plt.plot(X_train[:, 0], y_train, 'r.')
    # plt.title('Regular GP')
    # plt.show()


    print('Start NIGP')
    l_bounds = np.array([[0.01, 0.3], [0.01, 0.1], [0.1, 5.0]])
    gp = GPRegressor(1, 1, num_restarts_hyper=10)
    gp.fit(X_train, y_train, X_std ** 2, 0.0, l_bounds=l_bounds)
    yp, std = gp.predict(Xcv, True)
    print('End NIGP')
    # print
    # np.sqrt(np.average((yp - ycv) ** 2))

    plot_gp(yp, std, Xcv, X_train=X_train, Y_train=y_train, title='NIGP')
    # plt.figure()
    # plt.errorbar(Xcv[:, 0], yp, yerr=2 * std)
    # plt.plot(Xcv[:, 0], sincsig(Xcv))
    # plt.plot(X_train[:, 0], y_train, 'r.')
    # plt.title('Noisy GP')
    # plt.show()
    # print('End')


def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):

    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)
    plt.show()

