"""
    Function:
        plot_gp: plot the Gaussian processing
        plot_gp_f: plot the mean/covariance function of Gaussian processing

    Setting:

"""

import numpy as np
import matplotlib.pyplot as plt


# plot the mean and covariance of GPR with testing data
def plot_gp(mu, cov, X, X_train=None, Y_train=None, titles='GPR plot'):
    # X: test data:  X_train and y_train: training data and labels
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    # plt.plot(X, np.sin(Y))
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.title(titles)
    plt.show()


# plot the GP function f, calculated only use training data without testing data
def plot_gp_f(mu, cov, X_train=None, Y_train=None, titles='GPR function f'):
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

