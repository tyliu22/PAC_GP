"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""
import sys
import matplotlib
#matplotlib.use('agg')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os
import argparse
from sklearn import preprocessing
import tensorflow as tf

import utils.plotting as plotting
import utils.load_dataset as load_dataset
import utils.helpers as helpers


def run(dataset_name, fn_out, epsilon_range, test_size=0.1, n_repetitions=10,
        ARD=False, nInd=0, loss='01_loss'):
    """
    running methods

    input:
    dataset_name    :   sin function for testing the GPR without GPflow
    fn_out          :   name of results file
    epsilon_range   :   list of epsilons
    test_size       :   fraction of test data points
    n_repetitions   :   number of repetitions
    ARD             :   automatic relevance determinant
    epsilon         :   sensitivity parameter of the loss function
    loss            :   loss function to use (01_loss, inv_gauss)
    """

    X = np.arange(-10, 10, 0.05).reshape(-1, 1)
    Y = np.sin(X)

    F = X.shape[1]
    noise_x_variance = 2.0
    # noise_x = np.random.normal(0.0, noise_x_variance, size=X.shape)
    noise_x_covariance = np.eye(F) * noise_x_variance

    # sin data without uncertainty:
    X_original = X
    X_noise = X

    """ Full GP Regerssion Test:
    """

    data = []
    for i in range(n_repetitions):
        # vary epsilon
        print(i)
        for ie, epsilon in enumerate(epsilon_range):

            if nInd == 0: # standard GPR
                print('Start running:')
                print('Full GP Algorithm: GPflow Full GP')
                RV_gpflow = helpers.compare(X_noise, Y, X_original, 'GPflow Full GP', seed=i,
                                            test_size=test_size, ARD=ARD,
                                            epsilon=epsilon, loss=loss, noise_input_variance=noise_x_covariance)
                RVs = [RV_gpflow]
                print('End exact NIGP_sqrt-PAC HYP GP')

            for RV in RVs:
                data += RV

    print('Store data into DataFrame df')
    df = pd.DataFrame(data)
    df.to_pickle(fn_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running full GPs')
    parser.add_argument('-r', '--run', help='run', action='store_true',
                        default=True)
    parser.add_argument('-p', '--plot', help='plot', action='store_true',
                        default=True)
    parser.add_argument('-d', '--dataset', default='boston')
    parser.add_argument('-a', '--ARD', help='use ARD', action='store_true',
                        default=True)
    parser.add_argument('-t', '--test_size', help='testsize in [0.0, 1.0]',
                        default=0.2)
    parser.add_argument('-n', '--n_reps', help='number of repetitions',
                        default=1)
    parser.add_argument('-m', '--nInd', help='number of inducing points',
                        default=0)
    parser.add_argument('-l', '--loss', help='loss function',
                        default='01_loss')

    args = parser.parse_args()
    args.test_size = float(args.test_size)
    args.n_reps = int(args.n_reps)
    args.nInd = int(args.nInd)

    dir_results = 'epsilon'
    if args.nInd == 0:  # standard GPR
        models = ['GPflow Full GP']
        fn_args = (args.dataset, args.loss, args.ARD, 100.*args.test_size,
                   args.n_reps)
        fn_base = '%s_%s_ARD%d_testsize%d_nReps%d' % fn_args

        fn_results = os.path.join(dir_results, '%s.pckl' % fn_base)
        fn_png = os.path.join(dir_results, '%s.png' % fn_base)
        fn_pdf = os.path.join(dir_results, '%s.pdf' % fn_base)

    if not(os.path.exists(dir_results)):
        os.mkdir(dir_results)

    epsilon_range = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    if args.run:
        print('Run Experiments')
        run(args.dataset, fn_results, epsilon_range,
            test_size=float(args.test_size), ARD=args.ARD,
            n_repetitions=args.n_reps, nInd=args.nInd, loss=args.loss)

    args.plot = "True"
    if args.plot:
        matplotlib.rc('font', **{'size': 14})
        D = pd.read_pickle(fn_results)
        plotting.plot(D, models, x="epsilon", xticks=[0.2, 0.4, 0.6, 0.8, 1.0],
                      ylim=(0, 0.85))
        plt.show()
        # plt.savefig(fn_png)
        # plt.savefig(fn_pdf)
        # plt.close()


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))  # 95%的置信区间

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=2, ls='--', label=f'Sample{i + 1}')  # lw is the width of curve
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()