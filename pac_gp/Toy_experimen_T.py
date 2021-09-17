"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""
import sys
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os
import argparse
from pac_gp.gp import gpr, kerns, mean_functions

from sklearn import preprocessing
import tensorflow as tf

import utils.plotting as plotting
import utils.load_dataset as load_dataset
import utils.helpers as helpers

"""
    Toy model for GPR or NIGPR
    input data single dimension is {sin} function with noise
    y = sin(x) + noise_y
"""

def run(dataset_name, fn_out, epsilon_range, test_size=0.1, n_repetitions=10,
        ARD=False, nInd=0, loss='01_loss'):
    """
    running methods

    input:
    dataset_name    :   name of dataset to load
    fn_out          :   name of results file
    epsilon_range   :   list of epsilons
    test_size       :   fraction of test data points
    n_repetitions   :   number of repetitions
    ARD             :   automatic relevance determinant
    epsilon         :   sensitivity parameter of the loss function
    loss            :   loss function to use (01_loss, inv_gauss)
    """

    # training dataset
    X_train = np.random.random((150, 1)) * 20.0 - 10.0
    Y_train = np.sin(X_train)

    N, D = X_train.shape[0], X_train.shape[1]

    X_std = 0.4 # standard var sigma, variance is simga^2
    noise_y = np.random.normal(0.0, X_std, size=X_train.shape)

    Y_train = np.sin(X_train) + noise_y

    # test dataset
    Xcv = np.linspace(-10, 10, 100).reshape(-1, 1)
    ycv = np.sin(Xcv[:, 0])

    kern = kerns.RBF(input_dim=X_train.shape[1], ARD=ARD)



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
    args.n_reps = int(args.n_reps)
    args.nInd = int(args.nInd)

    dir_results = 'epsilon'
    if args.nInd == 0:
        models = ['NIGP_sqrt-PAC HYP GP', 'bkl-PAC HYP GP', 'sqrt-PAC HYP GP', 'GPflow Full GP']
        fn_args = (args.dataset, args.loss, args.ARD, 100.*args.test_size,
                   args.n_reps)
        fn_base = '%s_%s_ARD%d_testsize%d_nReps%d' % fn_args

        fn_results = os.path.join(dir_results, '%s.pckl' % fn_base)
        fn_png = os.path.join(dir_results, '%s.png' % fn_base)
        fn_pdf = os.path.join(dir_results, '%s.pdf' % fn_base)
    else:
        models = ['bkl-PAC Inducing Hyp GP', 'sqrt-PAC Inducing Hyp GP',
                  'GPflow VFE', 'GPflow FITC']
        fn_args = (args.dataset, args.loss, args.nInd, args.ARD,
                   100.*args.test_size, args.n_reps)
        fn_base = '%s_%s_nInd%d_ARD%d_testsize%d_nReps%d' % fn_args
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