# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

from gp.pac_gp import PAC_HYP_GP, PAC_INDUCING_HYP_GP, PAC_GP_BASE, NIGP_PAC_HYP_GP
from GPy.models import GPRegression, SparseGPRegression
from gp.mean_functions import Zero
from gp import kerns

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.cluster.vq import kmeans2
from utils import metrics
import gpflow
import time
import utils.gpflow_wrapper as gpflow_wrapper
import tensorflow as tf


def as_pac(param):
    """
    shapes parameter to PAC-GP format ??????

    input:
    param   :   parameter
    func: data dimension reduce dimension
    """
    if (param.ndim == 1 and param.shape[0] == 1) or (param.ndim == 0):
        return np.array([np.squeeze(param)], dtype=np.float64)
    else:
        return np.squeeze(param)


def transform_to_pac_gp(model, epsilon=0.1, delta=0.01, ARD=False,
                        loss='01_loss', **extra):
    """
    transforms model to PAC GP model
    transform the GP model into

    input
    model       :       trained GP-object
    epsilon     :       sensitivity parameter of loss function
    delta       :       PAC-bound holds with at least prob 1-delta
    ARD         :       automatic feature detection (default: false)
    loss        :       {01_loss, inv_gauss}

    issubclass(class, classinfo):    if class is the subclass of classinfo, then return True, otherwise return False
                                     "class" inherits "classinfo" method
    """

    if issubclass(type(model), PAC_GP_BASE):
        return model

    elif issubclass(type(model), GPRegression):
        kern = kerns.RBF(input_dim=model.input_dim, ARD=ARD)       # create kernal based dimensions
        sn2 = as_pac(model.Gaussian_noise.variance)                #
        m = PAC_HYP_GP(X=model.X, Y=model.Y, sn2=sn2, kernel=kern,
                       epsilon=epsilon, delta=delta, loss=loss)
        m.kernel.variance = as_pac(model.kern.variance)
        m.kernel.lengthscale = as_pac(model.kern.lengthscale)
        m.round_hyps()
        return m

    elif issubclass(type(model), SparseGPRegression):
        kern = kerns.RBF(input_dim=model.input_dim, ARD=ARD)
        m = PAC_INDUCING_HYP_GP(X=model.X, Y=model.Y, Z=model.inducing_inputs,
                                sn2=as_pac(model.Gaussian_noise.variance),
                                kernel=kern, epsilon=epsilon,
                                delta=delta, loss=loss)
        m.kernel.variance = as_pac(model.kern.variance)
        m.kernel.lengthscale = as_pac(model.kern.lengthscale)
        m.round_hyps()
        return m

    elif issubclass(type(model), gpflow_wrapper.GPflowSparseWrapper):
        Z = model.model.feature.Z.value
        sn2 = model.model.likelihood.variance.value
        X = model.model.X.value
        Y = model.model.Y.value
        kern = kerns.RBF(input_dim=X.shape[1], ARD=ARD)
        m = PAC_INDUCING_HYP_GP(X=X, Y=Y, Z=Z,
                                sn2=as_pac(sn2),
                                kernel=kern, epsilon=epsilon,
                                delta=delta, loss=loss)

        m.kernel.variance = as_pac(model.model.kern.variance.value)
        m.kernel.lengthscale = as_pac(model.model.kern.lengthscales.value)
        m.round_hyps()
        return m

    elif issubclass(type(model), gpflow_wrapper.GPflowFullWrapper):

        sn2 = model.model.likelihood.variance.value
        X = model.model.X.value
        Y = model.model.Y.value
        kern = kerns.RBF(input_dim=X.shape[1], ARD=ARD)
        m = PAC_HYP_GP(X=X, Y=Y, sn2=as_pac(sn2), kernel=kern, epsilon=epsilon,
                       delta=delta, loss=loss)
        m.kernel.variance = as_pac(model.model.kern.variance.value)
        m.kernel.lengthscale = as_pac(model.model.kern.lengthscales.value)
        m.round_hyps()
        return m

    else:
        return 1.


def run_model(_model, metrics, X_noise_test, Y_test, X_origin_test, epsilon, delta, ARD,
              loss='01_loss', noise_input_variance=None):
    """
    running model

    input
    model       :       inizialized un-trained GP-object
    metrics     :       dictionary where each entry is a evaluation function
    X_test      :       test data points
    Y_test      :       test outcome
    epsilon     :       sensitivity parameter of loss function
    delta       :       PAC-bound holds with at least prob 1-delta
    ARD         :       automatic feature detection (default: false)
    loss        :       {01_loss, inv_gauss}
    """

    # optimize
    _model.optimize()

    # transform to GP PAC
    model = transform_to_pac_gp(_model, epsilon=epsilon, delta=delta, ARD=ARD,
                                loss=loss)

    # predict
    ymean_hat, yvar_hat = model.predict(X_origin_test, full_cov=False)
    fmean_hat, fvar_hat = model.predict_noiseless(X_origin_test, full_cov=False)

    # compute evaluation metrics
    res = {}
    # Load "helpers -> compare() -> metric"   different model function:
    # model.get_upper_bound()  ;     model.get_upper_bound_bkl()
    # model.get_kl_divergence();     model.get_empirical_risk()
    for mkey in metrics.keys():
        metric = metrics[mkey]
        kwargs = {'Y_true': Y_test,
                  'Ymean_hat': ymean_hat,
                  'Yvar_hat': yvar_hat,
                  'epsilon': epsilon,
                  'fmean_hat': fmean_hat,
                  'fvar_hat': fvar_hat,
                  'delta': delta,
                  'model': model,
                  'ARD': ARD}
        res[mkey] = metric(**kwargs)

    return res


def init_inducing_points(X, m):
    """
    initialize m inducing points by using k-means on X

    inputs:
    X   :   data points
    m   :   number of clusters
    """
    seed = int(np.abs(X.flatten()[0]))
    numpy_rand_state = np.random.get_state()
    np.random.seed(seed)
    Z_init = kmeans2(X, k=m)[0]
    np.random.set_state(numpy_rand_state)
    return Z_init


def build_model(model_name, X_noise_train, Y, X_origin_train, ARD=False, delta=0.01, epsilon=0.2,
                nInd=None, suffix='', loss='01_loss', noise_input_variance=None):
    """
    setting up model

    input
    model_name  :       model name in {bkl-PAC HYP GP, sqrt-PAC HYP GP,
                        bkl-PAC Inducing Hyp GP, sqrt-PAC Inducing Hyp GP,
                        GPflow Full GP, GPflow VFE, GPflow FITC}
    metrics     :       dictionary where each entry is a evaluation function
    X           :       data points
    Y           :       outcome
    epsilon     :       sensitivity parameter of loss function
    delta       :       PAC-bound holds with at least prob 1-delta
    ARD         :       automatic feature detection (default: false)
    nInd        :       number of inducing points
    loss        :       {01_loss, inv_gauss}
    """
    F = X_origin_train.shape[1]

    if model_name == 'bkl-PAC HYP GP':
        kern = kerns.RBF(input_dim=F, ARD=ARD)
        sn2_init = np.asarray([1.0 ** 2], dtype=np.float64)
        mean_function = Zero()
        model = PAC_HYP_GP(X=X_noise_train, Y=Y, kernel=kern, sn2=sn2_init,
                           epsilon=epsilon, mean_function=mean_function,
                           delta=delta, verbosity=0, loss=loss, noise_input_variance=None)

    elif model_name == 'sqrt-PAC HYP GP':
        kern = kerns.RBF(input_dim=F, ARD=ARD)
        sn2_init = np.asarray([1.0 ** 2], dtype=np.float64) # transfer this as (flaot64) array
        mean_function = Zero()
        model = PAC_HYP_GP(X=X_noise_train, Y=Y, kernel=kern, sn2=sn2_init,
                           epsilon=epsilon, mean_function=mean_function,
                           delta=delta, verbosity=0, method='naive', loss=loss, noise_input_variance=None)

    elif model_name == 'NIGP_sqrt-PAC HYP GP':
        # change to another kernal: NIGP kernal which consider the noise input regularization item as variable
        kern = kerns.RBF(input_dim=F, ARD=ARD)
        sn2_init = np.asarray([1.0 ** 2], dtype=np.float64) # transfer this as (flaot64) array
        # random initialize noisy input matrix, shape of matrix should be the same as K_NN
        # data_dim = tf.shape(X)[0]
        # noise_x_init = tf.random_normal([data_dim, data_dim], dtype=tf.float64)

        mean_function = Zero()
        model = NIGP_PAC_HYP_GP(X=X_noise_train, Y=Y, kernel=kern, sn2=sn2_init,
                                epsilon=epsilon, mean_function=mean_function,
                                delta=delta, verbosity=0, method='naive', loss=loss, noise_input_variance=None)

    elif model_name == 'bkl-PAC Inducing Hyp GP':
        Z = init_inducing_points(X_noise_train, nInd)
        kern = kerns.RBF(input_dim=F, ARD=ARD)
        sn2_init = np.asarray([1.0 ** 2], dtype=np.float64) # transfer this as (flaot64) array
        mean_function = Zero()
        model = PAC_INDUCING_HYP_GP(X=X_noise_train, Y=Y, Z=Z, kernel=kern, sn2=sn2_init,
                                    epsilon=epsilon,
                                    mean_function=mean_function,
                                    delta=delta, verbosity=0, loss=loss, noise_input_variance=None)

    elif model_name == 'sqrt-PAC Inducing Hyp GP':
        Z = init_inducing_points(X_noise_train, nInd)
        kern = kerns.RBF(input_dim=F, ARD=ARD)
        sn2_init = np.asarray([1.0 ** 2], dtype=np.float64) # transfer this as (flaot64) array
        mean_function = Zero()
        model = PAC_INDUCING_HYP_GP(X=X_noise_train, Y=Y, Z=Z, kernel=kern, sn2=sn2_init,
                                    epsilon=epsilon,
                                    mean_function=mean_function,
                                    delta=delta, verbosity=0, method='naive',
                                    loss=loss, noise_input_variance=None)

    elif model_name == 'GPflow Full GP':
        kern = gpflow.kernels.RBF(F, ARD=ARD)
        _model = gpflow.models.GPR(X_noise_train, Y, kern)
        model = gpflow_wrapper.GPflowFullWrapper(_model)

    elif model_name == 'GPflow VFE':
        Z = init_inducing_points(X_noise_train, nInd)
        kern = gpflow.kernels.RBF(F, ARD=ARD)
        _model = gpflow.models.SGPR(X_noise_train, Y, kern, Z)
        model = gpflow_wrapper.GPflowSparseWrapper(_model)

    elif model_name == 'GPflow FITC':
        Z = init_inducing_points(X_noise_train, nInd)
        kern = gpflow.kernels.RBF(F, ARD=ARD)
        _model = gpflow.models.GPRFITC(X_noise_train, Y, kern, Z)
        model = gpflow_wrapper.GPflowSparseWrapper(_model)

    return model


def compare(X_noise, Y, X_original, model_name, seed, delta=0.01, test_size=0.2, ARD=False,
            epsilon=0.2, nInd=None, suffix='', loss='01_loss', noise_input_variance=None):
        """
        running and evaluating model

        input
        X           :       data points
        X_noise     :       noisy data points
        X_origin    :       original clear data points

        Y           :       outcome
        model_name  :       model name in {bkl-PAC HYP GP, sqrt-PAC HYP GP,
                            bkl-PAC Inducing Hyp GP, sqrt-PAC Inducing Hyp GP,
                            GPflow Full GP, GPflow VFE, GPflow FITC}
        seed        :       seed for random number generator
                            (used for training/ test split)
        epsilon     :       sensitivity parameter of loss function
        delta       :       PAC-bound holds with at least prob 1-delta
        ARD         :       automatic feature detection (default: false)
        nInd        :       number of inducing points
        loss        :       {01_loss, inv_gauss}
        """
        metric = {'MSE': metrics.mean_squared_error,
                  'negative predictive log-likelihood': metrics.neg_ll,
                  'gibbs-risk': metrics.gibbs_risk_noiseless,
                  'bayes-risk': metrics.bayes_risk,
                  'inv-gauss': metrics.inv_gauss,
                  'upper-bound-sqrt': lambda model, **extra:
                  model.get_upper_bound(),
                  'upper-bound-bkl': lambda model, **extra:
                  model.get_upper_bound_bkl(),
                  'KL-divergence': lambda model, **extra:
                  model.get_KL_divergence(),
                  'log-theta': lambda model, **extra: model.get_log_theta(),
                  'gibbs-risk-train': lambda model, **extra:
                  model.get_empirical_risk(),
                  'sigma squared': lambda model, **extra: model.sn2[0]}

        # split into training and test data
        rv = train_test_split(X_noise, Y, X_original, random_state=seed, test_size=test_size)
        # noise input setting: three parts
        X_noise_train, X_noise_test, Y_train, Y_test, X_origin_train, X_origin_test = rv

        # running model
        Ntest = Y_test.shape[0]
        N = Y_train.shape[0]
        t0 = time.time()
        model = build_model(model_name, X_noise_train, Y_train, X_origin_train, ARD=ARD, delta=delta,
                            epsilon=epsilon, nInd=nInd, suffix='', loss=loss, noise_input_variance=noise_input_variance)
        RV = run_model(model, metric, X_noise_test, Y_test, X_origin_test, epsilon, delta, ARD,
                       loss=loss, noise_input_variance=noise_input_variance)
        t1 = time.time()
        t_diff = t1 - t0

        # save metrics
        data = []
        for mkey in RV.keys():
            data.append({'N': N, 'model': model_name, 'epsilon': epsilon,
                         'nInd': nInd, 'metric': mkey, 'value': RV[mkey]})
        data.append({'N': N, 'model': model_name, 'epsilon': epsilon,
                     'nInd': nInd, 'metric': 'time', 'value': t_diff})

        return data
