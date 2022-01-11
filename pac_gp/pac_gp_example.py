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
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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

# GPy.models.gplvm

# %% Set up and train GPy model for comparison
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

# %% Predict on test data
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
print("PAC-NIGP error: " + str(mean_squared_error(y_true, y_mean_pac_nigp)))


plt.figure('Data and GPy/GPtf/PAC GP predictions')
plt.clf()


plt.subplot(1, 3, 1)
plt.title('Full GP (GPy)')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_full_gpy)
error = np.squeeze(2 * np.sqrt(y_var_full_gpy))
plt.plot(x_true, y, '-', color='C2', label='full GP (GPy)')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)
plt.xlabel('input $x$')
plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-1.5, 1.5])
plt.xlim([-3, 3])

plt.subplot(1, 3, 2)
plt.title('Sparse GP (GPy)')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_sparse_gpy)
error = np.squeeze(2 * np.sqrt(y_var_sparse_gpy))
plt.plot(x_true, y, '-', color='C2', label='sparse GP (GPy)')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)

plt.plot(np.squeeze(sparse_gpy.inducing_inputs), -1.5 * np.ones((M, )), 'o',
         color='C3', label='GPy inducing inputs')
plt.xlabel('input $x$')
plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-1.5, 1.5])
plt.xlim([-3, 3])


plt.subplot(1, 3, 3)
plt.title('PAC GP')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_pac_nigp)
error = np.squeeze(2 * np.sqrt(y_var_pac_nigp))
plt.plot(x_true, y, '-', color='C2', label='sparse GP (tf)')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)

plt.plot(np.squeeze(z_gpy), -1.5 * np.ones((M, )), 'o',
         color='C3', label='GPy inducing inputs')
# plt.plot(np.squeeze(Z_opt), -1.5 * np.ones((M, )), 'o',
#          color='C4', label='PAC GP inducing inputs')
plt.xlabel('input $x$')
plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-1.5, 1.5])
plt.xlim([-3, 3])
plt.show()
