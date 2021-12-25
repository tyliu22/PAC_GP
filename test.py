
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

from utils.data_generator import generate_sin_data

# Number of data points
N_train = 50
N_test = 100

# Input space dimension
D = 1

x_min = -3
x_max = 3
dx = (x_max - x_min) / 6.0

# %% Generate data
x_data, y_data = generate_sin_data(N_train, x_min+dx, x_max-dx,
                                   0.1**2, random_order=True)
x_true, y_true = generate_sin_data(N_test, x_min, x_max, None,
                                   random_order=False)


# Random Forest Regression
regr = RandomForestRegressor(max_depth=4, random_state=0)
regr.fit(x_data, y_data)
y_fit_RF = regr.predict(x_true)

# SVM Regression
svr = SVR().fit(x_data, y_data)
y_fit_SVR = svr.predict(x_true)

print(mean_squared_error(y_true, y_fit_RF))
print(mean_squared_error( y_true, y_fit_SVR ))

print('End')


