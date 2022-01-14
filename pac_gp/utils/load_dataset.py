# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import sklearn.datasets
# from uci_datasets import Dataset
from uci_datasets.uci_datasets import Dataset

def load(dataset_name):
    if dataset_name == 'boston':
        data = sklearn.datasets.load_boston(return_X_y=True)
    else:
        raise Exception('Dataset %s is not known.' % dataset_name)

    return data
# from uci_datasets import Dataset
# data = Dataset("challenger")
# x_train, y_train, x_test, y_test = data.get_split(split=0)
