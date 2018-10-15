# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import (
    absolute_import, division, print_function, unicode_literals)
try:
    input = raw_input
    range = xrange
except NameError:
    pass

import numpy as np
from sklearn import linear_model
from lib import helpers

from .model import Model


class LRModel(Model):
    desc = 'LRModel'

    def __init__(self, seq_len, n_features):
        self.regr = linear_model.LinearRegression()

    def fit(self, x_train_seqs, y_train,
            batch_size, epochs, validation_split, shuffle):
        pass

    def predict(self, xs):
        out = np.zeros((xs.shape[0], 1), dtype=np.float32)
        for i in range(xs.shape[0]):
            self.regr.fit(np.arange(xs.shape[1]).reshape((-1, 1)),
                          xs[i, :, 0:1])
            # out[i, 0] = self.regr.predict([[len(xs)]])[0]
            out[i, 0] = self.regr.predict([[xs.shape[1]]])[0]
            # out[i, 0] = self.regr.predict([[1]])[0]
        return out
