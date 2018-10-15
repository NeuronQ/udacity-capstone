# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import (
    absolute_import, division, print_function, unicode_literals)
try:
    input = raw_input
    range = xrange
except NameError:
    pass

import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from lib import helpers


def make_rnn_model_constructor(
        layers, optimizer=keras.optimizers.rmsprop, lr=None,
        loss='mse', activation='linear',
):
    def print_model(m):
        print("\n--- Created model id: {}".format(id(m)))
        print(m.input)
        print(m.summary())
        print(m.output)
        print()

    def make_model(seq_len, features_n):
        model = Sequential()
        model.add(LSTM(
            units=layers[0],
            input_shape=(seq_len, features_n),
            return_sequences=True,
        ))
        model.add(Dropout(layers[1]))
        model.add(LSTM(
            units=layers[2],
        ))
        model.add(Dropout(layers[3]))
        model.add(Dense(1, activation=activation))

        with helpers.timing('model compilation time'):
            optimizer_params = {'lr': lr} if lr else {}
            model.compile(loss=loss, optimizer=optimizer(**optimizer_params))

        print_model(model)
        return model

    make_model.desc = 'RNN({}|{}, {}{}, {})'.format(
        '|'.join(map(str, layers)),
        activation,
        getattr(optimizer, '__name__', optimizer),
        '(lr={})'.format(lr) if lr else '',
        loss)

    return make_model
