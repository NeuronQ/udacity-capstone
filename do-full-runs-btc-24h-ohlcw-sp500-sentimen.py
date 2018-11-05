# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import absolute_import, division, print_function, unicode_literals
try:
    input = raw_input
    range = xrange
except NameError:
    pass

import csv
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import pprint

from lib.helpers import (
    timing, print_shapes, fix_all_rngs,
    date_str_to_ts
)
from lib.visualization import simple_plot
from lib.walk_forward_predict import (
    run_walk_forward_validation_rnn,
    run_walk_forward_validation_arima,
    run_walk_forward_validation_rnn_retraining
)
import lib.etl as etl
from models.rnn import make_rnn_model_constructor
from models.lr import LRModel

pp = pprint.PrettyPrinter(indent=2).pprint


# Load data
##########################################################################
BC24H_FILENAME = './data_clean/btc_usd_24h_bitstamp.csv'
SP500_FILENAME = './data_clean/sp500.csv'
NEWS_DATA_FILENAME = './data_clean/abcnews-date-text.with-sentiment.csv'

# load BTC @ 24 h data
data24h = pd.read_csv(
    BC24H_FILENAME,
    index_col='datetime',
    parse_dates=['datetime']
)
data24h['weekday'] = data24h.index.dayofweek

# load SP500 data
data_sp500 = pd.read_csv(
    SP500_FILENAME,
    index_col='datetime',
    parse_dates=['datetime']
)

# add SP500-derived features to BTC @ 24 h data
data = etl.augment(data24h, data_sp500, 7, '_sp500')

# load bitcoin news with sentiment analysis data
data_news = pd.read_csv(
    NEWS_DATA_FILENAME,
    index_col='datetime',
    parse_dates=['datetime']
)

# add news sentiment feature to data
data = etl.augment_with_news_sentiment(data, data_news, 30, '_last30d')


# Full runs
##########################################################################
DEFAULT_PARAMS = {
    # data to extract:
    'data_df': data,
    'train_sz': 300,
    'test_sz': 300,
    # data processing:
    'normalize': 'seqs',  # 'seqs' | 'data'
    'detrend': False,
    # model and prediction arrangement:
    'seq_len': 7,
    'pred_len': 7,
    'model_maker': make_rnn_model_constructor(
        [7, 0.1, 7, 0.1],
        optimizer=keras.optimizers.RMSprop,
        lr=1e-4,
    ),
    # training:
    'epochs': 100,
    'batch_size': 32,
    'shuffle': False,
    # experiment setup:
    'times': 1,
    'skip': False,
    'fix_all_rngs_to': False,
    'fix_rngs_before_each': False,
    # plotting:
    'plot': False,
    'fig_size': (10, 8),
    'fast': True,
}

CSV_FIELDS = (
    'model_description',
    'train_from',
    'train_to',
    'test_on',
    'shuffle',
    'normalize',
    'training_loss',
    'rmse',
    'rmse_cp',
    'dir_acc',
)


def ddump(s):
    """Print to both notebook and system stdout."""
    # os.write(1, s)
    print(s)


def full_run(
    idx_from, idx_to, out_filename, write_csv_header, params,
    run_method=run_walk_forward_validation_rnn,
):
    train_sz = params['train_sz']
    test_sz = params['test_sz']

    # out_file = open(out_filename, 'ab')
    out_file = open(out_filename, 'wb')
    csv_writer = csv.DictWriter(out_file, fieldnames=CSV_FIELDS)
    if write_csv_header:
        csv_writer.writeheader()

    with timing('full run %d - %d' % (idx_from, idx_to)):
        for i in range(idx_from, idx_to, test_sz):
            params = dict(params,
                          from_i=i - train_sz)
            ddump("\n   ------ RUN %d (from %s, train on %d, test on %d)\n" % (
                i, params['from_i'], params['train_sz'], params['test_sz']))
            with timing('run segment'):
                out = run_method(**params)
                del out['train_val_losses']
                csv_writer.writerow(out)
                out_file.flush()


# OHLCW predictions
##########################################################################
OHLCW_FILE = './full_results/btc24h_ohlcw.2.csv'
OHLCW_PARAMS = DEFAULT_PARAMS.copy()
OHLCW_PARAMS['features'] = [
    'close', 'open', 'high', 'low', 'volume', 'weekday',
]
full_run(300, len(data) - 300 - 7, out_filename=OHLCW_FILE,
         write_csv_header=True, params=OHLCW_PARAMS)


# OHLCW predictions
##########################################################################
OHLCW_FILE = './full_results/btc24h_ohlcw.2.csv'
OHLCW_PARAMS = DEFAULT_PARAMS.copy()
OHLCW_PARAMS['features'] = [
    'close', 'open', 'high', 'low', 'volume', 'weekday',
]
full_run(300, len(data) - 300 - 7, out_filename=OHLCW_FILE,
         write_csv_header=True, params=OHLCW_PARAMS)


# OHLCW + SP500 features predictions
##########################################################################
OHLCW_SP500_FILE = './full_results/btc24h_ohlcw_sp500.2.csv'
OHLCW_PARAMS_SP500 = DEFAULT_PARAMS.copy()
OHLCW_PARAMS_SP500['features'] = [
    'close', 'open', 'high', 'low', 'volume', 'weekday',
    'slope_sp500', 'r2_sp500',
]
full_run(300, 1051, out_filename=OHLCW_SP500_FILE, write_csv_header=True,
         params=OHLCW_PARAMS_SP500)


# OHLCW + SP500 + sentiment predictions
##########################################################################
OHLCW_SP500_ST_FILE = './full_results/btc24h_ohlcw_sp500_st.2.csv'
OHLCW_PARAMS_SP500_ST = DEFAULT_PARAMS.copy()
OHLCW_PARAMS_SP500_ST['features'] = [
    'close', 'open', 'high', 'low', 'volume', 'weekday',
    'slope_sp500', 'r2_sp500',
    'sentiment_last30d',
]
full_run(300, 1051, out_filename=OHLCW_SP500_ST_FILE, write_csv_header=True,
         params=OHLCW_PARAMS_SP500_ST)
