# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import (
    absolute_import, division, print_function, unicode_literals)
try:
    input = raw_input
except NameError:
    pass

import keras
import pandas as pd

from lib import etl
from lib import helpers
from lib import walk_forward_predict
from models import rnn


BC24H_FILENAME = './data_clean/btc_usd_24h_bitstamp.csv'
BC5M_FILENAME = './data_clean/btc.300s.cbpro.csv'
SP500_FILENAME = './data_clean/sp500.csv'
NEWS_DATA_FILENAME = './data_clean/abcnews-date-text.with-sentiment.csv'
STOCKS = [
    'amd',
    'nvda',
    'gbtc',
    'mara',
    'ostk',
    'otiv',
    'riot',
    'sieb',
]
STOCKS_FILENAMES = {s: './data_clean/' + s + '.csv'
                    for s in STOCKS}

DEFAULT_RUN_PARAMS = {
    'normalize': 'seqs',  # 'seqs' | 'data'
    'detrend': False,
    'shuffle': False,
    'times': 1,
    'skip': False,
    'fix_all_rngs_to': False,
    'fix_rngs_before_each': False,
    'plot': True,
    'fig_size': (10, 8),
    'fast': True,
}

DEFAULT_24H_RUN_PARAMS = dict(
    DEFAULT_RUN_PARAMS,
    seq_len=7,
    features=[
        'close', 'open', 'high', 'low', 'volume', 'weekday',
        'slope_sp500', 'r2_sp500',
        # 'sentiment_last30d'
    ],
    model_maker=rnn.make_rnn_model_constructor(
        [7, 0.1, 7, 0.1],
        optimizer=keras.optimizers.RMSprop,
        lr=1e-4,
    ),
    epochs=100,
    batch_size=32,
)

DEFAULT_STOCK_RUNS_PARAMS = dict(
    DEFAULT_24H_RUN_PARAMS,
    features=[
        'close', 'open', 'high', 'low', 'volume', 'weekday',
        'slope_btc', 'r2_btc',
    ]
)

DEFAULT_5MIN_RUN_PARAMS = dict(
    DEFAULT_RUN_PARAMS,
    seq_len=100,
    features=['close', 'open', 'high', 'low', 'volume'],
    model_maker=rnn.make_rnn_model_constructor(
        [100, 0.1, 50, 0.1],
        optimizer=keras.optimizers.RMSprop,
        lr=1e-4,
    ),
    epochs=8,
    batch_size=512,
    skip=True,
)

data_cache = {}


def predict_btc24h_multiple(
    train_from_date,
    train_to_date,
    predict_n,
    predict_ahead,
):
    """
    User-friendly simplified wrapper around BTC 24h predictions model for quick experiments.

    Parameters
    ----------
    train_from_date : datetime or string
    train_to_date : datetime or string
    predict_n : int
        how many predictions to make
    predict_ahead : int
        how many points ahead to predict
    """
    if 'btc24h' not in data_cache:
        print("> first time, so we load BTC 24h data...")
        data_cache['btc24h'] = load_btc24h_data()
        print("> loaded BTC 24h data")
    data = data_cache['btc24h']
    params = DEFAULT_24H_RUN_PARAMS.copy()

    train_from_i = data.index.get_loc(train_from_date)
    train_to_i = data.index.get_loc(train_to_date)
    train_sz = train_to_i - train_from_i

    params['from_i'] = train_from_i
    params['train_sz'] = train_sz
    params['test_sz'] = predict_n
    params['pred_len'] = predict_ahead

    return walk_forward_predict.run_walk_forward_validation_rnn(data, **params)


def predict_stock_multiple(
    symbol,
    train_from_date,
    train_to_date,
    predict_n,
    predict_ahead,
):
    """
    User-friendly simplified wrapper around stocks prediction model for quick experiments.

    Parameters
    ----------
    symbol: str
        one of 'AMD', 'NVDA', 'GBTC', 'MARA', 'OSTK', 'OTIV', 'RIOT', 'SIEB'
    train_from_date : datetime or string
    train_to_date : datetime or string
    predict_n : int
        how many predictions to make
    predict_ahead : int
        how many points ahead to predict
    """
    symbol = symbol.upper()
    if symbol not in data_cache:
        print("> first time, so we load stock %s data..." % symbol)
        data_cache[symbol] = load_stock_data(symbol)
        print("> loaded stock %s data" % symbol)
    data = data_cache[symbol]

    params = DEFAULT_STOCK_RUNS_PARAMS.copy()

    train_from_i = data.index.get_loc(train_from_date)
    train_to_i = data.index.get_loc(train_to_date)
    train_sz = train_to_i - train_from_i

    params['from_i'] = train_from_i
    params['train_sz'] = train_sz
    params['test_sz'] = predict_n
    params['pred_len'] = predict_ahead

    return walk_forward_predict.run_walk_forward_validation_rnn(data, **params)


def predict_btc5min_multiple(
    train_from_date,
    train_to_date,
    predict_n,
    predict_ahead,
):
    if 'btc5min' not in data_cache:
        print("> first time, so we load BTC 5min data...")
        data_cache['btc5min'] = load_btc5min_data()
        print("> loaded BTC 5min data")
    data = data_cache['btc5min']
    params = DEFAULT_5MIN_RUN_PARAMS.copy()

    train_from_idx = data[data['datetime'] == train_from_date].index.values[0]
    train_to_idx = data[data['datetime'] == train_to_date].index.values[0]
    train_from_i = data.index.get_loc(train_from_idx)
    train_to_i = data.index.get_loc(train_to_idx)
    train_sz = train_to_i - train_from_i

    params['from_i'] = train_from_i
    params['train_sz'] = train_sz
    params['test_sz'] = predict_n
    params['pred_len'] = predict_ahead

    return walk_forward_predict.run_walk_forward_validation_rnn(data, **params)


def load_btc5min_data():
    return pd.read_csv(
        BC5M_FILENAME,
        index_col='timestamp',
        parse_dates=['datetime']
    )


def load_stock_data(symbol):
    data = pd.read_csv(
        STOCKS_FILENAMES[symbol.lower()],
        index_col='datetime',
        parse_dates=['datetime']
    )
    data = data[data.index >= '2015-02-28']
    data['weekday'] = data.index.dayofweek
    if 'btc24h' not in data_cache:
        print("> first time, so we load BTC 24h data...")
        data_cache['btc24h'] = load_btc24h_data()
        print("> loaded BTC 24h data")
    data_btc = data_cache['btc24h']
    data = etl.augment(data, data_btc, 7, '_btc')
    return data


def load_btc24h_data():
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

    return data
