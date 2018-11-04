# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import absolute_import, division, print_function, unicode_literals
try:
    input = raw_input
    range = xrange
except NameError:
    pass

import numpy as np
import pandas as pd
from IPython.display import display as dd
from sklearn.preprocessing import MinMaxScaler

from . import helpers
from . import visualization as viz
from sklearn import linear_model


def extract_data_matrix_from_df(
    data_frame,
    features,
    from_i,
    to_i,
):
    """
    Extract a [n_samples, n_features] numpy matrix of data from a pandas df.

    Parameters
    ----------
    data_frame : pandas df
    features : []string
    from_i : int
    to_i : int

    Returns
    -------
    [n_samples, n_features] np.array
    """
    data = [data_frame[f].iloc[from_i: to_i].values.copy()
            for f in features]
    data = np.array(data).T

    assert len(data.shape) == 2
    assert data.shape[0] == to_i - from_i, "%s != %s - %s" % (
        data.shape[0], to_i, from_i)
    assert data.shape[1] == len(features)

    return data


def scaled_data(data, train_sz):
    """Scale a [n_samples, n_features] numpy matrix of data to [-1, 1] range
    for each feature (column).
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data[:train_sz])
    sdata = scaler.transform(data)
    return sdata, scaler


def detrended_data(data):
    data_ = data.copy()  # don't modify params
    data_[1:] -= data_[:-1]
    data_[1, 1] = 0.
    return data_


def make_train_test_seqs(seqs, train_sz, shuffle):
    # print("train_sz = %s" % (train_sz,))
    # print("\nseqs ~ %s" % (seqs.shape,))

    seq_len = seqs.shape[1]  # this is length of training seq + 1
    # IMPORTANT: -seq_len is to prevent train-test cross-contamination
    #   bc of sequence overlaps
    train_seqs = seqs[:train_sz - seq_len, :, :]
    test_seqs = seqs[train_sz:, :, :]

    if shuffle:
        np.random.shuffle(train_seqs)

    x_train_seqs = train_seqs[:, :-1, :]
    y_train = train_seqs[:, -1, 0:1]
    assert x_train_seqs.shape == (
        train_sz - seq_len,
        seq_len - 1,
        seqs.shape[2]
    )
    assert y_train.shape == (train_sz - seq_len, 1)

    x_test_seqs = test_seqs[:, :-1, :]
    y_test = test_seqs[:, -1, 0:1]
    assert x_test_seqs.shape == (
        seqs.shape[0] - train_sz,
        seq_len - 1,
        seqs.shape[2]
    ), "unexpected shape %s" % (x_test_seqs.shape,)
    assert y_test.shape == (seqs.shape[0] - train_sz, 1)

    return x_train_seqs, y_train, x_test_seqs, y_test


def make_sliding_seqs(
    data,
    seq_len
):
    """Make overlapping/sliding sequences of multidimensional data from a
    [n_samples, n_features] numpy matrix of data.

    Parameters
    ----------
    data : [n_samples, n_features] np.array
    seq_len : int

    Returns
    -------
    [ n_samples - seq_len, seq_len + 1, n_features] np.array
    """
    seq_len += 1  # bc we need and extra point to be y
    seqs = []
    for i in range(len(data) - seq_len):
        s = data[i: i + seq_len]
        seqs.append(s)
    seqs = np.array(seqs)

    assert seqs.shape == (
        data.shape[0] - seq_len,
        seq_len,
        data.shape[1]
    )

    return seqs


def normalized_seqs_cols(seqs):
    """Normalize each sequence in a set by the formula:
    p_i := p_i / p_0 - 1
    """
    assert len(seqs.shape) == 3
    seqs_ = seqs.copy()  # don't modify params
    for i_seq in range(seqs_.shape[0]):
        for i_feature in range(seqs_.shape[2]):
            feature_seq = seqs_[i_seq, :, i_feature]
            assert feature_seq.shape == (seqs.shape[1],)
            if feature_seq[0] == 0:
                continue
            if np.isnan(feature_seq[0]):  # it's caller's responsibility to avoid NaNs!
                continue
            feature_seq /= feature_seq[0]
            feature_seq -= 1
            assert feature_seq[0] == 0, "%f != %f" % (
                feature_seq[0], 0
            )
    return seqs_


def load_and_preview(filename, graph_title='', xticks_span=30):
    with helpers.timing('load, reverse & plot data_bc'):
        data = pd.read_csv(filename, index_col='timestamp')
        print("\nEntries #: {}\n".format(len(data)))
        data = data.reindex(index=data.index[::-1])
        dd(data.iloc[:5])
        viz.simple_plot(data, xticks_span=xticks_span, title=graph_title)
    return data


def add_derived_features(data_df, extra_data_df):
    out = data_df.join(extra_data_df.loc[:, ('close',)], rsuffix='_extra')

    out['tf_slope'] = pd.Series(np.zeros(len(data_df)), index=data_df.index)
    out['tf_r2'] = pd.Series(np.zeros(len(data_df)), index=data_df.index)

    _last_n = 7

    regr = linear_model.LinearRegression()

    for i in range(_last_n, len(out)):
        trend_data = out.iloc[i - _last_n: i]['close_extra'].dropna()

        if len(trend_data) <= 1:
            print('insufficient data at', i)
            continue

        xs = trend_data.index.values.reshape((-1, 1))
        ys = trend_data.values.reshape((-1, 1))

        regr.fit(xs, ys)

        out.loc[out.index[i], ('tf_slope', 'tf_r2')] = (
            regr.coef_[0][0],
            regr.score(xs, ys)
        )

    return out


def augment_with_news_sentiment(data_df, extra_df, last_n_days, suffix):
    # add extra_df's "close" column to data_df
    out_df = data_df.join(extra_df.loc[:, ('sentiment',)])

    out_df['sentiment' + suffix] = pd.Series(
        np.zeros(len(out_df)), index=out_df.index)

    regr = linear_model.LinearRegression()
    for i in range(last_n_days, len(out_df)):
        # get last_n_days previous data (skip missing) for trend
        current_t = out_df.iloc[i].name
        sentiment = out_df[
            (out_df.index > current_t - pd.to_timedelta(last_n_days, unit='d')) &
            (out_df.index <= current_t)
        ]['sentiment'].dropna()

        # if current_t == pd.Timestamp('2015-10-01'):
        #     import pdb; pdb.set_trace()

        # skip trying to get trend from 0 less points
        if len(sentiment) == 0:
            # print('WARNING: insufficient data at', i)
            continue

        # add sentiment mean for last
        out_df.loc[out_df.index[i], ('sentiment' + suffix)] = (
            sentiment.values.mean(),
        )

    return out_df


def augment(data_df, extra_df, last_n, suffix):
    # add extra_df's "close" column to data_df
    out_df = data_df.join(extra_df.loc[:, ('close',)], rsuffix=suffix)

    out_df['slope' + suffix] = pd.Series(
        np.zeros(len(out_df)), index=out_df.index)
    out_df['r2' + suffix] = pd.Series(
        np.zeros(len(out_df)), index=out_df.index)

    regr = linear_model.LinearRegression()
    for i in range(last_n, len(out_df)):
        # get last_n previous data (skip missing) for trend
        trend_data = out_df.iloc[i - last_n: i]['close' + suffix].dropna()

        # skip trying to get trend from 1 or less points
        if len(trend_data) <= 1:
            print('WARNING: insufficient data at', i)
            continue

        # fit a simple linear regression on trend data
        xs = trend_data.index.values.reshape((-1, 1)).astype(np.int64)
        ys = trend_data.values.reshape((-1, 1))
        regr.fit(xs, ys)
        # get trend slope and R2 score from LR model
        trend_slope = regr.coef_[0][0]
#         import pdb; pdb.set_trace()
        trend_r2 = regr.score(xs, ys)

        # add trend slope and R2 to returned df
        out_df.loc[out_df.index[i], ('slope' + suffix, 'r2' + suffix)] = (
            trend_slope,
            trend_r2
        )

    return out_df
