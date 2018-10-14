# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import (
    absolute_import, division, print_function, unicode_literals)
try:
    input = raw_input
except NameError:
    pass
import os
import random
import time
import datetime as dtm
from contextlib import contextmanager
import numpy as np
import tensorflow as tf
from functools import wraps


@contextmanager
def timing(label):
    t0 = time.time()
    yield
    print("--- {:.3f}s to {}\n".format(time.time() - t0, label))


def print_shapes(locals_dict, *names):
    for name in names:
        print("%s ~ %s" % (
            name,
            locals_dict[name].shape
        ))


def fix_all_rngs(n):
    os.environ['PYTHONHASHSEED'] = str(n)
    random.seed(n)
    np.random.seed(n)
    tf.set_random_seed(n)


def date_str_to_ts(dt_str, format_str='%Y-%m-%d'):
    return int(
        (
            dtm.datetime.strptime(dt_str, format_str) -
            dtm.datetime(1970, 1, 1)
        ).total_seconds()
    )


def datetime_to_ts(dt):
    return time.mktime(dt.timetuple())


def ts_to_ymd(ts):
    dt = dtm.datetime.utcfromtimestamp(ts)
    return (dt.year, dt.month, dt.day)


def ymd_to_ts(y, m, d):
    dt = dtm.datetime(y, m, d)
    return time.mktime(dt.timetuple())


def ts_to_dt_str(ts):
    return dtm.datetime.utcfromtimestamp(ts).strftime(
        '%Y-%m-%d %H:%M:%S')
