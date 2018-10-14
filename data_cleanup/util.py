# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import (
    absolute_import, division, print_function, unicode_literals)
try:
    input = raw_input
    range = xrange
except NameError:
    pass

from functools import wraps
import time


class rate_limit(object):
    def __init__(self, n, dt, spacing_dt=None):
        self.n = n
        self.dt = dt
        self.remaining = n
        self.spacing_dt = spacing_dt
        self.t0 = time.time()

    def __call__(self, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.time() - self.t0 > self.dt:
                self.t0 = time.time()
                self.remaining = self.n
            r = func(*args, **kwargs)
            if self.spacing_dt:
                time.sleep(self.spacing_dt)
            self.remaining -= 1
            if self.remaining == 0:
                # wait if no runs left for current interval
                t_elapsed = time.time() - self.t0
                t_left = self.dt - t_elapsed
                if t_left > 0:
                    time.sleep(t_left)
                # reset
                self.t0 = time.time()
                self.remaining = self.n
            return r
        return wrapped_func


def retry(tries, delay=None):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            remaining = tries
            while remaining:
                if remaining != tries:
                    print("[retry]")
                r, err = func(*args, **kwargs)
                remaining -= 1
                if not err:
                    break
                if delay:
                    time.sleep(delay)
            return r, err
        return wrapped_func
    return decorator
