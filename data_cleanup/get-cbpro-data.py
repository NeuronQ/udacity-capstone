# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import (
    absolute_import, division, print_function, unicode_literals)
try:
    input = raw_input
    range = xrange
except NameError:
    pass

from datetime import datetime
import time
import datetime as dt
import csv
import sys
import os
from functools import wraps

import cbpro

from util import


OUT_FILE = '../data_src/btc.300s.cbpro.csv'
END_AT = dt.datetime(year=2018, month=10, day=10, hour=0)
START_AT = dt.datetime(year=2014, month=12, day=1, hour=0)
GRANULARITY_SECONDS = 300
MAX_DATA_POINTS = 300 - 10  # for one request

TEST_RUN = False

# pc.get_products()

# RATE LIMITS: 3 req/sec public, 5 req/sec authenticated
#   NOTE: seems to be more like 2 and 4...

pc = cbpro.PublicClient()
# pc = cbpro.AuthenticatedClient(
#     os.environ.get('CBPRO_API_KEY'),  # key
#     os.environ.get('CBPRO_API_SECRET_B64'),  # b64secret
#     os.environ.get('CBPRO_API_PASSPHRASE'),  # passphrase
# )

if TEST_RUN:
    print('\n====== test run ======')
    r = pc.get_product_historic_rates(
        'BTC-USD',  # from get_products
        datetime(2018, 1, 1).isoformat(),
        datetime(2018, 1, 2).isoformat(),
        granularity=300
    )
    if type(r) is list:
        print("> got #{} from {} to {}".format(
            len(r),
            datetime.fromtimestamp(r[0][0]),
            datetime.fromtimestamp(r[-1][0]).isoformat()
        ))
        print("> sample:")
        print(r[:5])
    else:
        print("> got unexpected result:", r)
    print('====== / test run ======\n')


max_req_interval = GRANULARITY_SECONDS * MAX_DATA_POINTS
n_requests = int((END_AT - START_AT).total_seconds() / max_req_interval)

to_dt = END_AT
from_dt = END_AT - dt.timedelta(seconds=max_req_interval)

print("""> Will fetch data from {} to {},
    in {} requests for intervals of {} seconds...
    """.format(START_AT, END_AT, n_requests, max_req_interval))


def save_data(data):
    for row in data:
        row.insert(1, dt.datetime.utcfromtimestamp(
            int(row[0])).strftime('%Y-%m-%d %H:%M:%S'))
    csv_writer.writerows(data)
    out_file.flush()

# with private ai key try these for faster resultst:
# @retry(tries=4, delay=1)
# @rate_limit(n=4, dt=1)


@retry(tries=4, delay=1)
@rate_limit(n=2, dt=1)
def fetch_data(from_dt, to_dt):
    print(time.time())
    print("Fetching data from {} to {}".format(
        from_dt, to_dt))
    try:
        r = pc.get_product_historic_rates(
            'BTC-USD',
            from_dt.isoformat(),
            to_dt.isoformat(),
            granularity=300
        )
        if type(r) is not list:
            raise Exception(r)
        print("> Received {} data points".format(len(r)))
        return r, None
    except Exception as e:
        print("--- ERROR: {}".format(e))
        return None, e


out_file = open(OUT_FILE, 'ab')
csv_writer = csv.writer(out_file)

if not os.path.isfile(OUT_FILE) or not os.stat(OUT_FILE).st_size:
    print("> empty or nonexistent file => writing csv header row")
    csv_writer.writerow(['timestamp', 'datetime', 'low',
                         'high', 'open', 'close', 'volume'])

i = 0
while from_dt >= START_AT:
    i += 1
    print("({} of {} @ {})> Will fetch data from {} to {}".format(
        i, n_requests, datetime.now(), from_dt, to_dt))
    data, err = fetch_data(from_dt, to_dt)
    if not err:

        save_data(data)

    to_dt = from_dt
    from_dt -= dt.timedelta(seconds=max_req_interval)

    # break
    # if i == 20:
    #     break
