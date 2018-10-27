# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import (
    absolute_import, division, print_function, unicode_literals)
try:
    input = raw_input
except NameError:
    pass
import matplotlib.pyplot as plt


def simple_plot(
    data, xticks_span=1, show_xlabels=True, show_markers=False, title=''
):
    fig = plt.figure(figsize=(10, 3), facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    # axes
    if show_xlabels:
        ax.set_xticks(data.index.values[::xticks_span])
        ax.set_xticklabels(data['datetime'][::xticks_span],
                           rotation=45, horizontalalignment='right')
    # lines
    line_style = '|-' if show_markers else '-'
    if title:
        plt.title(title)
    plt.plot(data.index.values, data['open'], line_style,
             color='orangered', alpha=0.5, label='open', linewidth=1)
    plt.plot(data.index.values, data['close'], line_style,
             color='blue', alpha=0.5, label='close', linewidth=1)
    plt.fill_between(
        data.index.values, data['low'], data['high'],
        color='gainsboro', alpha=0.5, linewidth=1)
    # legend
    plt.legend()
