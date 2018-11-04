# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import (
    absolute_import, division, print_function, unicode_literals)
try:
    input = raw_input
except NameError:
    pass
import matplotlib.pyplot as plt


def simple_plot(
    data, xticks_span=1, show_xlabels=True, show_markers=False, title='',
    show_volume=True,
):
    fig = plt.figure(figsize=(15, 5), facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel('price')
    # axes
    if show_xlabels:
        ax.set_xticks(data.index.values[::xticks_span])
        ax.set_xticklabels(data['datetime'][::xticks_span],
                           rotation=45, horizontalalignment='right')
    # lines
    line_style = '|-' if show_markers else '-'
    if title:
        plt.title(title)
    lopen = ax.plot(data.index.values, data['open'], line_style,
                    color='orangered', alpha=0.6, label='open', linewidth=1)
    lclose = ax.plot(data.index.values, data['close'], line_style,
                     color='blue', alpha=0.6, label='close', linewidth=1)
    lhilo = ax.fill_between(
        data.index.values, data['low'], data['high'],
        color='gainsboro', alpha=0.5, linewidth=1, label='low - high')
    ax.legend(loc=2)
    # volume
    if show_volume:
        ax2 = ax.twinx()
        lvolume = ax2.set_ylabel('volume')
        ax2.plot(data.index.values, data['volume'], line_style,
                 color='green', alpha=0.4, label='volume', linewidth=1)
    # legend
    ax2.legend(loc=1)


def plot_train_val_losses(train_val_losses):
    for i, losses in enumerate(train_val_losses):
        plt.figure(facecolor='white')
        plt.title("Training/validation loss" +
                  (' #%d' % (i + 1)) if len(train_val_losses) else '')
        plt.plot(losses['loss'], color='g', label='training')
        plt.plot(losses['val_loss'], color='m', label='validation')
        plt.legend()
        plt.show()
