# write code that works in both Python 2 (2.7+) and 3 (3.5+)
from __future__ import absolute_import, division, print_function, unicode_literals
try:
    input = raw_input
    range = xrange
except NameError:
    pass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA

from . import helpers
from . import etl


def run_walk_forward_validation_rnn(
    # data to extract:
    data_df,
    features,
    from_i,
    train_sz,
    test_sz,
    # data processing:
    normalize,  # 'seqs' | 'data'
    detrend,
    # model and prediction arrangement:
    seq_len,
    pred_len,
    model_maker,
    # training:
    epochs,
    batch_size,
    shuffle,
    # experiment setup:
    times,
    skip,
    fix_all_rngs_to,
    fix_rngs_before_each,
    # plotting:
    plot=True,
    model_desc='',
    fig_size=(10, 8),
    fast=False
):
    assert normalize in {'seqs', 'data'}
    assert times in {1, 4}, "plotting support for other values not implemented"

    if fix_all_rngs_to is not False:
        helpers.fix_all_rngs(fix_all_rngs_to)

    # extract data
    data = etl.extract_data_matrix_from_df(
        data_df,
        features,
        from_i,
        from_i + train_sz + test_sz
    )
    data.setflags(write=False)
    test_data = data[train_sz:]
    test_data.setflags(write=False)
    assert data.shape == (train_sz + test_sz, len(features))

    # normalize data
    scaler = None
    if normalize == 'data':
        data, scaler = etl.scaled_data(data, train_sz)
    # detrend
    if detrend:
        data = etl.detrended_data(data)

    ###
    # plt.plot(data)
    # plt.show()
    print("data ~ %s" % (data.shape,))

    # make sequences
    seqs = etl.make_sliding_seqs(data, seq_len)

    # normalize seqs
    if normalize == 'seqs':
        seqs = etl.normalized_seqs_cols(seqs)

    # x_train_seqs, x_test_seqs : [#, seq_len, n_features]
    # y_train, y_test : [#, 1]
    x_train_seqs, y_train, x_test_seqs, y_test =\
        etl.make_train_test_seqs(seqs, train_sz, shuffle)
    x_train_seqs.setflags(write=False)
    y_train.setflags(write=False)
    x_test_seqs.setflags(write=False)
    y_test.setflags(write=False)

    ###
    # plt.plot(x_train_seqs[0])
    # plt.show()
    # plt.plot(data[:seq_len])
    # plt.show()
    # plt.plot(y_train[:seq_len])
    # plt.show()
    # plt.plot(data[seq_len: seq_len * 2])
    # plt.show()

    model_description = (model_desc or getattr(model_maker, 'desc', '')) + \
        ' {} batch_size={} epochs={} seq_len={} pred_len={}'.format(
            ','.join(features),
            batch_size,
            epochs,
            seq_len,
            pred_len
    )

    # plotting setup
    if plot:
        plt.figure(figsize=fig_size, facecolor='white')
        desc = (
            'train on {from_idx}:{to_idx} bc@5m data shuffle={shuffle}, '
            'test on next {test_sz}, normalize={normalize}'
        ).format(
            from_idx=from_i,
            to_idx=from_i + train_sz,
            shuffle=int(shuffle),
            test_sz=test_sz,
            normalize=normalize,
        )
        plt.suptitle(
            model_description +
            (('\n' + desc) if desc else '') +
            '\n\n\n'
        )
        if times > 1:
            plt.subplots_adjust(top=0.8, hspace=0.5)
        # show a nice grid if multilple
        rows = cols = np.ceil(np.sqrt(times))

    out = dict(
        model_description=model_description,
        train_from=from_i,
        train_to=from_i + train_sz,
        test_on=test_sz,
        shuffle=shuffle,
        normalize=normalize,
        train_val_losses=[],
    )

    # possibly multiple runs
    for i in range(times):

        if fix_rngs_before_each:
            helpers.fix_all_rngs(fix_all_rngs_to)

        model = model_maker(seq_len, len(features))

        with helpers.timing('train model'):
            training_history = model.fit(
                x_train_seqs,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                shuffle=False,
            )
            out['train_val_losses'].append(training_history.history)

        final_training_loss = -1
        if training_history:
            final_training_loss = training_history.history['loss'][-1]  # val_loss

        with helpers.timing('walk and predict'):
            if not skip:
                pred_seqs, losses = walk_and_predict_all_batch(
                    model,
                    x_test_seqs,
                    pred_len,
                    compute_dacc_loss_vs_ct_y,
                    batch_sz=batch_size
                )
            else:
                predict_func = fast_walk_and_predict if fast else walk_and_predict
                pred_seqs, losses = predict_func(
                    model,
                    x_test_seqs,
                    pred_len,
                    compute_dacc_loss_vs_ct_y
                )

        ###
        # print('### pred_seqs: len={}, shape={}'.format(
        #     len(pred_seqs),
        #     pred_seqs[0].shape if len(pred_seqs) else -1
        # ))

        ###
        # plt.plot(pred_seqs[0])
        # plt.show()

        # denormalize predictions
        if normalize == 'data':
            for preds in pred_seqs:
                # hack because the scaler expects a number of columns equal to
                # no. of features, but we only predict feature 0
                if len(features) == 1:
                    scaler_input = preds.reshape((-1, 1))
                else:
                    scaler_input = preds.reshape((-1, 1)).repeat(
                        len(features), axis=1
                    )
                preds[:] = scaler.inverse_transform(scaler_input)[:, 0]
        elif normalize == 'seqs':
            for preds_i, preds in enumerate(pred_seqs):
                preds += 1
                preds *= test_data[preds_i if not skip else preds_i * seq_len, 0]

        ###
        # plt.plot(pred_seqs[0])
        # plt.show()

        # readd trend
        if detrend:
            if not skip:
                raise NotImplementedError(
                    'Using read_trend with no_skip=True is not implemented')
            for preds_i, preds in enumerate(pred_seqs):
                preds[0] += test_data[
                    (preds_i + 1) * seq_len - 1,
                    0
                ]
                for pidx in range(1, len(preds)):
                    preds[pidx] += preds[pidx - 1]

        # no more changes to predictions from this onwards
        pred_seqs = np.array(pred_seqs)
        pred_seqs.setflags(write=False)

        # process losses data for printing in (sub)plot title
        losses = np.array(losses, dtype=np.float32)
        if not skip:
            dir_accs = losses[0] * 100
            rmses = losses[1]
            rmses_cp = losses[2]
            # get last ones
            dir_acc = dir_accs[-1]
            rmse = rmses[-1]
            rmse_cp = rmses_cp[-1]
        else:
            if not fast:
                dir_acc = np.average(losses[:, 0]) * 100
                rmse = np.sqrt(np.average(losses[:, 2]))
                rmse_cp = np.sqrt(np.average(losses[:, 3]))
            else:
                dir_acc, _, rmse, rmse_cp = losses

        # return results/stats
        out.update(
            training_loss='{:.4e}'.format(final_training_loss),
            rmse='{:.4f}%'.format(rmse),
            rmse_cp='{:.4f}%'.format(rmse_cp),
            dir_acc='{:.4f}%'.format(dir_acc),
        )

        if plot:
            if times > 1:
                plt.subplot(rows, cols, i + 1)
            # plot title
            plt.title(
                'Dir. Acc.: {:.4f}%'.format(dir_acc) +
                ('\n' if times > 1 else ', ') +
                'RMSE: {:.4f} vs. {:.4f} for CP'.format(
                    rmse,
                    rmse_cp,
                ) +
                ('\n' if times > 1 else ' ') +
                '(Loss: {:.4e})'.format(final_training_loss)
            )

            # plot predictions test data with predictions
            with helpers.timing('plot incremental predictions'):
                if not skip:
                    plot_pred_seqs_all(
                        test_data[:, 0], seq_len, pred_seqs)
                else:
                    plot_pred_seqs(
                        test_data[:, 0], seq_len, pred_seqs)

    if plot:
        if not skip:
            plt.figure()
            plt.title('DACC(#ahead)')
            plt.plot(np.arange(1, len(dir_accs) + 1), dir_accs)
        plt.show()

    return out


def walk_and_predict(
        model,
        seqs,
        pred_len,
        compute_loss=None,
        extrapolate_features=True,
):
    """
    Parameters
    ----------
    model
    seqs : [n_sequences, seq_len, n_features] np.array
    pred_len : int
    compute_loss : func(pred_y, target_y, start_y)
    extrapolate_features : bool

    Returns
    -------
    pred_seqs : [(n_sequences // seq_len) * [pred_len] np.array] list
    losses
    """
    assert len(seqs.shape) == 3
    seq_len = seqs.shape[1]
    assert pred_len <= seq_len

    pred_seqs = []
    losses = []

    lr = LinearRegression() if extrapolate_features else None

    for i in range(len(seqs) // seq_len):
        xs = seqs[i * seq_len, :, :].copy()  # !!
        ys = np.zeros(pred_len, dtype=np.float32)  # !!

        for j in range(pred_len):
            ys[j] = model.predict(xs[None, :, :])[0, 0]
            xs[0: -1, :] = xs[1:, :]
            xs[-1, 0] = ys[j]
            if extrapolate_features:
                for fi in range(1, xs.shape[1]):
                    xs[-1, fi] = _extrapolate(lr, xs[0: -1, fi])

        pred_seqs.append(ys)

        if compute_loss:
            loss = compute_loss(
                ys[-1],
                seqs[i * seq_len + pred_len - 1, -1, 0],
                seqs[i * seq_len - 1, -1, 0]
            )
            if loss is not None:
                losses.append(loss)
            else:
                print("WARNING: can't compute loss for prediction at index %d" % (
                    i * seq_len + pred_len,
                ))

    return pred_seqs, losses


def fast_walk_and_predict(
    model,
    seqs,
    pred_len,
    compute_loss=None,
    extrapolate_features=True,
):
    """
    Parameters
    ----------
    model
    seqs : [n_sequences, seq_len, n_features] np.array
    pred_len : int
    compute_loss : func(pred_y, target_y, start_y)
    extrapolate_features : bool

    Returns
    -------
    pred_seqs : [(n_sequences // seq_len) * [pred_len] np.array] list
    losses
    """
    assert len(seqs.shape) == 3
    seq_len = seqs.shape[1]
    assert pred_len <= seq_len

    # pred_seqs = []
    # losses = []
    losses = np.zeros(4, dtype=np.float32)

    lr = LinearRegression() if extrapolate_features else None

    batch_sz = len(seqs) // seq_len - 1

    xs = np.zeros((batch_sz, seq_len, seqs.shape[2]), dtype=np.float32)
    ys = np.zeros((batch_sz, pred_len), dtype=np.float32)
    start_ys = np.zeros(batch_sz, dtype=np.float32)
    target_ys = np.zeros(batch_sz, dtype=np.float32)

    for i in range(batch_sz):
        xs[i, :, :] = seqs[i * seq_len, :, :]
        start_ys[i] = seqs[i * seq_len, -1, 0]
        target_ys[i] = seqs[i * seq_len + pred_len, -1, 0]

    extrapf_xs = None
    if extrapolate_features and xs.shape[2] > 1:
        extrapf_xs = _extrapolate_multi_features(lr, xs[:, :, 1:], pred_len)

    for j in range(pred_len):
        if len(xs) == 0:
            print("!!! empty prediction input passed")
            continue
        r = model.predict(xs)
        try:
            assert r.shape == (batch_sz, 1), (r.shape, batch_sz)
        except:
            print(
                r,
                batch_sz
            )
            import pdb
            pdb.set_trace()
        ys[:, j] = r[:, 0]
        xs[:, 0: -1, :] = xs[:, 1:, :]
        xs[:, -1, 0] = ys[:, j]

        if extrapolate_features and xs.shape[2] > 1:
            xs[:, -1, 1:] = extrapf_xs[j]

    losses = (
        (  # prediction dir acc.
            ((start_ys - target_ys) * (start_ys - ys[:, -1])) > 0
        ).astype(np.float32).sum() / batch_sz * 100,
        0.0,
        np.sqrt(
            np.sum((target_ys - ys[:, -1]) ** 2) / batch_sz
        ),  # prediction square error
        np.sqrt(
            np.sum((target_ys - start_ys) ** 2) / batch_sz
        ),  # constant prediction square error
    )

    return ys, losses


def _extrapolate(lr, xs):
    idxs = np.arange(xs.shape[0]).reshape((-1, 1))
    lr.fit(idxs, xs)
    return lr.predict(np.array([[xs.shape[0]]]))[0]


def walk_and_predict_all_batch(
    model,
    seqs,
    pred_len,
    compute_loss=None,
    batch_sz=100,
    extrapolate_features=True,
):
    """
    Visual eg.:

    N: 10, seq_len: 5, pred_len: 3, batch_sz: 4
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18

    _batch_1______________
    0 1 2 3 4 - - -
      1 2 3 4 5 - - -
        2 3 4 5 6 - - -
          3 4 5 6 7 - - --

            _batch_2__________________
            4 5 6 7 8 - -- --
              5 6 7 8 9 -- -- --
                6 7 8 9 10 -- -- --
                  7 8 9 10 11 -- -- --

                  _batch_3_____________________
                  7 8 9 10 11 -- -- --
                    8 9 10 11 12 -- -- --
                      9 10 11 12 13 -- -- --
                        10 11 12 13 14 -- -- --

                           _batch_4_(incomplete)__
                           11 12 13 14 15 -- -- --

                           (stop at 10 = 19 - 5 - 3)

    Parameters
    ----------
    model
    seqs : [n_sequences, seq_len, n_features] np. array
    pred_len : int
    batch_sz : int
    compute_loss : func(pred_y, target_y, start_y)

    Returns
    -------
    pred_seqs : [(n_sequences - seq_len - pred_len + 1) * [pred_len] np.array] list
    losses : [4] np.array
    """
    assert len(seqs.shape) == 3
    seq_len = seqs.shape[1]

    assert seq_len >= pred_len  # this simplifies some things

    # pred_seqs = []
    n_batches = (len(seqs) - seq_len - pred_len + 1) // batch_sz
    n_batches -= int(np.ceil(pred_len / seq_len))
    pred_seqs = np.zeros(
        (n_batches * batch_sz, pred_len),
        dtype=np.float32
    )
    # import pdb
    # pdb.set_trace()

    # losses = np.zeros(4, dtype=np.float32)  # aggregate compute_loss results
    losses = np.zeros((3, pred_len), dtype=np.float32)  # aggregate compute_loss results
    losses_count = 0

    # OPT: to prevent allocating new matrixes in loop
    xs = np.zeros((batch_sz, seq_len, seqs.shape[2]), dtype=np.float32)
    ys = np.zeros((batch_sz, pred_len), dtype=np.float32)

    lr = LinearRegression() if extrapolate_features else None

    # print("\n### len(seqs)={} seq_len={} pred_len={}, batch_sz={}".format(
    #     len(seqs), seq_len, pred_len, batch_sz,
    # ))

    for i in range(0, batch_sz * n_batches, batch_sz):
        # if i + batch_sz > seqs.shape[0]:
        #     break

        print("> Predicting (now at %d)..." % i)

        xs[:, :, :] = seqs[i: i + batch_sz, :, :]
        ys.fill(0)

        extrapf_xs = None
        if extrapolate_features and xs.shape[2] > 1:
            extrapf_xs = _extrapolate_multi_features(lr, xs[:, :, 1:], pred_len)

        start_ys = xs[:, -1:, 0].copy()
        target_ys = seqs[
            i + pred_len: i + batch_sz + pred_len,
            -pred_len:,
            0
        ]

        for j in range(pred_len):
            # predict
            y_preds = model.predict(xs)[:, 0]
            # store predicted value as last of input seq and shift the other ones
            ys[:, j] = y_preds
            xs[:, 0: -1, :] = xs[:, 1:, :]  # shift left
            xs[:, -1, 0] = y_preds  # put predicted values as last values
            # extrapolate values of other features for the added data point
            if extrapolate_features and xs.shape[2] > 1:
                xs[:, -1, 1:] = extrapf_xs[j]

        pred_seqs[i: i + batch_sz, :] = ys

        # compute losses
        dir_accs = (ys - start_ys) * (target_ys - start_ys) > 0
        dir_accs = dir_accs.astype(int)
        losses[0] += dir_accs.sum(0)  # direction accuracy
        losses[1] += np.sum((target_ys - ys) ** 2, 0)  # RMSE
        losses[2] += np.sum((target_ys - start_ys) ** 2, 0)  # constant pred. RMSE
        losses_count += batch_sz

        # if i == 10:
        #     import pdb
        #     pdb.set_trace()

        # for ys_idx in range(batch_sz):
        #     pred_seqs.append(ys[ys_idx].copy())

        #     loss = compute_loss(
        #         ys[ys_idx, -1],
        #         seqs[i + ys_idx + pred_len, -1, 0],
        #         seqs[i + ys_idx, -1, 0]
        #     )
        #     if loss is not None:
        #         losses += loss
        #         losses_count += 1
        #     else:
        #         print("WARNING: can't compute loss for prediction at index %d" % (
        #             i
        #         ))

    losses[0, :] /= losses_count
    losses[1, :] = np.sqrt(losses[1, :] / losses_count)
    losses[2, :] = np.sqrt(losses[2, :] / losses_count)

    return pred_seqs, losses


def _extrapolate_multi_features(lr, xs, pred_len):
    """
    Parameters
    ----------
    xs : [batch_sz, seq_len, n_features] np.array

    Returns
    -------
    [pred_len, batch_sz, n_features -  1]
    """
    batch_sz, seq_len, n_features = xs.shape
    out = np.zeros((pred_len, batch_sz, n_features), dtype=np.float32)

    for iseq in range(batch_sz):
        for ifeature in range(n_features):
            feature_seq = xs[iseq, :, ifeature]
            lr.fit(
                np.arange(seq_len).reshape((-1, 1)),
                feature_seq.reshape((-1, 1))
            )
            r = lr.predict(
                np.arange(seq_len, seq_len + pred_len).reshape((-1, 1))
            )
            # print("\n=== r ~", r.shape)
            out[:, iseq, ifeature] = r.flatten()

    return out


def compute_dacc_loss_vs_ct_y(pred_y, target_y, start_y):
    return (
        _mda(start_y, target_y, pred_y),  # prediction dir acc.
        _mda(start_y, target_y, start_y),  # constant prediction dir acc.
        (target_y - pred_y) ** 2,  # prediction square error
        (target_y - start_y) ** 2,  # constant prediction square error
    )


def _mda(start_y, target_y, predicted_y):
    return int((start_y - target_y) * (start_y - predicted_y) > 0)


def plot_pred_seqs(data, seq_len, pred_seqs):
    pred_len = len(pred_seqs[0])
    plt.plot(data, color='b', linewidth=1)
    for i, preds in enumerate(pred_seqs):
        plt.plot(
            np.arange(
                (i + 1) * seq_len,
                (i + 1) * seq_len + pred_len
            ),
            preds,
            linewidth=1
        )


def plot_pred_seqs_all(data, seq_len, pred_seqs):
    pred_len = len(pred_seqs[0])
    plt.plot(data, color='b', linewidth=1)
    for i, preds in enumerate(pred_seqs):
        plt.plot(
            np.arange(
                seq_len + i,
                seq_len + pred_len + i,
            ),
            preds,
            linewidth=1,
            color='m',
            alpha=0.4
        )


def arima_predict(data, order, ahead):
    try:
        model = ARIMA(data, order)
        res = model.fit()
        preds = res.forecast(ahead)[0]
    except:
        preds = np.ones(ahead) * np.average(data[-ahead:])
    return preds


def run_walk_forward_validation_arima(
        # data to extract:
        data_df,
        features,
        from_i,
        train_sz,
        test_sz,
        # data processing:
        normalize,
        # model and prediction arrangement:
        order,
        pred_len,
        # experiment setup:
        skip,
        fix_all_rngs_to,
        fix_rngs_before_each,
        # plotting:
        model_desc='',
        fig_size=(10, 8)
):
    if fix_all_rngs_to is not False:
        helpers.fix_all_rngs(fix_all_rngs_to)

    # extract data
    data = etl.extract_data_matrix_from_df(
        data_df,
        features,
        from_i,
        from_i + train_sz + test_sz
    )
    data.setflags(write=False)
    train_data = data[:train_sz]
    train_data.setflags(write=False)
    test_data = data[train_sz:]
    test_data.setflags(write=False)
    assert data.shape == (train_sz + test_sz, len(features))

    # normalize data
    scaler = None
    if normalize == 'data':
        data, scaler = etl.scaled_data(data, train_sz)

    # with helpers.timing('train'):
    #     model = ARIMA(train_data, order)
    #     res = model.fit()

    pred_seqs = []
    losses = np.zeros(4, dtype=np.float32)
    losses_count = 0

    step = 1 if not skip else pred_len
    for i in range(0, len(test_data) - pred_len, step):
        # ys = res.forecast(i + pred_len)[0][i:]
        ys = arima_predict(data[: train_sz + i], order, pred_len)
        pred_seqs.append(ys.copy())

        loss = compute_dacc_loss_vs_ct_y(
            ys[-1],
            test_data[i + pred_len, 0],
            test_data[i, 0]
        )
        if loss is not None:
            # print("~ losses: {}, ~ loss: {}".format(losses.shape, len(loss)))
            try:
                losses += np.array(loss)
            except:
                import pdb
                pdb.set_trace()
            losses_count += 1
        else:
            print("WARNING: can't compute loss for prediction at index %d" % (
                train_sz + i + pred_len,
            ))

    losses[0] /= losses_count
    losses[1] /= losses_count
    losses[2] = np.sqrt(losses[2] / losses_count)
    losses[3] = np.sqrt(losses[3] / losses_count)

    # stats
    losses = np.array(losses, dtype=np.float32)
    # print("losses ~", losses.shape)
    dir_acc = losses[0] * 100
    rmse = losses[2]
    rmse_cp = losses[3]
    stats_desc = (
        'Dir. Acc.: {:.4f}%'.format(dir_acc) +
        ', ' +
        'RMSE: {:.4f} vs. {:.4f} for CP'.format(
            rmse,
            rmse_cp,
        )
    )

    # plotting setup
    plt.figure(figsize=fig_size, facecolor='white')
    desc = (
        'train on {from_idx}:{to_idx} bc@5m data, '
        'test on next {test_sz}, normalize={normalize}'
    ).format(
        from_idx=from_i,
        to_idx=from_i + train_sz,
        test_sz=test_sz,
        normalize=normalize,
    )
    plt.suptitle(
        'ARIMA{} {} pred_len={}'.format(
            order,
            ','.join(features),
            pred_len
        ) +
        '\n' + desc + '\n' + stats_desc +
        '\n\n'
    )

    pred_len = len(pred_seqs[0])
    plt.plot(test_data, color='b', linewidth=1)
    for i, preds in enumerate(pred_seqs):
        plt.plot(
            np.arange(
                i,
                pred_len + i,
            ),
            preds,
            linewidth=1,
            color='m',
            alpha=0.4
        )


def run_walk_forward_validation_rnn_retraining(
    # data to extract:
    data_df,
    features,
    from_i,
    train_sz,
    test_sz,
    # data processing:
    normalize,  # 'seqs' | 'data'
    detrend,
    # model and prediction arrangement:
    seq_len,
    pred_len,
    model_maker,
    # training:
    epochs,
    batch_size,
    shuffle,
    # experiment setup:
    times,
    skip,
    fix_all_rngs_to,
    fix_rngs_before_each,
    # plotting:
    plot=True,
    model_desc='',
    fig_size=(10, 8),
    fast=False
):
    assert normalize in {'seqs', 'data'}
    assert times in {1, 4}, "plotting support for other values not implemented"

    if fix_all_rngs_to is not False:
        helpers.fix_all_rngs(fix_all_rngs_to)

    # extract data
    data = etl.extract_data_matrix_from_df(
        data_df,
        features,
        from_i,
        from_i + train_sz + test_sz
    )
    data.setflags(write=False)
    test_data = data[train_sz:]
    test_data.setflags(write=False)
    assert data.shape == (train_sz + test_sz, len(features))

    # normalize data
    scaler = None
    if normalize == 'data':
        data, scaler = etl.scaled_data(data, train_sz)
    # detrend
    if detrend:
        data = etl.detrended_data(data)

    ###
    # plt.plot(data)
    # plt.show()
    print("data ~ %s" % (data.shape,))

    # make sequences
    seqs = etl.make_sliding_seqs(data, seq_len)

    # normalize seqs
    if normalize == 'seqs':
        seqs = etl.normalized_seqs_cols(seqs)

    model_description = (model_desc or getattr(model_maker, 'desc', '')) + \
        ' {} batch_size={} epochs={} seq_len={} pred_len={}'.format(
            ','.join(features),
            batch_size,
            epochs,
            seq_len,
            pred_len
    )

    # plotting setup
    if plot:
        plt.figure(figsize=fig_size, facecolor='white')
        desc = (
            'train on {from_idx}:{to_idx} bc@5m data shuffle={shuffle}, '
            'test on next {test_sz}, normalize={normalize}'
        ).format(
            from_idx=from_i,
            to_idx=from_i + train_sz,
            shuffle=int(shuffle),
            test_sz=test_sz,
            normalize=normalize,
        )
        plt.suptitle(
            model_description +
            (('\n' + desc) if desc else '') +
            '\n\n\n'
        )
        if times > 1:
            plt.subplots_adjust(top=0.8, hspace=0.5)
        # show a nice grid if multilple
        rows = cols = np.ceil(np.sqrt(times))

    out = dict(
        model_description=model_description,
        train_from=from_i,
        train_to=from_i + train_sz,
        test_on=test_sz,
        shuffle=shuffle,
        normalize=normalize,
    )

    # possibly multiple runs
    for i in range(times):

        if fix_rngs_before_each:
            helpers.fix_all_rngs(fix_all_rngs_to)

        pred_seqs = []
        losses = np.zeros(4, dtype=np.float32)  # aggregate compute_loss results
        losses_count = 0

        step = 1 if skip is False else pred_len
        for idx in range(0, seqs.shape[0] - train_sz - pred_len, step):

            x_train_seqs, y_train, x_test_seqs, y_test =\
                etl.make_train_test_seqs(
                    seqs[idx: idx + train_sz + 1],
                    train_sz,
                    shuffle
                )
            x_train_seqs.setflags(write=False)
            y_train.setflags(write=False)
            x_test_seqs.setflags(write=False)
            y_test.setflags(write=False)

            print("""\n====== {}
                x_train_seqs ~ {} [...{}]
                y_train ~ {} [...{}]
                x_test_seqs ~ {} [...{}]
                y_test ~ {} [...{}]
                """.format(
                idx,
                x_train_seqs.shape,
                x_train_seqs[-1, -1, 0],
                y_train.shape,
                y_train[-1],
                x_test_seqs.shape,
                x_test_seqs[-1, -1, 0],
                y_test.shape,
                y_test[-1]
            ))

            model = model_maker(seq_len, len(features))

            training_history = model.fit(
                x_train_seqs,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.05,
                shuffle=False,
            )

            final_training_loss = -1
            if training_history:
                final_training_loss = training_history.history['loss'][-1]

            ys = np.zeros(pred_len, dtype=np.float32)
            xs = x_test_seqs[0].copy()
            for pidx in range(pred_len):
                r = model.predict(xs[None, :, :])[0, 0]
                ys[pidx] = r
                xs[0: -1, :] = xs[1:, :]
                xs[0: -1, :] = xs[1:, :]
                xs[-1, 0] = ys[pidx]

            pred_seqs.append(ys)

            print("\n ~~~~ pred_y={}, target_y={}, start_y={}".format(
                ys[-1],
                y_test[0, 0],
                x_test_seqs[0][-1, 0]
            ))
            loss = compute_dacc_loss_vs_ct_y(
                ys[-1],
                y_test[0, 0],
                x_test_seqs[0][-1, 0]
            )
            if loss is not None:
                losses += loss
                losses_count += 1
            else:
                print("WARNING: can't compute loss for prediction at index %d" % (
                    i * seq_len + pred_len,
                ))

        losses[0] /= losses_count
        losses[1] /= losses_count
        losses[2] = np.sqrt(losses[2] / losses_count)
        losses[3] = np.sqrt(losses[3] / losses_count)

        # denormalize predictionscompute_dacc_loss_vs_ct_y
        if normalize == 'data':
            for preds in pred_seqs:
                # hack because the scaler expects a number of columns equal to
                # no. of features, but we only predict feature 0
                if len(features) == 1:
                    scaler_input = preds.reshape((-1, 1))
                else:
                    scaler_input = preds.reshape((-1, 1)).repeat(
                        len(features), axis=1
                    )
                preds[:] = scaler.inverse_transform(scaler_input)[:, 0]
        elif normalize == 'seqs':
            for preds_i, preds in enumerate(pred_seqs):
                preds += 1
                preds *= test_data[preds_i if not skip else preds_i * seq_len, 0]

        # readd trend
        if detrend:
            if not skip:
                raise NotImplementedError(
                    'Using read_trend with no_skip=True is not implemented')
            for preds_i, preds in enumerate(pred_seqs):
                preds[0] += test_data[
                    (preds_i + 1) * seq_len - 1,
                    0
                ]
                for pidx in range(1, len(preds)):
                    preds[pidx] += preds[pidx - 1]

        # no more changes to predictions from this onwards
        pred_seqs = np.array(pred_seqs)
        pred_seqs.setflags(write=False)

        # process losses data for printing in (sub)plot title
        dir_acc, _, rmse, rmse_cp = losses
        dir_acc *= 100
        # losses = np.array(losses, dtype=np.float32)
        # if not skip:
        #     dir_acc = losses[0] * 100
        #     rmse = losses[2]
        #     rmse_cp = losses[3]
        # else:
        #     if not fast:
        #         dir_acc = np.average(losses[:, 0]) * 100
        #         rmse = np.sqrt(np.average(losses[:, 2]))
        #         rmse_cp = np.sqrt(np.average(losses[:, 3]))
        #     else:
        #         dir_acc, _, rmse, rmse_cp = losses

        # return results/stats
        out.update(
            training_loss='{:.4e})'.format(final_training_loss),
            rmse='{:.4f}%'.format(rmse),
            rmse_cp='{:.4f}%'.format(rmse_cp),
            dir_acc='{:.4f}%'.format(dir_acc),
        )

        if plot:
            # next subplot if multiple
            if times > 1:
                plt.subplot(rows, cols, i + 1)
            # plot title
            plt.title(
                'Dir. Acc.: {:.4f}%'.format(dir_acc) +
                ('\n' if times > 1 else ', ') +
                'RMSE: {:.4f} vs. {:.4f} for CP'.format(
                    rmse,
                    rmse_cp,
                ) +
                ('\n' if times > 1 else ' ') +
                '(Loss: {:.4e})'.format(final_training_loss)
            )
            # plot predictions test data with predictions
            with helpers.timing('plot incremental predictions'):
                if not skip:
                    plot_pred_seqs_all(
                        test_data[:, 0], seq_len, pred_seqs)
                else:
                    plot_pred_seqs(
                        test_data[:, 0], seq_len, pred_seqs)

    if plot:
        plt.show()

    return out
