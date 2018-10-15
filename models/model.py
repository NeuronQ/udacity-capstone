class Model(object):

    def fit(self, x_train_seqs, y_train,
            batch_size, epochs, validation_split, shuffle):
        """
        Parameters
        ----------
        x_train_seqs : [N, seq_len, n_features] np.array
        y_train : [N, 1] np.array
        batch_size : int
        epochs : int
        validation_split : float (0..1)
        shuffle : bool

        Returns
        -------
        NOTHING
        """
        raise NotImplementedError()

    def predict(self, xs):
        """
        Parameters
        ----------
        xs : [N, seq_len, n_features] np.array

        Returns
        -------
        [N, 1] np.array
        """
        raise NotImplementedError()
