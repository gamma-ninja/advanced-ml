from scipy.special import softmax


def forward(model, X):
    """Compute the output of the model for a given input.

    Parameters
    ----------
    model : dict
        Dictionary containing all the parameters of the model,
        `W1, b1, W2, b2` as np.ndarrays.
    X : ndarray, shape (batch_size, n_features)
        Input of the network.

    Returns
    -------
    y_proba : ndarray, shape (batch_size, n_classes)
        Output of the network. For each sample in the batch, each
        coordinate i corresponds to the probability of being of
        the class i.
    """
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']

    ####################
    # TODO

    z1 = X.dot(W1.T) + b1[None]
    a1 = np.maximum(z1, 0)
    z2 = a1.dot(W2.T) + b2[None]

    y_hat = softmax(z2, axis=1)

    # END TODO
    ##############

    return y_hat
