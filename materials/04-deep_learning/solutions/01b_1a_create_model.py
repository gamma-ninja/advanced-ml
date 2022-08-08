
def init_model(n_features, n_neurons, n_classes, random_state=None):
    """Create a dictionary with the parameter of the 2-layer NN.

    Parameters
    ----------
    n_features : int
        Dimension of the input of the network.
    n_neurons : int
        Dimension of the hidden layer.
    n_classes : int
        Dimension of the output of the network.
    random_state : RandomState | int | None
        Random state to generate reproducible results.

    Returns
    -------
    model : dict
        Dictionary containing all the parameters of the model,
        `W1, b1, W2, b2` as np.ndarrays.
    """
    rng = check_random_state(random_state)

    ####################
    # TODO
    W1 = rng.randn(n_neurons, n_features) * np.sqrt(2 / n_features)
    b1 = np.zeros(n_neurons)
    W2 = rng.randn(n_classes, n_neurons) * np.sqrt(2 / n_neurons)
    b2 = np.zeros(n_classes)

    model = dict(W1=W1, b1=b1, W2=W2, b2=b2)
    # END TODO
    ##############

    return model
