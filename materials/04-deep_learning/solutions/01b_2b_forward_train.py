
def forward_train(model, X):
    """Output of the model without the softmax.

    Also returns the intermediate activations.

    Parameters
    ----------
    model : dict
        Dict containing all the network parameters.
    X : ndarray, shape (batch_size, n_features)
        The input of the model.

    Returns
    -------
    logit : ndarray, shape (batch_size, n_classes)
        The unnormalized output of the model
    ctx : dict
        The intermediate values in the network.
    """
    ##################################
    # TODO

    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']

    z1 = X.dot(W1.T) + b1[None]
    a1 = np.maximum(z1, 0)
    z2 = a1.dot(W2.T) + b2[None]

    ctx = dict(z2=z2, a1=a1, z1=z1, X=X)

    # END TODO
    ####################################

    return z2, ctx
