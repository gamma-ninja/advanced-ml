def cross_entropy_and_grad(logit, y):
    """Output of the model without the softmax.

    Also returns the intermediate activations.

    Parameters
    ----------
    logit, y : ndarray, shape (batch_size, n_classes)
        Output of the model and target associated to these samples.

    Returns
    -------
    loss : float
        Loss of the model for these samples
    grad : ndarray, shape (batch_size, n_classes)
        Gradient of the loss relative to logit.
    """
    ##################################
    # TODO

    y_hat = softmax(logit, axis=1)
    loss = -(np.log(y_hat) * y).sum(axis=1).mean()
    grad = (y_hat - y) / y.shape[0]

    # END TODO
    ####################################
    return loss, grad
