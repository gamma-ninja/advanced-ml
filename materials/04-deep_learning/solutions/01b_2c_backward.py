
def backward(model, output_grad, ctx):
    """Output of the model without the softmax.

    Also returns the intermediate activations.

    Parameters
    ----------
    model : dict
        Dict containing all the network parameters.
    ctx : dict
        The intermediate values in the network.
    grad_output : ndarray, shape (batch_size, n_classes)
        Gradient of the loss relative to the output of the model.

    Returns
    -------
    all_grad : dict {parameter_name => gradient}
        Gradient of each parameter of the model.
    """
    ##################################
    # TODO

    a1, X = ctx['a1'], ctx['X']

    # Gradient of the parameters of the first layer
    grad_W2 = (output_grad.T @ a1)
    grad_b2 = (output_grad).sum(axis=0)

    # Gradient of the intermediate steps
    grad_a1 = output_grad @ model['W2']
    grad_z1 = grad_a1 * (a1 > 0)

    # Gradient of the parameters of the first layer
    grad_W1 = (grad_z1.T @ X)
    grad_b1 = (grad_z1).sum(axis=0)

    all_grad = dict(W1=grad_W1, b1=grad_b1, W2=grad_W2, b2=grad_b2)

    # END TODO
    ####################################

    return all_grad
