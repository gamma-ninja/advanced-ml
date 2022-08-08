
def log_likelihood_and_grad(w, b, X, y):
    """Log-likelihood of the logistic regression model and gradient.

    Parameters
    ----------
    w : ndarray, shape (n_classes, n_features)
        parameters for the linear model.
    b : ndarray, shape (n_classes,)
        biases for the linear model.
    X : ndarray, shape (n_samples, n_features)
        input data
    y : ndarray, shape (n_samples, n_classes)
        output targets one hot encoded.

    Returns
    -------
    loss : log-likelihood of the logreg model
    grad_w : gradient of the model parameters w.
    grad_b : gradient of the model parameters b.
    """

    #####################
    # TO DO

    y_proba = predict_proba_logreg(w, b, X)

    loss = - np.log(y_proba[y == 1]).mean()

    grad_logit = (y_proba - y)
    grad_w = grad_logit.T @ X
    grad_b = grad_logit.sum(axis=0)

    # END TO DO
    #####################
    return loss, grad_w, grad_b
