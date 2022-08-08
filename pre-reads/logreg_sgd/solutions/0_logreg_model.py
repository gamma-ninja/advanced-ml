from scipy.special import softmax


def predict_proba_logreg(w, b, X):
    """Return the proba of being in one of two classes.

    Parameters
    ----------
    w : ndarray, shape (n_classes, n_features)
        parameters for the linear model.
    b : ndarray, shape (n_classes,)
        biases for the linear model.
    X : ndarray, shape (n_samples, n_features)
        input data

    Return
    ------
    y_proba : ndarray, shape (n_samples, n_classes)
        probability of being from each class according to
        the linear model w.
    """

    ####################
    # TO DO
    logit = X @ w.T + b
    y_proba = softmax(logit, axis=1)

    # END TO DO
    ####################
    return y_proba
