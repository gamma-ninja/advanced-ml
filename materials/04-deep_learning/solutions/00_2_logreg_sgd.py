pobj = []
l_predict_proba = []


def logger(i, w, b, loss):
    pobj.append(loss)
    l_predict_proba.append(
        partial(predict_proba_logreg, w=w.copy(), b=b.copy())
    )

    if i % 100 == 0:
        y_pred = predict_proba_logreg(w, b, X_val).argmax(axis=1)
        print(
            f"Iterarion {i} - validation accuracy: {np.mean(y_pred == y_val)}"
        )


# Initialize the weights of the model
w = np.array([[3., 2], [-2., 3.], [5., 0]])
b = np.array([.1, -.1, .3])

# Parameters of the SGD
lr = 2e-2  # learning rate
batch_size = 16
patience = 100

# Constants for the early stopping
best_iter = 0
best_loss_val = 1e100
best_params = (w.copy(), b.copy())


n_samples, n_features = X_train.shape
rng = np.random.RandomState(72)

for it in range(5000):
    ##############################
    # TODO
    idx = rng.choice(range(n_samples), size=batch_size,
                     replace=False)
    loss, grad_w, grad_b = log_likelihood_and_grad(
        w, b, X_train[idx], y_ohe_train[idx]
    )
    w -= lr * grad_w
    b -= lr * grad_b

    val_loss, *_ = log_likelihood_and_grad(
        w, b, X_val, y_ohe_val
    )

    # END TODO
    ##########################

    # Logger to monitor the progress
    # of the training.
    logger(it, w, b, loss)

    # Early stopping mechanism:
    # - store the best loss and params
    # - stop if no progress after patience iterations
    if best_loss_val > val_loss:
        best_iter = it
        best_loss_val = val_loss
        best_params = (w.copy(), b.copy())

    if it - best_iter >= patience:
        print(
            "Stopping as no progress has been made "
            f"for {patience} iterations"
        )
        w, b = best_params
        break

plt.plot(pobj)

y_pred = predict_proba_logreg(w, b, X_val).argmax(axis=1)
print(f"Final validation accuracy: {np.mean(y_pred == y_val)}")

show_decision_boundary(partial(predict_proba_logreg, w=w, b=b), data=(X, y))
