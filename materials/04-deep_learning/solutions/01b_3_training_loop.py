#####################################
# Logger to allow simple plot for the
# evolution of the training loss and
# the decision boundary.
from copy import deepcopy

pobj = []
l_predict_proba = []


def logger(i, model, loss):
    pobj.append(loss)
    l_predict_proba.append(partial(forward, model=deepcopy(model)))

    if i % 100 == 0:
        y_pred = forward(model, X).argmax(axis=1)
        print(f"Iterarion {i} - train accuracy: {np.mean(y_pred == y)}")


############################

lr = 1e-3
batch_size = 128
rng = check_random_state(29)

# Initialize the model
model = init_model(n_features, n_neurons, n_classes, random_state=118)

for i in range(3000):

    ########################
    # TODO - implement SGD

    # Sample a batch
    idx = rng.choice(range(n_samples), size=batch_size,
                     replace=False)
    X_batch, y_batch = X[idx], y_[idx]

    # Compute the loss and gradient
    logit, ctx = forward_train(model, X_batch)
    loss, grad_output = cross_entropy_and_grad(logit, y_batch)
    all_grad = backward(model, grad_output, ctx)

    for p in model:
        model[p] -= lr * all_grad[p]

    # END TODO
    ######################

    logger(i, model, loss)

plt.plot(pobj)

show_decision_boundary(partial(forward, model), data=(X, y))
