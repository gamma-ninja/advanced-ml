
class Network():
    def __init__(self, model):
        self.layers = [
            Linear(model['W1'], model['b1']), Relu(),
            Linear(model['W2'], model['b2']),
        ]
        self.loss = CrossEntropyLogit()

    def __call__(self, X, y):

        X, y = Parameter(X), Parameter(y)
        ######################
        # TODO

        for layer in self.layers:
            X = layer(X)
        return self.loss(X, y)

        # END TODO
        ########################

    def backward(self):

        ######################
        # TODO

        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()

        # END TODO
        ########################
