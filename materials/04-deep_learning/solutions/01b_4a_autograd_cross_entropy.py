
class CrossEntropyLogit():
    def __call__(self, logit, y):
        ######################
        # TODO

        self.logit = logit
        self.y = y

        self.out = Parameter(
            -np.log(softmax(self.logit, axis=1)[y == 1]).mean()
        )

        # END TODO
        ########################
        return self.out

    def backward(self):
        ######################
        # TODO

        self.logit.g = (self.logit - self.y)

        # END TODO
        ########################
        pass
