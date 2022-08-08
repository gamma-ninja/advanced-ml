
class Linear():
    def __init__(self, w, b):
        self.w, self.b = w, b

    def __call__(self, inp):
        ######################
        # TODO

        self.inp = inp
        self.out = Parameter(inp @ self.w.T + self.b)

        # END TODO
        ########################
        return self.out

    def backward(self):

        ######################
        # TODO

        self.inp.g = self.out.g @ self.w
        self.w.g = self.inp.T @ self.out.g / self.inp.shape[0]
        self.b.g = self.out.g.mean(axis=0)

        # END TODO
        ########################
