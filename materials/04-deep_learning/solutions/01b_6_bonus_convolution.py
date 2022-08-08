import numpy as np


class Convolution(LayerFunction):
    def __init__(self, w, b):
        self.w, self.b = w, b
        self.k = w.shape[-1]

    def forward(self, inp):
        ####################
        # TODO

        out = np.array([[
            [
                np.convolve(inp_ik, w_jk, mode='full')
                for inp_ik, w_jk in zip(inp_i, w_j)
            ] for w_j in self.w] for inp_i in inp
        ]).sum(axis=2)

        out += self.b[None, :, None]

        # END TODO
        #####################
        return out

    def bwd(self, out, inp):

        ####################
        # TODO

        inp.g = np.array([[
            [
                np.correlate(g_ij, w_jk, mode='valid')
                for w_jk in w_j
            ] for g_ij, w_j in zip(g_i, self.w)
        ] for g_i in self.out.g]).sum(axis=1)

        self.w.g = np.array([
            [
                [np.correlate(inp_ik, g_ij, mode='valid')
                 for inp_ik in inp_i] for g_ij in g_i
            ] for inp_i, g_i in zip(inp, self.out.g)
        ]).sum(axis=0)
        self.b.g = out.g.sum(axis=(0, 2))

        # END TODO
        #####################


################################
#  Check the gradient
#
rng = np.random.RandomState(42)
# shape: n_channel_out x n_channel_in x kernel_size
w = Parameter(rng.randn(8, 3, 4) / 4)
# shape: n_channel_out
b = Parameter(rng.randn(8))
# shape: batch_size x n_channel x n_times
inp = Parameter(rng.randn(2, 3, 28) / 28)

conv = Convolution(w, b)
out = conv(inp)
v = 0.5 * (out ** 2).sum()
out.g = out
conv.backward()


# Make sure that small movement reduces the loss
for i in range(500):
    w -= 1e-4 * w.g
    b -= 1e-4 * b.g
    out = conv(inp)
    out.g = out
    conv.backward()
    loss = 0.5 * (out ** 2).sum()
    assert loss <= v, (i, v - loss)

    v = loss

print("Looks good!")
