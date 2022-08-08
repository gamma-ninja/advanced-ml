# Gradient of the function f computed with forward diff
def g_forward(x, n=1000):
    ##############
    # TODO

    v, dv = 1, 0
    for _ in range(n):
        v, dv = (1 - x) * v + 1, (1 - x) * dv - v

    # END TODO
    ############
    return dv


x = .5
g_forward(x), -1 / x**2
