# Gradient of the function f computed with forward diff
def g_forward(x, n=4):
    ##############
    # TODO

    v, dv = x, 1
    for _ in range(n):
        v, dv = 1 + 2 / v, -2 * dv / v ** 2

    # END TODO
    ############
    return dv


x = 2
g_forward(x), approx_fprime(x, f, 1e-9)[0]
