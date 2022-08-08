def g_backward(x, n=4):
    ##############
    # TODO
    memory = []
    v = x
    for _ in range(n):
        memory.append(v)
        v = 1 + 2 / v

    dv = 1
    for v in memory[::-1]:
        dv = -2 * dv / v ** 2

    # END TODO
    ############
    return dv


x = 2
g_backward(x), approx_fprime(x, f, 1e-9)[0]
