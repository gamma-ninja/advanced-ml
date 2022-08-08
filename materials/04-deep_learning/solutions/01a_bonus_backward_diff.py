def g_backward(x, n=1000):
    v = 1
    memory = []
    for _ in range(n):
        memory.append(v)
        v = (1 - x) * v + 1
    dv, df = 0, 1
    for v in reversed(memory):
        dv -= df * v
        df = df * (1 - x)
    return dv


x = .5
g_backward(x), -1 / x**2
