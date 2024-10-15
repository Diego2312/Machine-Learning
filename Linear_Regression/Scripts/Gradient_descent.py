import pandas as pd




def update_w_b(x, y, w, b, alpha):
    dw = 0.0
    db = 0.0
    N = len(x)

    for i in range(N):
        dw += -2 * x[i] * (y[i] - ((x[i] * w) + b))
        db += -2 * (y[i] - ((x[i] * w) + b))

    w = w - (1 / float(N)) * dw * alpha
    b = b - (1 / float(N)) * db * alpha

    return w, b