import numpy as np


# Implementing methods from functions.h
def fftderiv(in_elem, order: int, sigma: int, n: int):
    df = 1 / float(n)
    aux = n * df
    n_p = int(n / 2 + 1)
    f = np.zeros(n_p, dtype=np.complex)

    for i in range(n_p):
        f[i] = i * df

    for i in range(int(n/2-1), n_p):
        f[i] = f[i] - aux

    g_lambda = np.vectorize(lambda x: np.complex(np.exp(-2*np.power(sigma*np.pi*x, 2)), 0))
    g = g_lambda(f)

    if order == 1:
        u_lambda = np.vectorize(lambda x: np.complex(0, 2.*np.pi*x))
        u = u_lambda(f)
    else:
        u_lambda = np.vectorize(lambda x: np.complex(-1*np.power(2.*np.pi*x, 2), 0))
        u = u_lambda(f)

    f = np.fft.rfft(in_elem)
    f = f * g * u
    return np.fft.irfft(f)


def entropy(k):
    s = 0.0
    for k_ in k:
        if k_ > 0.0:
            s += k_ * np.log(k_)
    return (s * s) / 2.0
