import numpy as np

# implementando os métodos do funtions.h
def fftderiv(in_elem, ord: int, sigma: int, n: int):
    df = 1 / float(n)
    aux = n * df
    n_p = int(n / 2 + 1)
    f = np.zeros(n_p, dtype=np.complex)

    for i in range(n_p):
        f[i] = i * df

    for i in range(int(n/2-1), n_p):
        f[i] = f[i] - aux

    g_lambda = np.vectorize(lambda x: np.complex(np.exp(-2*np.power(sigma*np.pi*x, 2)), 0))
    G = g_lambda(f)

    U = None
    if ord == 1:
        u_lambda = np.vectorize(lambda x: np.complex(0, 2.*np.pi*x))
        U = u_lambda(f)
    else:
        u_lambda = np.vectorize(lambda x: np.complex(-1*np.power(2.*np.pi*x, 2), 0))
        U = u_lambda(f)

    F = np.fft.rfft(in_elem)
    F = F * G * U
    return np.fft.irfft(F)

def entropy(k):
    sum = 0.0
    for k_ in k:
        if k_ > 0.0:
            sum += k_ * np.log(k_)
    return (sum * sum) / 2.0