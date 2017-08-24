import numpy as np


# PI = np.pi
PI = 3.1415927


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

    # f = np.round(f, 7)
    # print("total de elementos", n_p)
    # with open('/tmp/novo', 'w') as fh:
    #
    #     for blah in f:
    #         fh.write(str(blah))
    #         fh.write('\n')

    g_lambda = np.vectorize(lambda x: np.complex(np.exp(-2*np.power(sigma*PI*x, 2)), 0))
    g = g_lambda(f)


    # vetor g looks fine

    # with open('/tmp/novo', 'w') as fh:
    #     fh.write('valor de g\n')
    #     for blah in g:
    #         fh.write(str(blah))
    #         fh.write('\n')
    # exit()

    if order == 1:
        u_lambda = np.vectorize(lambda x: np.complex(0, 2.*PI*x))
        u = u_lambda(f)
    else:
        u_lambda = np.vectorize(lambda x: np.complex(-1*np.power(2.*PI*x, 2), 0))
        u = u_lambda(f)

    # with open('/tmp/novo', 'w') as fh:
    #     fh.write('order: ' + str(order) + '\n')
    #     fh.write('valor de U\n')
    #     for blah in u:
    #         fh.write(str(blah))
    #         fh.write('\n')
    #
    # exit()

    f = np.fft.rfft(in_elem)
    f = f * u * g
    return np.fft.irfft(f)


def entropy(k):
    s = 0.0
    for k_ in k:
        if k_ > 0.0:
            s += k_ * np.log(k_)
    return (s * s) / 2.0
