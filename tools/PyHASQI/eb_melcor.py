import numpy as np
import math


def eb_melcor(x, y, thr, addnoise):
    nbands = x.shape[0]
    nbasis = 6
    freq = np.array(range(nbasis))
    k = np.array(range(nbands))

    cepm = np.zeros((nbands, nbasis))

    for nb in range(nbasis):
        basis = np.cos(freq[nb] * math.pi * k / (nbands - 1))
        cepm[:, nb] = basis / np.linalg.norm(basis)

    xLinear = 10 ** (x / 20)
    xsum = np.sum(xLinear, axis=0) / nbands
    xsum = 20 * np.log10(xsum)
    index = np.where(xsum > thr)[0]
    nsamp = len(index)

    vad = np.zeros(len(xsum))
    vad[index] = 1

    if nsamp <= 1:
        m1 = 0
        xy = np.zeros(nbasis)
        # print("Function eb_melcor: Signal below threshold, outputs set to 0")
        return m1, xy, vad

    x = x[:, index]
    y = y[:, index]

    xcep = np.zeros((nbasis, nsamp))
    ycep = np.zeros((nbasis, nsamp))

    for n in range(nsamp):
        for k in range(nbasis):
            xcep[k, n] = np.sum(x[:, n] * cepm[:, k])
            ycep[k, n] = np.sum(y[:, n] * cepm[:, k])

    for k in range(nbasis):
        xcep[k, :] = xcep[k, :] - np.mean(xcep[k, :])
        ycep[k, :] = ycep[k, :] - np.mean(ycep[k, :])

    xy = np.zeros(nbasis)

    small = 1.0e-30

    for k in range(nbasis):
        xsum = np.sum(xcep[k, :] ** 2)
        ysum = np.sum(ycep[k, :] ** 2)
        if (xsum < small) or (ysum < small):
            xy[k] = 0.0
        else:
            xy[k] = np.abs(np.sum(xcep[k, :] * ycep[k, :]) / np.sqrt(xsum * ysum))

    m1 = np.sum(xy[1:]) / (nbasis - 1)

    return m1, xy, vad
