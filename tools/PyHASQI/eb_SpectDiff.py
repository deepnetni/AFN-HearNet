import numpy as np

def eb_SpectDiff(xSL,ySL):
    nbands = len(xSL)
    x = 10 ** (xSL / 20)
    y = 10 ** (ySL / 20)
    xsum = np.sum(x)
    x = x / xsum
    ysum = np.sum(y)
    y = y / ysum

    dloud = np.zeros(3)
    d = x - y
    dloud[0] = np.sum(np.abs(d))
    dloud[1] = nbands * np.std(d, axis=0)
    dloud[2] = np.max(np.abs(d))

    dnorm = np.zeros(3)
    d = (x - y) / (x + y)
    dnorm[0] = np.sum(np.abs(d))
    dnorm[1] = nbands * np.std(d, axis=0)
    dnorm[2] = np.max(np.abs(d))

    dslope = np.zeros(3)
    dx = (x[1:] - x[:(nbands - 1)])
    dy = (y[1:] - y[:(nbands - 1)])
    d = dx - dy
    dslope[0] = np.sum(np.abs(d))
    dslope[1] = nbands * np.std(d, axis=0)
    dslope[2] = np.max(np.abs(d))

    return dloud, dnorm, dslope