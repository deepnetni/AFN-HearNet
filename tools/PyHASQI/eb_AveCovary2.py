import numpy as np
from .eb_operations import eb_CenterFreq


def eb_AveCovary2(sigcov, sigMSx, thr):
    nchan = sigcov.shape[0]
    cfreq = eb_CenterFreq(nchan)
    p = [1, 3, 5, 5, 5, 5]
    fcut = 1000 * np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    fsync = np.zeros((6, nchan))

    for n in range(6):
        fc2p = fcut[n] ** (2 * p[n])
        freq2p = np.longdouble(cfreq) ** np.longdouble(2 * p[n])
        fsync[n, :] = np.sqrt(fc2p / (fc2p + freq2p))

    sigRMS = np.sqrt(sigMSx)
    sigLinear = 10 ** (sigRMS / 20)
    xsum = np.sum(sigLinear, axis=0) / nchan
    xsum = 20 * np.log10(xsum)
    index = np.where(xsum > thr)[0]
    nseg = len(index)

    if nseg <= 1:
        # print("Function eb_AveCovary: Ave signal below threshold, outputs set to 0.\n")
        avecov = 0
        syncov = np.zeros(6)
        return avecov, syncov

    sigcov = sigcov[:, index]
    sigRMS = sigRMS[:, index]
    weight = np.zeros((nchan, nseg))

    wsync1 = np.zeros((nchan, nseg))
    wsync2 = np.zeros((nchan, nseg))
    wsync3 = np.zeros((nchan, nseg))
    wsync4 = np.zeros((nchan, nseg))
    wsync5 = np.zeros((nchan, nseg))
    wsync6 = np.zeros((nchan, nseg))

    for k in range(nchan):
        for n in range(nseg):
            if sigRMS[k, n] > thr:
                weight[k, n] = 1
                wsync1[k, n] = fsync[0, k]
                wsync2[k, n] = fsync[1, k]
                wsync3[k, n] = fsync[2, k]
                wsync4[k, n] = fsync[3, k]
                wsync5[k, n] = fsync[4, k]
                wsync6[k, n] = fsync[5, k]
    csum = np.sum(np.sum(weight * sigcov))
    wsum = np.sum(np.sum(weight))
    fsum = np.zeros(6)
    ssum = np.zeros(6)

    fsum[0] = np.sum(np.sum(wsync1 * sigcov))
    ssum[0] = np.sum(np.sum(wsync1))
    fsum[1] = np.sum(np.sum(wsync2 * sigcov))
    ssum[1] = np.sum(np.sum(wsync2))
    fsum[2] = np.sum(np.sum(wsync3 * sigcov))
    ssum[2] = np.sum(np.sum(wsync3))
    fsum[3] = np.sum(np.sum(wsync4 * sigcov))
    ssum[3] = np.sum(np.sum(wsync4))
    fsum[4] = np.sum(np.sum(wsync5 * sigcov))
    ssum[4] = np.sum(np.sum(wsync5))
    fsum[5] = np.sum(np.sum(wsync6 * sigcov))
    ssum[5] = np.sum(np.sum(wsync6))

    if wsum < 1:
        avecov = 0
        syncov = fsum / ssum
        # print("Function eb_AveCovary: Signal tiles below threshold, outputs set to 0.\n")
    else:
        avecov = csum / wsum
        syncov = fsum / ssum

    return avecov, syncov
