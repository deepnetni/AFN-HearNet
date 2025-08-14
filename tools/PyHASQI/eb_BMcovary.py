import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.signal


def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """

    py = np.pad(y.conj(), 2 * maxlag, mode="constant")
    T = as_strided(
        py[2 * maxlag :],
        shape=(2 * maxlag + 1, len(y) + 2 * maxlag),
        strides=(-py.strides[0], py.strides[0]),
    )
    px = np.pad(x, maxlag, mode="constant")
    return T.dot(px)


def eb_BMcovary(xBM, yBM, segsize, fsamp):
    small = 1.0e-30
    lagsize = 1.0
    maxlag = round(lagsize * (0.001 * fsamp))

    nwin = round(segsize * (0.001 * fsamp))

    window = np.hanning(nwin)

    # win_full = np.correlate(window, window, 'full')
    # mid = int((len(win_full) - 1) / 2)
    # wincorr = 1.0 / win_full[(mid - maxlag): (mid + maxlag + 1)]
    # winsum2 = 1.0 / np.sum(window**2)

    wincorr = 1.0 / crosscorrelation(window, window, maxlag)
    winsum2 = 1.0 / np.sum(window**2)

    nhalf = int(nwin / 2)

    halfwindow = window[nhalf:]
    # halfwin_full = np.correlate(halfwindow, halfwindow, 'full')
    # halfmid = int((len(halfwin_full) - 1) / 2)
    # halfcorr = 1.0 / halfwin_full[(halfmid - maxlag): (halfmid + maxlag + 1)]
    # halfsum2 = 1.0 / np.sum(halfwindow**2)

    halfcorr = 1.0 / crosscorrelation(halfwindow, halfwindow, maxlag)
    halfsum2 = 1.0 / np.sum(halfwindow**2)

    nchan = xBM.shape[0]
    frame_num = int(1 + (xBM.shape[1] - nwin) // nhalf) + 1
    sigMSx = np.zeros((nchan, frame_num))
    sigMSy = np.zeros((nchan, frame_num))
    sigcov = np.zeros((nchan, frame_num))

    for k in range(nchan):
        for i in range(frame_num):
            if i == 0:
                segx = xBM[k, :nhalf] * halfwindow
                segy = yBM[k, :nhalf] * halfwindow
                segx = segx - np.mean(segx)
                segy = segy - np.mean(segy)
                MSx = np.sum(segx**2) * halfsum2
                MSy = np.sum(segy**2) * halfsum2
                Mxy = np.max(np.abs(crosscorrelation(segx, segy, maxlag) * halfcorr))

                Mxy_np = np.correlate(segx, segy, mode="full")[
                    len(segy) - maxlag - 1 : len(segy) + maxlag
                ]
                Mxy_fft = scipy.signal.fftconvolve(segx, segy[::-1], mode="full")[
                    len(segy) - maxlag - 1 : len(segy) + maxlag
                ]

                if (MSx > small) and (MSy > small):
                    sigcov[k, i] = Mxy / np.sqrt(MSx * MSy)
                else:
                    sigcov[k, i] = 0.0
                sigMSx[k, i] = MSx
                sigMSy[k, i] = MSy
            elif i == frame_num - 1:
                segx = xBM[k, i * nhalf :] * window[:nhalf]
                segy = yBM[k, i * nhalf :] * window[:nhalf]
                segx = segx - np.mean(segx)
                segy = segy - np.mean(segy)
                MSx = np.sum(segx**2) * halfsum2
                MSy = np.sum(segy**2) * halfsum2
                Mxy = np.max(np.abs(crosscorrelation(segx, segy, maxlag) * halfcorr))

                if (MSx > small) and (MSy > small):
                    sigcov[k, i] = Mxy / np.sqrt(MSx * MSy)
                else:
                    sigcov[k, i] = 0.0
                sigMSx[k, i] = MSx
                sigMSy[k, i] = MSy
            else:
                segx = xBM[k, i * nhalf : (i * nhalf + nwin)] * window
                segy = yBM[k, i * nhalf : (i * nhalf + nwin)] * window
                segx = segx - np.mean(segx)
                segy = segy - np.mean(segy)
                MSx = np.sum(segx**2) * winsum2
                MSy = np.sum(segy**2) * winsum2
                Mxy = np.max(np.abs(crosscorrelation(segx, segy, maxlag) * wincorr))

                if (MSx > small) and (MSy > small):
                    sigcov[k, i] = Mxy / np.sqrt(MSx * MSy)
                else:
                    sigcov[k, i] = 0.0
                sigMSx[k, i] = MSx
                sigMSy[k, i] = MSy

    sigcov = np.maximum(sigcov, 0)
    sigcov = np.minimum(sigcov, 1)

    sigMSx = 2.0 * sigMSx
    sigMSy = 2.0 * sigMSy

    return sigcov, sigMSx, sigMSy


def eb_BMcovary_new(xBM, yBM, segsize, fsamp):
    small = 1.0e-30
    lagsize = 1.0
    maxlag = round(lagsize * (0.001 * fsamp))

    nwin = round(segsize * (0.001 * fsamp))

    window = np.hanning(nwin)

    wincorr = 1.0 / crosscorrelation(window, window, maxlag)
    winsum2 = 1.0 / np.sum(window**2)

    nhalf = int(nwin / 2)

    halfwindow = window[nhalf:]

    halfcorr = 1.0 / crosscorrelation(halfwindow, halfwindow, maxlag)
    halfsum2 = 1.0 / np.sum(halfwindow**2)

    nchan = xBM.shape[0]
    frame_num = int(1 + (xBM.shape[1] - nwin) // nhalf) + 1
    # sigMSx = np.zeros((nchan, frame_num))
    # sigMSy = np.zeros((nchan, frame_num))
    # sigcov = np.zeros((nchan, frame_num))

    nchan_sigMSx = np.zeros((nchan, frame_num))
    nchan_sigMSy = np.zeros((nchan, frame_num))
    nchan_sigcov = np.zeros((nchan, frame_num))

    for i in range(frame_num):

        if i == 0:
            head_segx = np.zeros((nchan, nhalf))
            head_segx[:, :nhalf] = xBM[:, :nhalf] * halfwindow
            head_segx = head_segx - np.expand_dims(np.mean(head_segx, axis=1), axis=1)

            head_segy = np.zeros((nchan, nhalf))
            head_segy[:, :nhalf] = yBM[:, :nhalf] * halfwindow
            head_segy = head_segy - np.expand_dims(np.mean(head_segy, axis=1), axis=1)

            head_MSx = np.sum(head_segx**2, axis=1) * halfsum2
            head_MSy = np.sum(head_segy**2, axis=1) * halfsum2
            head_Mxy = np.max(
                np.abs(
                    scipy.signal.fftconvolve(head_segx, head_segy[:, ::-1], mode="full", axes=1)[
                        :, head_segx.shape[1] - maxlag - 1 : head_segx.shape[1] + maxlag
                    ]
                    * halfcorr
                ),
                axis=1,
            )

            nchan_sigMSx[:, i] = head_MSx
            nchan_sigMSy[:, i] = head_MSy
            nchan_sigcov[:, i] = head_Mxy / np.sqrt(head_MSx * head_MSy + small)

        elif i == frame_num - 1:
            tail_segx = np.zeros((nchan, nhalf))
            tail_segx[:, :nhalf] = xBM[:, i * nhalf : (i + 1) * nhalf] * window[:nhalf]
            tail_segx = tail_segx - np.expand_dims(np.mean(tail_segx, axis=1), axis=1)

            tail_segy = np.zeros((nchan, nhalf))
            tail_segy[:, :nhalf] = yBM[:, i * nhalf : (i + 1) * nhalf] * window[:nhalf]
            tail_segy = tail_segy - np.expand_dims(np.mean(tail_segy, axis=1), axis=1)

            tail_MSx = np.sum(tail_segx**2, axis=1) * halfsum2
            tail_MSy = np.sum(tail_segy**2, axis=1) * halfsum2
            tail_Mxy = np.max(
                np.abs(
                    scipy.signal.fftconvolve(tail_segx, tail_segy[:, ::-1], mode="full", axes=1)[
                        :, tail_segx.shape[1] - maxlag - 1 : tail_segx.shape[1] + maxlag
                    ]
                    * halfcorr
                ),
                axis=1,
            )

            nchan_sigMSx[:, i] = tail_MSx
            nchan_sigMSy[:, i] = tail_MSy
            nchan_sigcov[:, i] = tail_Mxy / np.sqrt(tail_MSx * tail_MSy + small)

        else:

            segx = xBM[:, i * nhalf : (i * nhalf + nwin)] * window
            segx = segx - np.expand_dims(np.mean(segx, axis=1), axis=1)

            segy = yBM[:, i * nhalf : (i * nhalf + nwin)] * window
            segy = segy - np.expand_dims(np.mean(segy, axis=1), axis=1)

            MSx = np.sum(segx**2, axis=1) * winsum2
            MSy = np.sum(segy**2, axis=1) * winsum2
            Mxy = np.max(
                np.abs(
                    scipy.signal.fftconvolve(segx, segy[:, ::-1], mode="full", axes=1)[
                        :, segx.shape[1] - maxlag - 1 : segx.shape[1] + maxlag
                    ]
                    * wincorr
                ),
                axis=1,
            )

            nchan_sigMSx[:, i] = MSx
            nchan_sigMSy[:, i] = MSy
            nchan_sigcov[:, i] = Mxy / np.sqrt(MSx * MSy + small)

    nchan_sigcov = np.maximum(nchan_sigcov, 0)
    nchan_sigcov = np.minimum(nchan_sigcov, 1)

    nchan_sigMSx = 2.0 * nchan_sigMSx
    nchan_sigMSy = 2.0 * nchan_sigMSy

    return nchan_sigcov, nchan_sigMSx, nchan_sigMSy
