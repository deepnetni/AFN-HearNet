import numpy as np
import librosa


def eb_EnvSmooth(env, segsize, fsamp):
    nwin = round(segsize * (0.001 * fsamp))
    nhalf = int(nwin / 2)
    window = np.hanning(nwin)

    halfwindow = window[(nhalf):]
    halfsum = sum(halfwindow)
    halfwindow_new = np.expand_dims(halfwindow, axis=0)

    wsum = sum(window)

    window_new = np.expand_dims(window, axis=0)

    nchan = env.shape[0]

    npts = env.shape[1]

    frame_num = int(1 + (env.shape[1] - nwin) // nhalf) + 1

    nn_frames = np.zeros((env.shape[0], nwin, frame_num))

    smooth = np.zeros((nchan, frame_num))

    # data = np.zeros((nchan, nwin, frame_num))

    for i in range(frame_num):

        if i == 0:
            head = np.zeros((nchan, nwin))
            head[:, :nhalf] = env[:, :nhalf] * halfwindow_new
            nn_frames[:, :, i] = head
        elif i == frame_num - 1:
            tail = np.zeros((nchan, nwin))
            tail[:, :nhalf] = env[:, i * nhalf : (i + 1) * nhalf] * np.expand_dims(
                window[:nhalf], axis=0
            )
            nn_frames[:, :, i] = tail
        else:
            nn_frames[:, :, i] = env[:, i * nhalf : (i * nhalf + nwin)] * window_new

    # data[:, :, 1:frame_num] = nn_frames[:, :, :(frame_num-1)]
    smooth[:, 0] = np.sum(nn_frames[:, :, 0], axis=1) / halfsum
    smooth[:, -1] = np.sum(nn_frames[:, :, -1], axis=1) / halfsum
    smooth[:, 1 : (frame_num - 1)] = np.sum(nn_frames[:, :, 1 : (frame_num - 1)], axis=1) / wsum

    return smooth
