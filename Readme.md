
# Description

- `tools/PyFIG6/pyFIG6.py` provides APIs for loudness compensation based on the FIG6 fitting algorithm.
- `tools/PyHASQI/HASQI_revised.py` implements computation of the HASQI metric using the NumPy library.
- `data/Patient_Information` contains the hearing-impaired audiograms used in our work.


# Audio file naming convention

For each test audio, there are four files, each suffixed with `_src`, `_nearend`, `_nearend_fig6`, and `_target`. 
Among them:
- `_src` is the original clean audio;
- `_nearend` indicates the file containing noisy;
- `_nearend_fig6` refer to the noisy file compensated by FIG6;
- `_target` is the target compensated and denoised audio.
