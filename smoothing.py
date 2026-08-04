import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


def savgol_smooth(y, window_length=None, polyorder=3):
    """Smooth a 1D curve with a Savitzky-Golay filter.

    window_length must be odd and <= len(y); if not given, it is derived
    from len(y) so short curves (e.g. 6-point subsamples) don't crash.
    """
    y = np.asarray(y, dtype=float)
    if window_length is None:
        window_length = min(len(y) // 2 * 2 - 1, 11)
    window_length = max(window_length, polyorder + 1 + (polyorder % 2 == 0))
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(y):
        window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
    if window_length <= polyorder:
        return y.copy()
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)


def spline_smooth(x, y, s=None):
    """Smooth a 1D curve with a univariate smoothing spline, evaluated at x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) <= 3:
        return y.copy()
    spline = UnivariateSpline(x, y, s=s)
    return spline(x)


def smooth_curve(x, y, method="savgol", **kwargs):
    """Dispatch to the requested smoothing method ('savgol' or 'spline')."""
    if method == "savgol":
        return savgol_smooth(y, **kwargs)
    elif method == "spline":
        return spline_smooth(x, y, **kwargs)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
