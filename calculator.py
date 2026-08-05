import numpy as np


def get_polarization_phases(df, polarity_column, min_phase_length):
    """Flag runs of >= min_phase_length consecutive frames with a positive diff as polarizing."""
    diff_col = f"{polarity_column}_diff"
    if diff_col not in df.columns:
        df[diff_col] = df[polarity_column].diff()

    rising = df[diff_col].shift(-1) > 0
    run_id = (~rising).cumsum()
    run_length = rising.groupby(run_id).transform("sum")
    df["polarizing"] = rising & (run_length >= min_phase_length)

    return df


def quantify_phases(phases, phase_prefix):
    return {
        f"{phase_prefix}_phases_count": len(phases["duration"]),
        f"{phase_prefix}_phases_mean_duration": np.mean(phases["duration"]),
        f"{phase_prefix}_phases_mean_delta": np.mean(phases["duration"]),
        f"{phase_prefix}_phases_max_delta": np.max(phases["duration"]),
        f"{phase_prefix}_phases_mean_slope": np.mean(phases["slope"]),
    }


def get_stalling_phases(df, polarity_column, min_phase_length, sigma):
    """Flag runs of >= min_phase_length consecutive frames whose diff stays within sigma stds of zero."""
    diff_col = f"{polarity_column}_diff"
    if diff_col not in df.columns:
        df[diff_col] = df[polarity_column].diff()

    threshold = sigma * df[diff_col].std()
    stalling_frame = df[diff_col].shift(-1).abs() <= threshold

    run_id = (~stalling_frame).cumsum()
    run_length = stalling_frame.groupby(run_id).transform("sum")
    df["stalling"] = stalling_frame & (run_length >= min_phase_length)

    return df


def true_phases(df, column, polarity_column=None):
    m = df[column].to_numpy(dtype=bool)
    d = np.diff(np.concatenate(([False], m, [False])).astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # exclusive
    last = ends - 1  # inclusive

    x = df.index.to_numpy()
    out = {
        "start": starts,
        "end": ends,
        "n_samples": ends - starts,
        "x_start": x[starts],
        "x_end": x[last],
    }
    out["duration"] = out["x_end"] - out["x_start"]

    if polarity_column is not None:
        p = df[polarity_column].to_numpy(dtype=float)
        out["p_start"] = p[starts]
        out["p_end"] = p[last]
        out["delta"] = out["p_end"] - out["p_start"]

        dur = out["duration"]
        if np.issubdtype(dur.dtype, np.timedelta64):
            dur_s = dur / np.timedelta64(1, "s")
        else:
            dur_s = dur.astype(float)
        out["slope"] = out["delta"] / np.where(dur_s == 0, np.nan, dur_s)

    return out


def get_local_extrems(df, polarity_column):
    """Flag local maxima/minima by sign change in the diff immediately before vs. after each point."""
    diff_col = f"{polarity_column}_diff"
    if diff_col not in df.columns:
        df[diff_col] = df[polarity_column].diff()

    diff_before = df[diff_col]
    diff_after = df[diff_col].shift(-1)

    df["local_max"] = (diff_before > 0) & (diff_after < 0)
    df["local_min"] = (diff_before < 0) & (diff_after > 0)

    return df


def get_above_polar_threshold(df, polarity_column, min_phase_length, threshold):
    """Flag runs of >= min_phase_length consecutive frames above threshold as polar."""
    above = df[polarity_column] > threshold
    run_id = (~above).cumsum()
    run_length = above.groupby(run_id).transform("sum")
    df["polar"] = above & (run_length >= min_phase_length)

    return df


def get_time_since_local_min_to_polar(df, polarity_column):
    """For each polar phase, count timepoints since the last local minimum before it started.
    Phases with no preceding local minimum (e.g. the first phase) are discarded."""
    phases = true_phases(df, "polar", polarity_column)
    starts = phases["start"]

    local_min_idx = np.flatnonzero(df["local_min"].to_numpy())

    pos = np.searchsorted(local_min_idx, starts, side="right") - 1
    valid = pos >= 0

    starts = starts[valid]
    last_min_idx = local_min_idx[pos[valid]]

    x = df.index.to_numpy()
    return {
        "polar_start": starts,
        "local_min_idx": last_min_idx,
        "n_timepoints": starts - last_min_idx,
        "x_polar_start": x[starts],
        "x_local_min": x[last_min_idx],
    }
