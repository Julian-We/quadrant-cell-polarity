def get_polarization_phases(df, polarity_column, min_phase_length):
    """Flag runs of >= min_phase_length consecutive frames with a positive diff as polarizing."""
    diff_col = f"{polarity_column}_diff"
    if diff_col not in df.columns:
        df[diff_col] = df[polarity_column].diff()

    rising = df[diff_col] > 0
    run_id = (~rising).cumsum()
    run_length = rising.groupby(run_id).transform("sum")
    df["polarizing"] = rising & (run_length >= min_phase_length)

    return df


def get_stalling_phases(df, polarity_column, min_phase_length, sigma):
    """Flag runs of >= min_phase_length consecutive frames whose diff stays within sigma stds of zero."""
    diff_col = f"{polarity_column}_diff"
    if diff_col not in df.columns:
        df[diff_col] = df[polarity_column].diff()

    threshold = sigma * df[diff_col].std()
    stalling_frame = df[diff_col].abs() <= threshold

    run_id = (~stalling_frame).cumsum()
    run_length = stalling_frame.groupby(run_id).transform("sum")
    df["stalling"] = stalling_frame & (run_length >= min_phase_length)

    return df


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
