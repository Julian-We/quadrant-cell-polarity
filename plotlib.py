import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np
from calculator import true_phases

from smoothing import smooth_curve

sns.set_style("ticks")
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,  # default text
        "axes.titlesize": 8,  # subplot titles
        "axes.labelsize": 6,  # x/y axis labels
        "axes.labelweight": "bold",  # x/y axis labels
        "xtick.labelsize": 6,  # x tick labels
        "ytick.labelsize": 6,  # y tick labels
        "legend.fontsize": 6,  # legend
        "figure.titlesize": 8,  # suptitle
    }
)


def fill_true(df, column, ax, color="red", alpha=0.2, phases=None):
    if phases is None:
        phases = true_phases(df, column)
    for xs, xe in zip(phases["x_start"], phases["x_end"]):
        ax.axvspan(xs, xe, color=color, alpha=alpha)
    return phases


def plot_time_series(
    path: str | Path,
    images: dict,
    measurements: pd.DataFrame,
    measuremnt_masks: dict,
    uid: str = "cell_series",
    polarity_column: str = "Q1_norm_smooth",
    **kwargs,
):
    """
    params:
    path: str | Path - path to save the figure
    images: dict - dictionary of timepoint to image
    measurements: pd.DataFrame - dataframe of measurements
    measuremnt_masks: dict - dictionary of timepoint to measurement label images
    smoothing_method: str | None - "savgol", "spline", or None to disable smoothing overlay
    """

    if isinstance(path, str):
        path = Path(path)
    ncols = 6
    nrows = 3

    width = 1.5
    height = 1.5

    fig = plt.figure(figsize=(ncols * width, nrows * height))

    gs = fig.add_gridspec(nrows, ncols)

    # Establish the axes for the images and plots
    ax_img0 = fig.add_subplot(gs[0, 0])
    ax_img1 = fig.add_subplot(gs[0, 1])
    ax_img2 = fig.add_subplot(gs[0, 2])
    ax_img3 = fig.add_subplot(gs[0, 3])
    ax_img4 = fig.add_subplot(gs[0, 4])
    ax_img5 = fig.add_subplot(gs[0, 5])
    axes = [ax_img0, ax_img1, ax_img2, ax_img3, ax_img4, ax_img5]

    ax_plot1 = fig.add_subplot(gs[1, :3])
    ax_plot2 = fig.add_subplot(gs[1, 3:])
    ax_plot3 = fig.add_subplot(gs[2:, :3])
    ax_plot4 = fig.add_subplot(gs[2:, 3:])

    if len(images) != 6:
        raise ValueError("The number of images must be 6.")

    quadrant_colors = {
        4: "magenta",
        1: "cyan",
        3: "green",
        2: "yellow",
    }

    for idx, (timepoint, image) in enumerate(images.items()):
        ax = axes[idx]
        ax.imshow(image, cmap="gray")
        ax.set_title(f"tp={timepoint}")

        for quadrant_num in [2, 3, 4, 1]:
            mask = measuremnt_masks[timepoint] == quadrant_num
            ax.contour(mask, colors=quadrant_colors[quadrant_num], linewidths=0.5)

        ax.axis("off")

    ax_plot1.plot(
        measurements["timepoint"],
        measurements["polarity_magnitude"],
        marker="o",
        markersize=2,
        alpha=0.5,
    )
    ax_plot1.plot(measurements["timepoint"], measurements["polarity_magnitude_smooth"])
    ax_plot1.set_title("Polarity Magnitude")
    ax_plot1.set_xlabel("Timepoint")

    ax_plot2.plot(
        measurements["timepoint"],
        measurements["polarity_angle_diff"],
        marker="o",
        markersize=2,
    )
    ax_plot2.set_title("Polarity Angle Diff")
    ax_plot2.set_xlabel("Timepoint")
    ax_plot2.axhline(0, color="black", linestyle="--", linewidth=0.5)

    if "smooth" in polarity_column:
        pc = polarity_column.replace("_smooth", "")
        pc_smooth = polarity_column
    else:
        pc = polarity_column
        pc_smooth = f"{polarity_column}_smooth"

    ax_plot3.plot(
        measurements["timepoint"],
        measurements[pc],
        marker="o",
        markersize=2,
        alpha=0.5,
    )
    ax_plot3.plot(
        measurements["timepoint"],
        measurements[pc_smooth],
    )
    fill_true(measurements, "polar", ax_plot3, color="red", alpha=0.2)

    ax_plot4.plot(
        measurements["timepoint"],
        measurements[pc],
        marker="o",
        markersize=2,
        alpha=0.5,
    )
    ax_plot4.plot(
        measurements["timepoint"],
        measurements[pc_smooth],
    )
    fill_true(measurements, "polarizing", ax_plot4, color="red", alpha=0.2)
    fill_true(measurements, "stalling", ax_plot4, color="blue", alpha=0.2)

    fig.suptitle(f"{uid} - Quadrant Polarity analysis")

    plt.tight_layout()
    sns.despine()
    fig.savefig(path / f"{uid}__polarity_time_series.pdf", transparent=True)
    # plt.show()


def plot_cell_measurements(
    df: pd.DataFrame, path: str | Path, graph_size: tuple | list = (2, 1)
):
    features_to_plot = []
    no_plot_cols = ["uid", "condition"]
    for col in df.columns:
        if col not in no_plot_cols and df[col].dtype not in [object, str]:
            features_to_plot.append(col)

    nrows = 1
    ncols = len(features_to_plot)

    height, width = graph_size

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height))
    for idx, feature in enumerate(features_to_plot):
        sns.stripplot(
            data=df,
            y=feature,
            x="condition",
            hue="condition",
            alpha=0.4,
            size=3,
            ax=axes[idx],
        )
        sns.pointplot(
            data=df,
            x="condition",
            y=feature,
            hue="condition",
            # dodge=0.4,
            errorbar="sd",  # standard error
            estimator="mean",  # or "median"
            capsize=0.075,
            linestyle="none",
            markersize=10,
            marker="_",
            err_kws=dict(linewidth=0.4, color="black"),
            markeredgewidth=1,
            palette="dark:black",
            zorder=5,
            ax=axes[idx],
        )
    plt.tight_layout()
    sns.despine()
    fig.savefig(path, transparent=True)
