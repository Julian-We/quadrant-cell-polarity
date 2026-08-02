from pathlib import Path
import tifffile as tiff
from cell import CellTimepoint, CellSeries
import matplotlib.pyplot as plt
import seaborn as sns


root = Path(__file__).parent / "data"

sample_folder_name = "cell1"

cell_series = CellSeries(path=root / sample_folder_name, pixel_size=6.5 / 63)

cell_series.plot()


df = cell_series.get_data()


# fig_ts, ax_ts = plt.subplots(2, 1)
# ax_ts[0].plot(df["timepoint"], df["polarity_magnitude"], marker="o")
# ax_ts[0].set_xlabel("Timepoint")
# ax_ts[0].set_ylabel("Polarity Magnitude")
# ax_ts[0].set_title("Polarity Magnitude over Time")
# ax_ts[1].plot(df["timepoint"], df["Q1_norm"], marker="o", color="orange")
# ax_ts[1].set_xlabel("Timepoint")
# ax_ts[1].set_ylabel("Normalized Q1 Intensity")
# ax_ts[1].set_title("Normalized Q1 Intensity over Time")
# plt.tight_layout()
# sns.despine()
# fig_ts.savefig(root / sample_folder_name / "polarity_time_series.pdf")

# num_timepoints = len(cell_series.timepoints)
# fig, axs = plt.subplots(1, num_timepoints, figsize=(5 * num_timepoints, 5))
# for i, timepoint in enumerate(cell_series.timepoints):
#     # axs[i].imshow(timepoint.image, cmap="gray")
#     # axs[i].contour(timepoint.mask, colors="r", linewidths=0.5)
#     timepoint.set_measurement_masks(rim_thickness=None)
#     timepoint.plot_polarity(ax=axs[i])
#     timepoint.plot_measured_quadrants(ax=axs[i])
#
#     axs[i].axis("off")
#     axs[i].set_title(i + 1)
#
# plt.tight_layout()
# fig.savefig(root / sample_folder_name / "polarity_quadrants.pdf")
