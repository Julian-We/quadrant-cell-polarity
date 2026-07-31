import datetime
import numpy as np
from pathlib import Path
from helper import (
    get_centroid,
    get_polarity,
    get_angled_unitvector,
    generate_quadrant_mask,
    get_ring_mask,
    read_images,
)


class CellSeries:
    def __init__(
        self,
        path: Path | str,
        time_clip: list | tuple | None = None,
        channel_of_interest: int | None = None,
        pixel_szie: float | None = None,
    ):
        """A series of CellTimepoints for a single cell across timepoints in a movie"""
        if isinstance(path, str):
            path = Path(path)


class CellTimepoint:
    def __init__(self, timepoint_id, cell_id, image, mask, pixel_size=1.0):
        """Image and conneced segmentation and features  for a timpoint in a movie"""

        # Context info
        self.cell_id = cell_id
        self.timepoint_id = timepoint_id

        # Image and segmentation
        self.image = image
        self.mask = mask
        self.pixel_size = pixel_size
        self.centroid = (None, None)

        # Frame polarity data
        self.polarity_unit_vector = (None, None)
        self.polarity_magnitude = None

        # Quadrant masks: Q1=front, Q2=counter clockwise side, Q3=clockwise side, Q4=back
        self.quadrant_masks = {"Q1": None, "Q2": None, "Q3": None, "Q4": None}
        self.measuremnt_masks = {
            "Q1": None,
            "Q2": None,
            "Q3": None,
            "Q4": None,
            "Total": None,
        }
        self.measurements = {
            "Q1": None,
            "Q2": None,
            "Q3": None,
            "Q4": None,
            "Total": None,
        }

        self.logs = {}

    def log(self, message):
        """Log a message for this cell timepoint"""
        self.logs[datetime.datetime.now().isoformat("#", "microseconds")] = message

    def setup(self):
        self.centroid = get_centroid(self.mask)
        self.polarity_unit_vector, self.polarity_magnitude = get_polarity(
            self.image, self.mask
        )

    def plot_polarity(self, ax=None):
        """Plot the polarity vector on the image"""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(self.image, cmap="gray")
        ax.contour(self.mask, colors="r", linewidths=0.5)

        if self.polarity_unit_vector[0] is not None:
            start = self.centroid
            end = (
                self.centroid[0]
                + self.polarity_unit_vector[0] * self.polarity_magnitude,
                self.centroid[1]
                + self.polarity_unit_vector[1] * self.polarity_magnitude,
            )
            ax.annotate(
                "",
                xy=end[::-1],
                xytext=start[::-1],
                arrowprops=dict(arrowstyle="->", color="yellow"),
            )

        ax.set_title(f"Cell {self.cell_id} at timepoint {self.timepoint_id}")

    def get_quadrants_masks(self, unit_vector=None):
        if unit_vector is None:
            unit_vector = self.polarity_unit_vector

        quadrant_angles = {
            "Q1": (320, 45),
            "Q2": (320, 230),
            "Q3": (45, 135),
            "Q4": (135, 230),
        }

        for q, (angle_start, angle_end) in quadrant_angles.items():
            v1 = get_angled_unitvector(unit_vector, angle_start)
            v2 = get_angled_unitvector(unit_vector, angle_end)
            self.quadrant_masks[q] = generate_quadrant_mask(
                self.mask.shape, self.centroid, v1, v2
            )

    def plot_measured_quadrants(self, ax=None):
        """Plot the quadrant masks on the image"""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.contour(self.measuremnt_masks["Q1"], colors="cyan")
        ax.contour(self.measuremnt_masks["Q2"], colors="magenta")
        ax.contour(self.measuremnt_masks["Q3"], colors="yellow")
        ax.contour(self.measuremnt_masks["Q4"], colors="green")

    def set_measurement_masks(self, rim_thickness=3.0):
        main_measurement_mask = self.mask.copy()
        if rim_thickness:
            main_measurement_mask = get_ring_mask(
                self.mask, thickness=rim_thickness, pixel_size=self.pixel_size
            )

        self.measuremnt_masks["Total"] = main_measurement_mask

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            self.measuremnt_masks[q] = (
                main_measurement_mask * self.quadrant_masks[q]
            ).astype(bool)

    def measure(self, func):
        if func is None:
            func = np.mean
        for area, mask in self.measuremnt_masks.items():
            self.measurements[area] = func(self.image[mask.astype(bool)])
