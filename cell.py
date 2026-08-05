import datetime
from typing import Type
import numpy as np
from pathlib import Path
from smoothing import savgol_smooth
import helper as hlp
import plotlib
import calculator as calc
import pandas as pd


class CellSeries:
    def __init__(
        self,
        path: Path | str,
        condition: str = "cremig",
        time_clip: list | tuple | None = None,
        channel_of_interest: int | None = None,
        pixel_size: float = 1.0,
        outer_ring_thickness: float | None = None,
        quadrant_method: str = "max",  # 'max' or 'adaptive'
        normalization_area: str = "Total",  # 'Total', Center' or Q2-Q4
        polarity_column: str = "Q1_norm_smooth",
        output_dir: Path | str | None = None,
    ):
        """A series of CellTimepoints for a single cell across timepoints in a movie"""

        if isinstance(path, str):
            path = Path(path)

        self.uid = None
        self.condition = condition
        self.path = path
        self.output_dir = (
            Path(output_dir) if isinstance(output_dir, str) else output_dir
        )

        self.image = None
        self.mask = None

        self.timepoints = []
        self.measurements = []
        self.df = pd.DataFrame()
        self.cell_measurements = {
            "uid": self.uid,
            "condition": self.condition,
        }

        self.max_polarity_vector = (None, None)

        self.quadrant_method = quadrant_method
        self.normalization_area = normalization_area
        self.polarity_column = polarity_column

        mask_path = [p for p in path.glob("*.tif") if "mask" in p.name.lower()][0]
        image_path = [p for p in path.glob("*.tif") if "mask" not in p.name.lower()][0]

        raw_image = hlp.read_image(image_path)
        self.uid = hlp.get_unique_id(raw_image)

        if raw_image.ndim == 3 and time_clip is None:
            self.image = raw_image
        elif raw_image.ndim == 3 and time_clip is not None:
            self.image = hlp.time_clipper(raw_image, *time_clip)
        elif (
            raw_image.ndim == 4
            and channel_of_interest is not None
            and time_clip is None
        ):
            self.image = raw_image[:, channel_of_interest, :, :]
        elif (
            raw_image.ndim == 4
            and channel_of_interest is not None
            and time_clip is not None
        ):
            self.image = hlp.time_clipper(
                raw_image[:, channel_of_interest, :, :], *time_clip
            )
        else:
            raise ValueError(
                "Unexceted image dimenstionsm or neither time_clip nor channel_of_interest is specified for a 4D image."
            )

        self.mask = hlp.read_image(mask_path)
        if self.mask.ndim == 3 and time_clip is not None:
            self.mask = hlp.time_clipper(self.mask, *time_clip)
        elif self.mask.ndim == 3 and time_clip is None:
            pass
        elif self.mask.ndim == 4 and channel_of_interest is not None:
            self.mask = self.mask[:, channel_of_interest, :, :]
        elif (
            self.mask.ndim == 4
            and channel_of_interest is not None
            and time_clip is not None
        ):
            self.mask = hlp.time_clipper(
                self.mask[:, channel_of_interest, :, :], *time_clip
            )
        else:
            raise ValueError(
                "Unexceted mask dimenstionsm or neither time_clip nor channel_of_interest is specified for a 4D mask."
            )

        for idx, (image_slice, mask_slice) in enumerate(zip(self.image, self.mask)):
            if np.sum(mask_slice) == 0:
                raise ValueError(
                    f"Mask slice has no non-zero pixels. Check the mask file: {mask_path}"
                )

            self.timepoints.append(
                CellTimepoint(
                    timepoint_id=idx,
                    cell_id=self.uid,
                    condition=self.condition,
                    image=image_slice,
                    mask=mask_slice.astype(bool),
                    pixel_size=pixel_size,
                    sigma=5.0,
                )
            )

        self.quadrant_processing(rim_thickness=outer_ring_thickness)
        self.df = pd.DataFrame(self.measurements)
        self.polarity_analysis()

        if not self.check_mask_consistency():
            print(
                f"WARNING: Masks for cell {self.uid} are not consistent across timepoints. Check the mask file: {mask_path}"
            )

    def polarity_analysis(
        self,
        time_resolution: float = 1.0,
        polarity_column: str = "Q1_norm_smooth",
        polarity_threshold: float = 0.275,
        min_phase_length: int = 4,
        stalling_sigma: float = 0.5,
        savgol_polyorder: int = 4,
        savgol_window_length: int | None = None,
    ):

        self.df["time"] = self.df["timepoint"] * time_resolution

        # Smooth and diff Q1_norm
        self.df["Q1_norm_smooth"] = savgol_smooth(
            self.df["Q1_norm"],
            window_length=savgol_window_length,
            polyorder=savgol_polyorder,
        )
        self.df["Q1_norm_smooth_diff"] = self.df["Q1_norm_smooth"].diff()

        # Smooth and diff polarity magnitude
        self.df["polarity_magnitude_smooth"] = savgol_smooth(
            self.df["polarity_magnitude"],
            window_length=savgol_window_length,
            polyorder=savgol_polyorder,
        )
        self.df["polarity_magnitude_smooth_diff"] = self.df[
            "polarity_magnitude_smooth"
        ].diff()

        # Get polarizing phases and measurements
        calc.get_polarization_phases(self.df, polarity_column, min_phase_length)
        phases_polarizing = calc.true_phases(self.df, "polarizing", polarity_column)
        self.cell_measurements.update(
            calc.quantify_phases(phases_polarizing, "polarizing")
        )

        # Get stalling phases and measurements
        calc.get_stalling_phases(
            self.df, polarity_column, min_phase_length, stalling_sigma
        )
        phases_stalling = calc.true_phases(self.df, "stalling", polarity_column)
        self.cell_measurements.update(calc.quantify_phases(phases_stalling, "stalling"))

        # Get local extrema and above threshold phases and measurements
        calc.get_local_extrems(self.df, polarity_column)
        calc.get_above_polar_threshold(
            self.df, polarity_column, min_phase_length, polarity_threshold
        )

        ramp_on_phases = calc.get_time_since_local_min_to_polar(
            self.df, polarity_column
        )
        self.cell_measurements.update(
            {
                "min_to_polar_ramp_on_mean": ramp_on_phases["n_timepoints"].mean(),
            }
        )

        polarity_angle_diff_apolar = self.df[self.df["polar"] == False][
            "polarity_angle_diff"
        ]
        self.cell_measurements.update(
            {
                "apolar_polarity_angle_diff_mean": polarity_angle_diff_apolar.mean(),
                "apolar_polarity_angle_diff_std": polarity_angle_diff_apolar.std(),
            }
        )

    def get_max_polarity_timepoint(self):
        """Return the timepoint with the maximum polarity magnitude"""
        max_polarity = -np.inf
        max_timepoint = None
        for timepoint in self.timepoints:
            if (
                timepoint.polarity_magnitude is not None
                and timepoint.polarity_magnitude > max_polarity
            ):
                max_polarity = timepoint.polarity_magnitude
                max_timepoint = timepoint

        if max_timepoint is None:
            print(
                "WARNING: No timepoint has a valid polarity magnitude. Using the first timepoint as default."
            )
            self.max_polarity_vector = self.timepoints[0].polarity_unit_vector
        else:
            self.max_polarity_vector = max_timepoint.polarity_unit_vector

    def quadrant_processing(
        self,
        rim_thickness=None,
        sigma: float | None = None,
        measurement_function: str = "sum",
    ):
        """Process all timepoints to get quadrant masks and measurements"""
        self.get_max_polarity_timepoint()
        for idx, timepoint in enumerate(self.timepoints):
            if isinstance(measurement_function, str):
                if measurement_function.lower() == "sum":
                    func = np.sum
                elif measurement_function.lower() == "mean":
                    func = np.mean
                elif measurement_function.lower() == "std":
                    func = np.std
                else:
                    print(
                        f"WARNING measurement_function <{measurement_function}> is not known, using np.sum instead. please supply function"
                    )
                    func = np.sum
            elif isinstance(measurement_function, type(np.sum)):
                func = measurement_function
            else:
                raise TypeError(
                    f"measurement_function of type {type(measurement_function)} is not supported"
                )
            if self.quadrant_method == "max":
                timepoint.set_measurement_masks(
                    rim_thickness=rim_thickness, unit_vector=self.max_polarity_vector
                )
            elif self.quadrant_method == "adaptive":
                timepoint.set_measurement_masks(rim_thickness=rim_thickness)
            else:
                raise ValueError(
                    f"{self.quadrant_method} is not a valid quadrant method. Use 'max' or 'adaptive'."
                )

            timepoint.measure(func=func, sigma=sigma)

            # Get differential measurements
            if not idx == 0:
                timepoint.measurements["polarity_angle_diff"] = (
                    hlp.get_angle_between_vectors(
                        timepoint.polarity_unit_vector,
                        self.timepoints[idx - 1].polarity_unit_vector,
                    )
                )
                timepoint.measurements["Q1_norm_diff"] = (
                    timepoint.measurements["Q1_norm"]
                    - self.timepoints[idx - 1].measurements["Q1_norm"]
                )

            self.measurements.append(timepoint.measurements)

    def check_mask_consistency(self):
        return hlp.mask_checker([tp.mask for tp in self.timepoints])

    def get_data(self, as_dict=False, path: Path | str | None = None):
        """Get data of all timpoints as list of dictionarties or as pandas dataframe"""
        if isinstance(path, str):
            path = Path(path)
        if as_dict:
            return self.measurements
        else:
            return hlp.export_data(self.measurements, output_path=path)

    def get_cell_measurements(self):
        df = self.df
        self.cell_measurements.update(
            {
                "uid": self.uid,
                "condition": self.condition,
                "polarity_magnitude_mean": df["polarity_magnitude"].mean(),
                "polarity_magnitude_std": df["polarity_magnitude"].std(),
                "polarity_magnitude_max": df["polarity_magnitude"].max(),
                "polarity_magnitude_min": df["polarity_magnitude"].min(),
                "polarity_angle_diff_mean": df["polarity_angle_diff"].mean(),
                "polarity_angle_diff_std": df["polarity_angle_diff"].std(),
                "polarity_angle_diff_max": df["polarity_angle_diff"].max(),
                "polarity_angle_diff_min": df["polarity_angle_diff"].min(),
                "Q1_norm_mean": df["Q1_norm"].mean(),
                "Q1_norm_std": df["Q1_norm"].std(),
                "Q1_norm_max": df["Q1_norm"].max(),
                "Q1_norm_min": df["Q1_norm"].min(),
            }
        )
        return self.cell_measurements

    def plot(self, **kwargs):
        # Equispaced timepoints for plotting including first and last timepoint
        plot_timepoints = np.linspace(0, len(self.timepoints) - 1, 6, dtype=int)
        plot_images = {}
        plot_masks = {}
        for plot_tp in plot_timepoints:
            plot_images[plot_tp] = self.timepoints[plot_tp].image
            plot_masks[plot_tp] = self.timepoints[plot_tp].get_quadrant_label_image()

        if self.output_dir is not None:
            figure_path = self.output_dir / "quadrant_analysis_plots"
        else:
            figure_path = self.path / "figures"
        figure_path.mkdir(exist_ok=True, parents=True)
        plotlib.plot_time_series(
            figure_path,
            plot_images,
            self.df,
            plot_masks,
            uid=str(self.uid),
            polarity_column=self.polarity_column,
            **kwargs,
        )


class CellTimepoint:
    def __init__(
        self,
        timepoint_id,
        cell_id,
        image,
        mask,
        pixel_size=1.0,
        condition: str = "cremig",
        center_savety_distance=2,
        **kwargs,
    ):
        """Image and conneced segmentation and features  for a timpoint in a movie
        timepoint_id: timepoint id (frame) of the timpoint
        cell_id: ID of cell (blinded number)
        image: Intensity image for polarity and measurement (actin)
        mask: binary mask that segments the cell
        pixel_size: size of a pixel in micrometer
        condition: Assignment of a condition (control vs. treatment)
        center_savety_distance: for center normalization thickness of the subtracted ring
        """

        # Context info
        self.cell_id = cell_id
        self.timepoint_id = timepoint_id
        self.condition = condition

        # Image and segmentation
        self.image = image
        self.mask = mask
        self.pixel_size = pixel_size
        self.centroid = (None, None)

        # Frame polarity data
        self.polarity_unit_vector = (None, None)
        self.polarity_magnitude = None

        self.center_savety_distance = center_savety_distance

        # Quadrant masks: Q1=front, Q2=counter clockwise side, Q3=clockwise side, Q4=back
        self.quadrant_masks = {"Q1": None, "Q2": None, "Q3": None, "Q4": None}
        self.measurement_masks = {
            "Q1": None,
            "Q2": None,
            "Q3": None,
            "Q4": None,
            "Total": None,
        }
        self.measurements = {
            "cell_id": self.cell_id,
            "timepoint": self.timepoint_id,
            "condition": self.condition,
        }

        self.logs = {}

        self.setup(**kwargs)

    def log(self, message):
        """Log a message for this cell timepoint"""
        self.logs[datetime.datetime.now().isoformat("#", "microseconds")] = message

    def setup(self, sigma: None | float = None, **kwargs):
        if sigma is None:
            intensity_image = self.image.copy()
        else:
            intensity_image = hlp.apply_gaussian_filter(self.image, sigma=sigma)
        self.centroid = hlp.get_centroid(self.mask)
        self.polarity_unit_vector, self.polarity_magnitude = hlp.get_polarity(
            self.image, self.mask
        )
        self.measurements["polarity_magnitude"] = self.polarity_magnitude
        self.measurements["polarity_angle_diff"] = np.nan

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
            v1 = hlp.get_angled_unitvector(unit_vector, angle_start)
            v2 = hlp.get_angled_unitvector(unit_vector, angle_end)
            self.quadrant_masks[q] = hlp.generate_quadrant_mask(
                self.mask.shape, self.centroid, v1, v2
            )

    def get_quadrant_label_image(self):
        # Generate a labnel image for the quadrants
        canvas = np.zeros(self.mask.shape, dtype=np.uint8)
        for q_num in [1, 2, 3, 4]:
            q_mask = self.measurement_masks[f"Q{q_num}"]
            canvas[q_mask > 0] = q_num
        return canvas

    def plot_measured_quadrants(self, ax=None):
        """Plot the quadrant masks on the image"""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.contour(self.measurement_masks["Q1"], colors="cyan")
        ax.contour(self.measurement_masks["Q2"], colors="magenta")
        ax.contour(self.measurement_masks["Q3"], colors="yellow")
        ax.contour(self.measurement_masks["Q4"], colors="green")

    def set_measurement_masks(self, rim_thickness=3.0, unit_vector=None):
        self.get_quadrants_masks(unit_vector=unit_vector)
        main_measurement_mask = self.mask.copy()
        if rim_thickness:
            main_measurement_mask = hlp.get_ring_mask(
                self.mask, thickness=rim_thickness, pixel_size=self.pixel_size
            )

        self.measurement_masks["Total"] = main_measurement_mask
        self.measurement_masks["Center"] = hlp.get_center_mask(
            self.mask, self.center_savety_distance, pixel_size=self.pixel_size
        )

        for q in ["Q1", "Q2", "Q3", "Q4"]:
            self.measurement_masks[q] = (
                main_measurement_mask * self.quadrant_masks[q]
            ).astype(bool)

    def measure(self, func, sigma: float | None = None, normalization_area="Total"):
        if func is None:
            func = np.sum

        if sigma is not None:
            image = hlp.apply_gaussian_filter(self.image, sigma=sigma)
        else:
            image = self.image.copy()
        func_name = str(func.__name__) if hasattr(func, "__name__") else str(func)
        for area, mask in self.measurement_masks.items():
            self.measurements[f"{area}_{func_name}"] = func(image[mask.astype(bool)])
        if normalization_area not in self.measurement_masks.keys():
            raise KeyError(
                f"{normalization_area} not found in measurement areas. Select from {self.measurement_masks.keys()}"
            )
        self.measurements["Q1_norm"] = (
            self.measurements[f"Q1_{func_name}"]
            / self.measurements[f"{normalization_area}_{func_name}"]
        )
