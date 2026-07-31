from pathlib import Path
import tifffile as tiff
from cell import CellTimepoint
import matplotlib.pyplot as plt


root = Path(__file__).parent / "data"

sample_folder_name = "cell1"


CellTimepoint_1 = CellTimepoint(
    cell_id=1,
    timepoint_id=1,
    image=tiff.imread(next((root / sample_folder_name / "image").glob("*.tif"))),
    mask=tiff.imread(root / sample_folder_name / "masks" / "cell.tif"),
    pixel_size=0.072,
)
CellTimepoint_1.setup()


fig, ax = plt.subplots()
CellTimepoint_1.plot_polarity(ax=ax)
CellTimepoint_1.log("Polarity vector plotted.")
CellTimepoint_1.get_quadrants_masks()
CellTimepoint_1.set_measurement_masks(rim_thickness=None)
CellTimepoint_1.plot_measured_quadrants(ax=ax)
plt.show()

