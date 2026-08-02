from pathlib import Path
from cell import CellSeries
import argparse
import pandas as pd
from plotlib import plot_cell_measurements

parser = argparse.ArgumentParser(description="Analyze cell time series data")

root = parser.add_argument(
    "--data_dir",
    type=str,
    default=None,
    help="Path to the directory containing cell data folders",
)
condition_list_str = parser.add_argument(
    "--conditions",
    type=str,
    default="",
    help="Comma-separated list of conditions corresponding to each cell folder",
)


condition_list_str = (
    condition_list_str
    if isinstance(condition_list_str, str)
    else condition_list_str.default
)
condition_list = condition_list_str.split(",")

if isinstance(root, str):
    root = Path(root)
else:
    root = Path(__file__).parent / "data"

if not root.exists():
    raise FileNotFoundError(f"Data directory {root} does not exist.")

cell_folders = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]


cell_measurements = []
for cell_folder in cell_folders:
    condition = None
    for condition_candidate in condition_list:
        if condition_candidate.lower().strip() in cell_folder.name.lower():
            condition = condition_candidate.strip()

    cell_series = CellSeries(path=cell_folder, pixel_size=6.5 / 63, condition=condition)
    cell_series.plot()
    cell_measurements.append(cell_series.get_cell_measurements())


df = pd.DataFrame(cell_measurements)
cell_figure_path = root.parent / "figures"
cell_figure_path.mkdir(exist_ok=True)
plot_cell_measurements(df, cell_figure_path / "cell_measurements.pdf")
