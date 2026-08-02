from pathlib import Path
from cell import CellSeries
import argparse
import pandas as pd
from plotlib import plot_cell_measurements
from tkinter import Tk
from tkinter.filedialog import askdirectory

parser = argparse.ArgumentParser(description="Analyze cell time series data")
parser.add_argument(
    "--data_dir",
    type=str,
    default=None,
    help="Path to the directory containing cell data folders",
)
parser.add_argument(
    "--conditions",
    type=str,
    default="",
    help="Comma-separated list of conditions corresponding to each cell folder",
)
args = parser.parse_args()

condition_list = [c.strip() for c in args.conditions.split(",") if c.strip()]
print(f"Conditions: {condition_list}")

if args.data_dir is not None:
    root = Path(args.data_dir)
else:
    tk_root = Tk()
    tk_root.withdraw()
    selected_dir = askdirectory(
        title="Select the data directory containing cell folders",
        initialdir=str(Path.cwd()),
    )
    tk_root.destroy()
    file_data_dir = Path(__file__).parent / "data"
    if not selected_dir and file_data_dir.exists():
        print(
            "WARNING: No data directory selected. Using the 'data' directory in the script folder."
        )
        selected_dir = str(file_data_dir)
    elif not selected_dir and not file_data_dir.exists():
        raise FileNotFoundError(
            "No data directory selected. And data dirctory does not exist in the script folder."
        )
    root = Path(selected_dir)

output_dir = root.parent / "output"
output_dir.mkdir(exist_ok=True)

if not root.exists():
    raise FileNotFoundError(f"Data directory {root} does not exist.")

cell_folders = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]


cell_measurements = []
for cell_folder in cell_folders:
    condition = None

    for condition_candidate in condition_list:
        if condition_candidate.lower().strip() in cell_folder.name.lower():
            condition = condition_candidate.strip()

    if condition is None and condition_list == []:
        condition = "cremig"
    elif condition is None and condition_list != []:
        print(f"Warning: No condition found for folder {cell_folder.name}")

    cell_series = CellSeries(
        path=cell_folder,
        pixel_size=6.5 / 63,
        condition=condition,
        output_dir=output_dir,
    )
    cell_series.plot()
    cell_measurements.append(cell_series.get_cell_measurements())


df = pd.DataFrame(cell_measurements)
df.to_csv(output_dir / "cell_measurements.csv", index=False)
plot_cell_measurements(df, output_dir / "cell_measurements.pdf", graph_size=(2, 1.6))
