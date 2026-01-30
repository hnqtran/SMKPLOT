# SMKPLOT - Emissions Visualization Tool

A comprehensive tool for visualizing emissions data from SMOKE reports, FF10 files, NetCDF, and CSV formats. Supports both a modern interactive GUI (Qt-based) and automated batch processing modes.

## Features

*   **Modern Interactive GUI**: High-performance Qt-based interface (via PySide6) with a sleek, modern light theme.
*   **Legacy Support**: Fallback support for Tkinter-based GUI on legacy systems.
*   **Dual Mode Operation**: Interactive exploration or headless batch mode for cluster automation.
*   **Multiple Data Formats**: SMOKE reports, FF10 point/nonpoint, NetCDF (including inline sources), CSV/TXT.
*   **Large Dataset Handling**: Support for multi-column pollutant selection (grid view) for files with hundreds of chemical species.
*   **Spatial Filtering**: Filter data by geographic regions using overlay shapefiles with robust centroid/representative-point detection.
*   **Flexible Plotting**: County-based (FIPS) or grid-based visualization.
*   **Automatic Projections**: Smart handling of WGS84, Lambert Conformal Conic, and IOAPI NetCDF coordinate systems.
*   **Parallel Processing**: Multi-core batch processing for large temporal or spatial datasets.

## Installation

### Prerequisites
*   Python 3.9 or higher.
*   X11 or Wayland display (for GUI mode).

### Quick Setup

Use the provided installation script to create a virtual environment and install all dependencies (including the Qt runtime):
```bash
chmod +x install.sh
./install.sh
```

This ensures the tool uses its own isolated environment and does not depend on system-wide libraries like `python3-tk`.

### Manual Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### GUI Mode (Preferred)

Launch the modern Qt interface:
```bash
./smkplot.py --gui-lib qt
# OR simply (defaults to Qt if installed)
./smkplot.py
```

The new GUI features:
*   **Multi-column Pollutant Selection**: Efficiently browse and select from long lists of pollutants.
*   **Real-time Status Updates**: Thread-safe status bar with progress indicators for data loading and rendering.
*   **Harden Shutdown**: Immediate process termination on window close, preventing hanging background tasks.

### Batch Mode

Generate plots automatically without a display:

```bash
./smkplot.py --run-mode batch \
  -f emissions.csv \
  --county-shapefile counties.gpkg \
  --pollutant "NOX,VOC" \
  --outdir ./output_maps
```

## NetCDF & Grid Plotting

SMKPLOT excels at handling gridded data:
*   **IOAPI Support**: Direct reading of IOAPI-formatted NetCDF files.
*   **Auto-Grid Detection**: Automatically extracts domain parameters (NCOLS, NROWS, XORIG, etc.) from NetCDF headers.
*   **Dimensional Operations**: Easily aggregate across time (`avg`, `sum`, `max`) or layers directly from the command line or GUI.

```bash
./smkplot.py --run-mode batch \
  -f CCTM_emissions.ncf \
  --pltyp grid \
  --ncf-tdim sum \
  --ncf-zdim 0
```

## Experimental: Headless Operation
If you are running on a cluster without an X-server, always use `--run-mode batch`. The tool will automatically switch to the `Agg` backend to avoid display errors.

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **GUI won't quit** | Ensure you are using the latest `gui_qt.py` which uses `os._exit` for hardening. |
| **"Could not determine file format"** | This is INFO only; it means the file lacks a header but will be parsed as a standard SMKREPORT. |
| **Pollutant list is empty** | Verify the `--delim` setting matches your file structure. |
| **Qt launch failed** | Ensure `PySide6` is installed in the local `.venv`. Use `./install.sh` to fix. |

---
**Version**: 1.0 (Modernized)  
**Author**: tranhuy@email.unc.edu
