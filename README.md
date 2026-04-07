# SMKPLOT - Emissions Visualization Tool

A high-performance tool for visualizing emissions data from SMOKE reports, FF10 (Point, Nonpoint, Fire) sectors, NetCDF gridded files, and standard CSV formats. SMKPLOT supports a modern interactive GUI (Qt-based), legacy Tkinter support, and comprehensive headless batch processing.

## 🚀 Key Features

*   **Modern Interactive GUI**: Native PySide6 interface with high-performance rendering, a clean modern theme, and thread-safe data loading.
*   **Dual Operation Modes**: Interactive exploration mode for analysis and automated batch mode for cluster pipelines.
*   **Broad Format Support**:
    *   **SMOKE Reports**: Auto-detects pollutants and units.
    *   **FF10 Sectors**: Direct support for standard EPA emission formats.
    *   **NetCDF (IOAPI)**: Automatic header parsing (XORIG, XCELL, etc.) for gridded domain generation.
*   **Pollutant Aggregation (Derived Columns)**:
    *   Define new output columns using inline arithmetic formulas in the `pollutant` field:
        `NOX_TOTAL = S-NO + S-NO2, RATIO = VOC / NOX`
    *   Supported operators (space-separated): `+`, `-`, `x` (multiply), `/` (divide).
    *   Scalar literals are also supported: `PM25_ADJ = PM2_5 x 1.1`.
    *   Works in both batch and GUI modes; derived columns are treated as regular pollutants for plotting and export.
*   **Per-Pollutant Colorbar Bins (`bins_map`)**:
    *   Assign different colorbar bin edges to each pollutant in a single YAML run via the `bins_map` key.
    *   `bins_map` takes precedence over the global `bins` argument.
    *   Only available through YAML/JSON configuration (not via CLI `--bins`).
*   **Interactive Analysis**:
    *   **3D Vertical Profiling & Time Series Extraction (Cursor Plot)**: Seamlessly click on any grid cell to extract multi-dimensional cross-sections. Extract temporal patterns (`by-TSTEP`) or vertical concentrations across layers (`by-LAY`).
    *   **Cell Inspection**: Point-and-click logic to identify grid/county indices and emissions values.
    *   **Time Series Animation**: Integrated animation controls for NetCDF files to visualize temporal patterns.
    *   **Domain Stats**: Real-time calculation of spatial statistics (Max, Min, Avg, Total) for current vertical/temporal slices.
*   **Administrative Robustness**:
    *   **County Data**: Automatic downloading of verified US Census Cartographic Boundary files if local versions are missing.
*   **Advanced Spatial Operations**:
    *   **Filtering**: Spatial filtering using auxiliary shapefiles (modes: `intersect`, `clipped`, `within`) with robust centroid/representative-point logic.
    *   **Overlay**: Support for secondary visual overlays (roads, water, custom regions) on any plot.
*   **Data & Reproducibility**:
    *   **Configuration Snapshots**: Export current session settings to YAML/JSON for exact reproducibility.
    *   **Data Export**: Direct export of processed, filtered, or aggregated data to standard CSV files.
    *   **Metadata Explorer**: Comprehensive view of NetCDF global attributes and variable-specific metadata.
*   **Performance & Clusters**:
    *   **Fast QuadMesh Rendering**: Instantaneous visualization of national-scale gridded datasets using optimized vertex-buffer logic.
    *   **Parallel Processing**: Multi-core support for batch plotting tasks.
    *   **Headless Support**: Automatic offscreen rendering on systems without X-server/Wayland.
    *   **Configuration Snapshots**: Reproducible runs via JSON or YAML configuration files.

## 📦 Installation

### Prerequisites
*   Python 3.9+
*   (Recommended) `pip`, `venv`, and `libegl1` (for Qt GUI).

### Quick Start
Use the specialized installation script to set up an isolated environment:
```bash
chmod +x install.sh
./install.sh
```
The script will install dependencies, configure the `smkplot.py` shebang to use the local venv, and verify GUI capabilities.

### Updating SMKPLOT
If the repository has been updated on GitHub, you can safely pull the latest changes and refresh your environment using the provided updater script:
```bash
./update.sh
```
*(Note: This will discard any local modifications to the tracked scripts and automatically re-run `install.sh` to ensure all Python dependencies are current).*

## 📂 Usage

### 🖥️ Interactive GUI
```bash
./smkplot.py
```
Options in the GUI allow for selection of pollutants (via multi-column grid), setting colormaps (vibrant, sequential, divergent), and configuring log/linear scales.

### ⚙️ Batch Processing
Ideal for large datasets or scheduled runs.

**Example: Plotting NOX and VOC from a CSV file**
```bash
./smkplot.py --run-mode batch \
  -f /path/to/emis.csv \
  --pollutant "NOX,VOC" \
  --outdir ./maps
```

**Example: Using a YAML Configuration**
```bash
# General use
./smkplot.py -f my_config.yaml

# Run the 3D NetCDF example
./smkplot.py -f example_inputs/config_3d_netcdf.yaml
```

**Example: Gridded NetCDF with Time Aggregation**
```bash
./smkplot.py --run-mode batch \
  -f CCTM_emissions.ncf \
  --pltyp grid \
  --ncf-tdim sum \
  --ncf-zdim 0 \
  --bins "0,1,5,10,50,100"
```
*Note: NetCDF dimensions (`ncf-tdim`, `ncf-zdim`) strictly follow a **0-based integer index** convention (e.g., `0` = LAY 0 / TSTEP 0).*

## 📊 Example Dataset
The `example_inputs/` directory contains pre-configured datasets for testing all tool capabilities:
*   `example_3d.nc`: A 3D gridded NetCDF file with 5 time steps and 10 vertical layers.
*   `example_gridded.nc`: A standard 2D gridded NetCDF file.
*   `example_ff10.csv`: A standard FF10 Nonpoint sector file.
*   `example_ff10_point.csv`: An FF10 Point source file with lat/lon coordinates.
*   `example_smkreport.csv`: A standard SMOKE report output.
*   `example_griddesc.txt`: Example grid definitions for the Lambert Conformal projection.

Pre-configured run files:

| Config file | Feature demonstrated |
|---|---|
| `config_ff10.yaml` | FF10 nonpoint, county choropleth, multi-pollutant batch |
| `config_ff10_point_county.yaml` | FF10 point, county aggregation |
| `config_ff10_point_grid.yaml` | FF10 point, grid mapping |
| `config_smkreport.yaml` | SMOKE report, county choropleth |
| `config_netcdf.yaml` | 2D NetCDF, gridded plot |
| `config_3d_netcdf.yaml` | 3D NetCDF, layer/time selection |
| `config_bins_map.yaml` | **Per-pollutant colorbar bins** (`bins_map`) |
| `config_pollutant_agg.yaml` | **Derived columns via arithmetic formulas** (`col_formulas`) |


## 🛠️ Advanced Logic

### Inline Point Source Processing
When processing **Inline (1D)** point sources, SMKPLOT requires a matching `STACK_GROUPS` file. The tool includes intelligence to:
1.  Verify dimensions between emissions and stack headers.
2.  Auto-locate date-matched `STACK_GROUPS` files within the same directory if not explicitly provided.
3.  Project lat/lon point coordinates into the target grid CRS with sub-cell precision.


### Pollutant Aggregation — Derived Columns
The `pollutant` field (YAML, JSON, or `--pollutant` CLI) accepts inline arithmetic formulas to create derived output columns on-the-fly.

**Syntax:** `DERIVED_NAME = COL1 <op> COL2 <op> ...`  
Operators must be **surrounded by spaces**: `+`, `-`, `x` (multiply), `/` (divide).  
Plain column names (no `=`) pass through unchanged and are plotted as-is.

```yaml
# YAML example — mix plain columns and formulas in one run
arguments:
  pollutant: "NOX, VOC, NOX_PLUS_VOC = NOX + VOC, RATIO = NOX / VOC"
```

```bash
# CLI example (bash quoting required for spaces)
./smkplot.py --pollutant "NOX, NOX_PLUS_VOC = NOX + VOC" -f emissions.csv
```

| Expression | Meaning |
|---|---|
| `NOX` | Plot existing `NOX` column unchanged |
| `TOTAL = NOX + VOC` | Sum `NOX` and `VOC` into new column `TOTAL` |
| `ADJ = PM2_5 x 1.1` | Multiply `PM2_5` by scalar `1.1` |
| `RATIO = VOC / NOX` | Divide `VOC` by `NOX` cell-by-cell |
| `NET = NO + NO2 - NO3` | Chain multiple operators |

See `example_inputs/config_pollutant_agg.yaml` for a ready-to-run example.


### Per-Pollutant Colorbar Bins (`bins_map`)
When plotting multiple pollutants in a single batch run, different species often span very different value ranges. The `bins_map` key in YAML/JSON lets you assign independent colorbar bin edges per pollutant.

**Precedence:** `bins_map[pollutant]` → `bins` (global) → auto-scale.

```yaml
arguments:
  pollutant: NOX, VOC, PM2_5
  bins_map:
    NOX:   [0, 5, 10, 25, 50, 100]
    VOC:   [0, 1, 3, 7, 15, 30]
    PM2_5: [0, 2, 5, 10, 20, 35]
```

- Values can be a YAML list (as above) or a space/comma-separated string: `"0 5 10 50"`.
- Pollutants not listed in `bins_map` fall back to the global `bins` value or auto-scale.
- `bins_map` is only honoured from YAML/JSON config files; there is no equivalent CLI flag.

See `example_inputs/config_bins_map.yaml` for a ready-to-run example.


### Spatial Filter Operations
*   `intersect`: Keeps any feature that overlaps with the filter geometry.
*   `within`: Keeps features whose calculated representative point or centroid lies inside the filter.
*   `clipped`: Geometrically intersects the grid/county boundaries with the filter, resulting in a true spatial "cookie-cut."

## ❓ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Qt Failed to Load** | Ensure `PySide6` is in your venv. Run `./install.sh` to refresh. |
| **Memory Errors** | Use fewer `--workers` in batch mode if plotting very high-resolution grids. |
| **Map is Empty** | Check if `--delim` is correct or if tribal mapping filtered the data. |
| **Hanging on Exit** | Use the Native Qt interface (default) which includes hardened shutdown routines. |

---
**Version**: 2.0
**Author**: tranhuy@email.unc.edu
