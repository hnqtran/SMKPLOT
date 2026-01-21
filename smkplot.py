#!/proj/ie/proj/SMOKE/htran/Emission_Modeling_Platform/utils/smkplot/.venv/bin/python
"""
Main entry point for the SMKPLOT application.

This script serves as the launcher for both the Graphical User Interface (GUI)
and the headless Batch mode. It handles:
- Command-line argument parsing.
- Configuration loading (JSON/YAML).
- Environment setup (PROJ_DATA, PROJ_LIB).
- Dispatching execution to `gui.py` or `batch.py` based on arguments and environment.
"""

import os
import sys

# Set PROJ data paths to fix pyogrio/pyproj issues in virtual environment

try:
    import pyproj
    from pathlib import Path
    _proj_share = Path(pyproj.__file__).resolve().parent / 'proj_dir' / 'share' / 'proj'
    if _proj_share.is_dir():
        os.environ['PROJ_LIB'] = os.environ['PROJ_DATA'] = str(_proj_share)
except Exception:
    pass

import argparse
import datetime
import json
import yaml
import logging
import re
import matplotlib
from typing import List, Set

from utils import USING_TK, tk, _prune_incompatible_bundled_libs, _import_numpy_with_diagnostics

SMKPLOT_VERSION = "2.1"

_POLLUTANT_SPLIT_RE = re.compile(r'[\s,]+')


def _normalize_list_arg(primary, secondary=None) -> List[str]:
    """
    Normalize argument values (strings, lists, comma-separated) into a distinct clean list.
    Handles splitting strings by commas/spaces and flattening lists.
    """
    raw_vals = []

    def _collect(val):
        if val is None:
            return
        if isinstance(val, (list, tuple, set)):
            raw_vals.extend(val)
        else:
            raw_vals.append(val)

    _collect(primary)
    _collect(secondary)

    normalized = []
    seen = set()

    for item in raw_vals:
        if item is None:
            continue
        # Check string first
        if isinstance(item, str):
            # Split by commas/spaces
            parts = _POLLUTANT_SPLIT_RE.split(item.strip()) if item.strip() else []
        elif isinstance(item, (list, tuple, set)):
            # Flatten one level if encountered (just in case)
            parts = [str(p) for p in item]
        else:
            # Fallback for numbers etc
            parts = [str(item)]

        for p in parts:
            clean = p.strip()
            if clean and clean not in seen:
                seen.add(clean)
                normalized.append(clean)

    return normalized


def _load_config_dict(filepath):
    """
    Load JSON or YAML configuration file. 
    Returns (payload_dict, abs_path).
    Raises Exception on failure.
    """
    abs_path = os.path.abspath(filepath)
    lower = abs_path.lower()
    
    with open(abs_path, 'r', encoding='utf-8') as f:
        if lower.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f), abs_path
        elif lower.endswith('.json'):
            return json.load(f), abs_path
            
    raise ValueError("File must end with .json, .yaml, or .yml")

_prune_incompatible_bundled_libs()
_import_numpy_with_diagnostics()


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "SMKPLOT emissions choropleth: run as GUI (default when Tk/display exists) "
            "or as batch/headless to write PNG maps."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples\n"
            "  # 1) Launch GUI (auto-detects Tk/display)\n"
            "  python3 smkplot.py --filepath report.csv\n\n"
            "  # 2) GUI with a specific counties layer and delimiter\n"
            "  python3 smkplot.py --filepath report.txt --delim tab --county-shapefile counties.zip\n\n"
                "  # 3) Batch: selected pollutants (comma separated)\n"
                "  python3 smkplot.py --no-gui \\\n"
                "                   --filepath report.csv --county-shapefile counties.zip \\\n"
                "                   --pollutant NOX,VOC,PM2_5 --outdir maps\n\n"

                "  # 4) Batch: grid plotting\n"
                "  python3 smkplot.py --no-gui --pltyp grid --griddesc GRIDDESC --gridname 12US1_36-12-4 \\\n"
                    "                   --filepath report.csv --pollutant CO --outdir maps\n\n"
            "Notes\n"
            "- Delimiter tokens accepted by --delim: comma | semicolon | tab | pipe | space | \\\"\\t\\\"\n"
            "- GUI grid mode (ROW/COL) requires providing a GRIDDESC in the GUI.\n"
            "- Batch grid mode (for CSV/text) requires --pltyp grid, --griddesc, and --gridname.\n"
            "- NetCDF inputs (including Inline Point Sources) derive grid parameters from headers; --griddesc is ignored.\n"
            "- If Tk is unavailable, install python3-tk or use --no-gui.\n"
        ),
    )
    ap.add_argument('-f', '--filepath', help='Path to smkreport or ff10 input file; emissions list file allowed')
    ap.add_argument('--sector', default=None, help='Name of source sector; e.g., rwc, ptfire-rx, ptfire-wild, etc.')
    ap.add_argument('--filter-col', default=None, help='Name of column to filter on; e.g., release_id,scc, region_cd, etc.')
    ap.add_argument('--filter-start', default=None, help='Start value (inclusive) for filtering column.')
    ap.add_argument('--filter-end', default=None, help='End value (inclusive) for filtering column.')
    ap.add_argument('--filter-val', action='append', dest='filter_val', help='Discrete value(s) to keep when filtering by --filter-col. Repeat or separate with commas/spaces.')
    ap.add_argument('--delim', help='Explicit delimiter for emissions file (e.g., "," or "\\t").')
    ap.add_argument('--skiprows', type=int, help='Skip first N lines before header.')
    ap.add_argument('--comment', help='Comment character to ignore lines (e.g., #).')
    ap.add_argument('--encoding', help='File encoding (e.g., latin1, utf-8).')
    ap.add_argument('--county-shapefile', help='Path/URL to counties shapefile (.shp, .gpkg or .zip) for county plots.')
    ap.add_argument('--overlay-shapefile', action='append', help='Path/URL to auxiliary shapefile for overlaid on plots. Can be specified multiple times.')
    ap.add_argument('--filtered-by-overlay', nargs='?', const='intersect', help='Clip/filter data using overlay shapefile. Values: "intersect" (default if flag present), "clipped", "within".')
    ap.add_argument('--stack-groups', help='Path to STACK_GROUPS file (for Inline Point Source processing).')
    ap.add_argument('--griddesc', help='Path to GRIDDESC file (for batch grid plotting).')
    ap.add_argument('--gridname', help='Name of the grid in GRIDDESC (for batch grid plotting).')
    ap.add_argument('--pltyp', choices=['county', 'grid'], default='county', help='Plot by county or grid (batch mode). Default: county.')
    ap.add_argument('--force-lcc', action='store_true', help='Force legacy CONUS LCC projection in batch mode even without GRIDDESC (deprecated behavior).')
    ap.add_argument('--projection', choices=['auto','wgs84','lcc'], default='auto', help='Projection mode: auto (if gridname provided, use grid projection; else WGS84), wgs84 (always geographic), lcc (always LCC; grid-specific if grid provided else default CONUS).')
    ap.add_argument('--pollutant', help='Pollutant(s) to plot (batch mode or to preselect in GUI). Separate multiple values with comma or space.')
    ap.add_argument('--zoom-to-data', action='store_true', help='In batch mode, limit map extent to data (non-zero pollutant cells) with small padding.')
    ap.add_argument('--zoom-pad', type=float, default=0.02, help='Padding fraction to apply around data extent when using --zoom-to-data (default 0.02 = 2%%).')
    ap.add_argument('--bins', help='Custom colorbar ticks (comma or space separated). Plots remain continuous (linear/log).') # Custom bins for colorbar ticks (batch mode)
    ap.add_argument('--cmap', default='viridis', help='Matplotlib colormap name (e.g., viridis, plasma, turbo, Reds).')
    ap.add_argument('--workers', type=int, default=0, help='Number of parallel workers for batch plotting (0=auto).')
    ap.add_argument('--outdir', default='outputs', help='Output directory for batch mode (default to outputs).')
    ap.add_argument('--log-scale', action='store_true', help='Use log scale for color mapping in batch mode.')
    ap.add_argument('--self-test', action='store_true', help='Run a quick self-test: generate synthetic data and produce sample outputs to --outdir.')
    ap.add_argument('--run-mode', choices=['gui', 'batch'], help='Execution mode: "gui" to open interactive window, "batch" to run headless (default).')
    ap.add_argument('--log-file', help='Write logs (INFO..ERROR) to this file (appends). If directory given, a timestamped file is created.')
    ap.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], help='Logging level (default INFO).')
    ap.add_argument('--export-csv', action='store_true',  help='Export processed emissions data to CSV file(s).')
    ap.add_argument('--fill-nan', default=None, help='Value to fill missing data with (e.g. 0.0). Applies to both missing emission values and empty map regions (map holes).')
    ap.add_argument('--ncf-tdim', default='avg', help='NetCDF Time Dimension operation: avg|sum|max|min or specific time step index (0-based). Default: avg.')
    ap.add_argument('--ncf-zdim', default='0', help='NetCDF Layer Dimension operation: avg|sum|max|min or specific layer index (0-based). Default: 0 (layer 1).')

    args = ap.parse_args()

    # Intelligent handling: if filepath looks like a config file, redirect it
    config_payload = None
    config_path = None

    if args.filepath:
        lower_path = args.filepath.lower()
        if lower_path.endswith(('.json', '.yaml', '.yml')):
            logging.info("Input filepath '%s' detected as configuration. Switching mode.", args.filepath)
            try:
                config_payload, config_path = _load_config_dict(args.filepath)
                # Clear args.filepath so it can be populated from the config file
                args.filepath = None
            except Exception as exc:
                ap.error(f"Failed to read settings from {args.filepath}: {exc}")

    if config_payload:
        if not isinstance(config_payload, dict):
            ap.error(f"Settings file must contain an object at the top level: {config_path}")
        
        json_args = config_payload.get('arguments', config_payload)
        if not isinstance(json_args, dict):
            ap.error(f"Settings file must provide an 'arguments' object: {config_path}")
            
        for key, value in json_args.items():
            target_key = key
            if not hasattr(args, target_key):
                target_key = key.replace('-', '_')
            
            if target_key in {'json', 'yaml'} or not hasattr(args, target_key):
                continue
            
            default = ap.get_default(target_key)
            current = getattr(args, target_key)
            if current == default:
                setattr(args, target_key, value)
        
        # Preserve the config file format for output snapshot
        if config_path.lower().endswith(('.yaml', '.yml')):
            args.yaml = config_path
            args.json = None
        else:
            args.json = config_path
            args.yaml = None
        
        args.config_path = config_path
        args.json_payload = config_payload
    else:
        args.json_payload = None

    # Normalize --filter-val argument into a distinct ordered list
    args.filter_values = _normalize_list_arg(
        getattr(args, 'filter_val', None),
        getattr(args, 'filter_values', None)
    )

    # Normalize --pollutant argument into a list for batch processing
    args.pollutant_list = _normalize_list_arg(getattr(args, 'pollutant', None))
    args.pollutant_first = args.pollutant_list[0] if args.pollutant_list else None
    return args


def main():
    args = parse_args()
    # Setup logging early
    if args.log_file:
        log_path = args.log_file
        if os.path.isdir(log_path):
            ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            log_path = os.path.join(log_path, f'emission_gui_{ts}.log')
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper(), logging.INFO),
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging started -> %s", log_path)
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper(), logging.INFO),
            format='%(levelname)s %(message)s'
        )
    
    # Redirect Python warnings (e.g. DeprecationWarning, UserWarning) to the logging system
    logging.captureWarnings(True)

    def _excepthook(exc_type, exc, tb):
        logging.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))
        # Preserve default behavior after logging
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _excepthook

    # Determine execution mode from args
    run_mode = getattr(args, 'run_mode', None)

    # Only run in batch mode if explicitly requested
    if run_mode == 'batch':
        logging.info("Batch mode explicitly requested.")
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        from batch import _batch_mode  # Lazy import
        return _batch_mode(args)

    # Otherwise default to GUI (run_mode='gui' or run_mode=None)

    if run_mode is None:
        if len(sys.argv) == 1:
            logging.info("No arguments provided; defaulting to GUI mode.")
        else:
            logging.info("Run mode not specified; defaulting to GUI mode.")

    # GUI requested
    reason = None
    if not USING_TK:
        reason = f"Matplotlib backend '{matplotlib.get_backend()}' is not TkAgg"
    elif not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        reason = 'No DISPLAY environment variable'
    
    # Since user explicitly requested 'gui', we attempt to force it:
    if reason is not None:
        logging.info("GUI mode requested but environment issues detected (%s). Attempting to force TkAgg...", reason)
        if not matplotlib.get_backend().lower().startswith('tk'):
            try:
                matplotlib.use('TkAgg')
            except Exception as e:
                logging.warning("Backend switch failed: %s", e)
        # Re-evaluate
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
                # Try legacy naive check
                try:
                    import tkinter as tk
                    t = tk.Tk(); t.withdraw(); t.destroy()
                    reason = None
                    logging.info("Forced GUI: Tkinter display connection succeeded.")
                except Exception as e:
                    logging.error("Forced GUI failed: %s", e)
        else:
                reason = None # Assume success if DISPLAY exists or backend switch worked
    
    # Fallback to batch if GUI is impossible and force failed
    if reason is not None:
        logging.error("Unable to initialize GUI: %s. Falling back to batch mode.", reason)
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        from batch import _batch_mode  # Lazy import
        return _batch_mode(args)

    # GUI path
    try:
        import tkinter as tk  # re-import safe
        from gui import EmissionGUI  # Lazy import
        root = tk.Tk()
        app = EmissionGUI(
            root,
            args.filepath,
            args.county_shapefile,
            emissions_delim=args.delim,
            cli_args=args,
            app_version=SMKPLOT_VERSION,
        )
        # Pass batch-mode grid settings to the GUI instance if provided
        if args.griddesc:
            app.griddesc_path = args.griddesc
            app.griddesc_entry.delete(0, tk.END)
            app.griddesc_entry.insert(0, args.griddesc)
            app.load_griddesc()
            if args.gridname:
                # Check if the grid name is valid before setting
                available_grids = app.grid_name_menu['menu'].winfo_children()
                available_names = [item.cget('label') for item in available_grids]
                if args.gridname in available_names:
                    app.grid_name_var.set(args.gridname)
                    app.load_grid_shape()
        if args.pltyp:
            app.plot_by_var.set(args.pltyp)
            
    except Exception as e:
        logging.exception("Failed creating Tk root (final); falling back to batch mode")
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        from batch import _batch_mode  # Lazy import
        return _batch_mode(args)
    selected_pollutant = args.pollutant_first or args.pollutant
    if selected_pollutant:
        # Preselect pollutant if provided and present
        def _set_pol():
            if app.pollutants and selected_pollutant in app.pollutants:
                app.pollutant_var.set(selected_pollutant)
        root.after(500, _set_pol)
    root.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())