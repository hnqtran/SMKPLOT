"""
Batch processing module for SMKPLOT.

##############################################################################
# STRICT PARAMETER SOURCE RULES
# 1. Grid Generation Consistency:
#    - All grid generation (NetCDF or GRIDDESC) MUST produce identical GeoDataFrames.
#    - All grid generation MUST use the same identical function.
#    - REQUIRED CRS: EPSG:4326 (WGS84 Lat/Lon).
#    - REQUIRED COLUMNS: ROW (int), COL (int), GRID_RC (str "R_C"), geometry (Polygon).
#
# 2. Parameter Source Determination:
#    - NetCDF/Inline Files: MUST use internal Global Attributes (XORIG, XCELL, etc).
#      - DO NOT use external GRIDDESC files for NetCDF inputs.
#    - SMOKE Reports / FF10: MUST use external GRIDDESC file + Grid Name.
##############################################################################

This module handles headless (non-GUI) execution of the emissions plotting tool.
It supports:
- Loading configuration from command-line arguments, JSON, or YAML files.
- Processing multiple pollutants in parallel.
- Generating map plots (County or Grid based).
- Exporting processed data to CSV.
- Saving run configurations (snapshots) for reproducibility.
"""

import os
import logging
import copy
import json
import yaml
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib.colors import BoundaryNorm, LogNorm
import pyproj
from pathlib import Path
from shapely.geometry import Polygon, Point, box
from typing import Any, Dict

# Prefer the pyproj-distributed share directory when PROJ_LIB was not preset.
if not os.environ.get('PROJ_LIB'):
    _proj_share = Path(pyproj.__file__).resolve().parent / 'proj_dir' / 'share' / 'proj'
    if _proj_share.is_dir():
        os.environ['PROJ_LIB'] = os.environ['PROJ_DATA'] = str(_proj_share)

if os.environ.get('PROJ_LIB'):
    try:
        pyproj.datadir.set_data_dir(os.environ['PROJ_LIB'])
    except Exception:
        # Non-fatal; older pyproj versions may not expose datadir setter
        pass

from data_processing import (
    read_inputfile,
    read_shpfile,
    create_domain_gdf,
    detect_pollutants,
    map_latlon2grd,
    get_emis_fips,
    filter_dataframe_by_range,
    filter_dataframe_by_values,
    apply_spatial_filter
)
from utils import safe_sector_slug, serialize_attrs, normalize_delim, coerce_merge_key

from plotting import _draw_graticule, create_map_plot

from config import (
    USE_SPHERICAL_EARTH, key_cols,
    COUNTRY_COLS, TRIBAL_COLS, REGION_COLS, FACILITY_COLS, UNIT_COLS, REL_COLS,
    EMIS_COLS, SCC_COLS, POL_COLS, LAT_COLS, LON_COLS, COUNTRY_CODE_MAPPINGS
)




_PLOT_CONTEXT: Optional[Dict[str, Any]] = None
try:
    _PLOT_MP_CONTEXT = multiprocessing.get_context('fork')
except Exception:
    _PLOT_MP_CONTEXT = None


def _resolve_worker_count(num_jobs: int) -> int:
    # Aggressively use all cores if needed
    cpu = os.cpu_count() or 1
    if cpu <= 1:
        return 1
    # Use max available cores minus 1 for system responsiveness, but always at least 1
    return max(1, cpu - 1)


def _write_settings_snapshot(args, plots, files, attrs=None, input_basename=None):
    """Persist run configuration and generated artifacts for reuse."""
    try:
        os.makedirs(args.outdir, exist_ok=True)
        
        arg_dict = {
            k: getattr(args, k) for k in vars(args)
            if k not in {'json', 'yaml', 'json_payload', 'pollutant_list', 'pollutant_first','filter_values'}
        }
        
        payload = {
            'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
            'arguments': arg_dict,
            'outputs': {
                'output_directory': os.path.abspath(args.outdir),
                'plots': [os.path.abspath(p) for p in plots if p],
                'files': [os.path.abspath(p) for p in files if p],
            }
        }
        if attrs is not None:
            payload['outputs']['emis_df_attrs'] = serialize_attrs(attrs)
            # Add stack_groups_path to arguments if present in attrs
            if 'stack_groups_path' in attrs:
                payload['arguments']['stack_groups'] = attrs['stack_groups_path']
        
        # Prioritize sector name, fallback to input_basename if sector is missing
        sector_slug = safe_sector_slug(getattr(args, 'sector', None))
        if sector_slug and sector_slug != 'default':
            base_name = sector_slug
        elif input_basename:
            base_name = input_basename
        else:
            base_name = 'default'
        
        if getattr(args, 'yaml', None):
            yaml_path = os.path.join(args.outdir, f"{base_name}.yaml")
            payload['outputs']['settings_yaml'] = os.path.abspath(yaml_path)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(payload, f, sort_keys=False)
            logging.info("Saved settings snapshot to %s", yaml_path)
        else:
            json_path = os.path.join(args.outdir, f"{base_name}.json")
            payload['outputs']['settings_json'] = os.path.abspath(json_path)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            logging.info("Saved settings snapshot to %s", json_path)
    except Exception:
        logging.exception("Failed to write settings snapshot")


def _render_single_pollutant(pol: str, ctx: Dict[str, Any]) -> Tuple[str, str]:
    args = ctx['args']
    merged = ctx['merged']
    merged_plot = ctx['merged_plot']
    overlay_county = ctx['overlay_county']
    overlay_geom = ctx['overlay_geom']
    crs_proj = ctx['crs_proj']
    tf_fwd = ctx['tf_fwd']
    tf_inv = ctx['tf_inv']
    emis_df = ctx['emis_df']
    batch_src = ctx['batch_src']
    input_basename = ctx['input_basename']

    # Calculate figure size based on data aspect ratio to optimize map coverage.
    try:
        minx, miny, maxx, maxy = merged.total_bounds
        data_ratio = (maxy - miny) / (maxx - minx)
        # Target map width 8 inches.
        map_w = 8.0
        map_h = map_w * data_ratio
        
        # Add GENEROUS padding for title, stats, colorbar, and graticule labels
        # Width: Map + Left Margin + Right Margin + Colorbar (~2.5 inches)
        # Height: Map + Top Margin (Title) + Bottom Margin (~2.0 inches)
        fig_w = map_w + 2.5 
        fig_h = map_h + 2.0
        
        # Safety clamps to avoid extreme sizes
        fig_w = max(6.0, min(fig_w, 20.0))
        fig_h = max(5.0, min(fig_h, 20.0))
        
        figsize = (fig_w, fig_h)
    except Exception:
        # Fallback if bounds calc fails
        figsize = (11, 8.0)

    fig, ax = plt.subplots(figsize=figsize)
    pol_unit = None
    try:
        attrs = getattr(emis_df, 'attrs', None)

        
        if isinstance(attrs, dict):
            key_variants = []
            base_key = str(pol)
            for candidate in (pol, base_key.strip(), base_key.upper(), base_key.lower()):
                if candidate not in key_variants:
                    key_variants.append(candidate)
            for attr_name in ('units_map', 'pollutant_units'):
                mapping = attrs.get(attr_name)
                if not isinstance(mapping, dict):
                    continue
                for key in key_variants:
                    if key in mapping and mapping[key]:
                        pol_unit = mapping[key]
                        break
                if pol_unit:
                    break
    except Exception:
        pol_unit = None

    bins = None
    if args.bins:
        try:
            parts = [p for p in args.bins.replace(',', ' ').split() if p]
            bins = sorted(set(float(p) for p in parts))
        except Exception:
            bins = None



    # Auto-enable zoom_to_data if spatial filtering is active and no explicit override exists.
    zoom_flag = args.zoom_to_data
    filter_mode_arg = getattr(args, 'filtered_by_overlay', None)
    if not zoom_flag and filter_mode_arg and str(filter_mode_arg).lower() not in ('false', 'none'):
        zoom_flag = True
        logging.info("Auto-enabling zoom_to_data because spatial filtering is active.")

    # Use shared plotting function
    create_map_plot(
        gdf=merged_plot,
        column=pol,
        title=f"{pol} Emissions" if not batch_src else f"{batch_src} - {pol}",
        ax=ax,
        cmap_name=args.cmap or 'jet',
        bins=bins,
        log_scale=args.log_scale,
        unit_label=pol_unit,
        overlay_counties=overlay_county,
        overlay_shape=overlay_geom,
        crs_proj=crs_proj,
        tf_fwd=tf_fwd,
        tf_inv=tf_inv,
        zoom_to_data=zoom_flag,
        zoom_pad=args.zoom_pad
    )

    # Axis settings handled by create_map_plot (borders enabled)



    stats_source = None
    if zoom_flag:
        try:
            # Re-calculate statistics based on visible extent
            # Get current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Use spatial indexing to find features within the plot window
            # .cx handles intersection with the bounding box
            visible_data = merged_plot.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
            
            if not visible_data.empty and pol in visible_data.columns:
                stats_source = visible_data[pol]
                logging.debug("Statistics updated for visible region (%d features)", len(visible_data))
            else:
                logging.debug("No features found in visible region for stats calc; falling back to full data")
        except Exception:
            logging.exception("Failed to calculate stats for visible region; falling back to full data")

    if stats_source is None and isinstance(emis_df, pd.DataFrame) and pol in emis_df.columns:
        stats_source = emis_df[pol]

    if stats_source is None:
        stats_source = pd.Series([]) # Fallback empty

    numeric_values = pd.to_numeric(stats_source, errors='coerce').dropna()

    if numeric_values.empty:
        min_val = mean_val = max_val = sum_val = float('nan')
    else:
        arr = numeric_values.to_numpy(dtype=float, copy=False)
        min_val = float(arr.min())
        mean_val = float(arr.mean())
        max_val = float(arr.max())
        sum_val = float(arr.sum())

    qa_series_full = pd.to_numeric(emis_df[pol], errors='coerce').dropna()
    qa_sum_full = float(qa_series_full.sum()) if not qa_series_full.empty else float('nan')
    
    # Only warn about sum mismatch if NOT zooming (or if we expect them to match)
    # If zooming, sum_val (visible) should typically be <= qa_sum_full (total)
    if not zoom_flag:
        if not np.isclose(sum_val, qa_sum_full, rtol=1e-6, atol=1e-6):
            logging.warning(
                "Sum mismatch for %s: plot sum=%s, emis_df sum=%s", pol, sum_val, qa_sum_full
            )
    else:
        # Optional debug logging for zoom case
        logging.debug("Zoom active: Visible Sum=%s, Total Sum=%s", sum_val, qa_sum_full)

    def _fmt_stat(value: float) -> str:
        if np.isfinite(value):
            return f"{value:.4g}"
        return "--"

    stats_str = (
        f"Min: {_fmt_stat(min_val)}  Mean: {_fmt_stat(mean_val)}  "
        f"Max: {_fmt_stat(max_val)}  Sum: {_fmt_stat(sum_val)}"
    )
    if pol_unit:
        stats_str += f" ({pol_unit})"
    if batch_src:
        title = f"{pol} emissions from {batch_src}"
    else:
        title = f"{pol} emission"
    if pol_unit:
        title = f"{title} [{pol_unit}]"

    ax.set_title(f"{title}\n{stats_str}")
    


    pltyp_raw = getattr(args, 'pltyp', None)
    if pltyp_raw:
        pltyp_token = ''.join(ch if (ch.isalnum() or ch in {'-', '_'}) else '_' for ch in str(pltyp_raw))
        pltyp_suffix = f"_{pltyp_token.strip('_') or 'plot'}"
    else:
        pltyp_suffix = ''

    ncf_info = ""
    st = emis_df.attrs.get('source_type', '')
    # Append NetCDF dimension operators if processing gridded/inline data to the filename
    if st and ('netcdf' in st or 'inline' in st):
        tdim = getattr(args, 'ncf_tdim', 'avg')
        zdim = getattr(args, 'ncf_zdim', '0')
        ncf_info = f".t_{tdim}.z_{zdim}"

    if args.filter_col:
        fname_base = f"{input_basename}_{pol}{pltyp_suffix}{ncf_info}.filterby_{args.filter_col}"
    else:
        fname_base = f"{input_basename}_{pol}{pltyp_suffix}{ncf_info}"
    
    # Add overlay filter mode to filename if active
    filter_mode_arg = getattr(args, 'filtered_by_overlay', None)
    if filter_mode_arg and str(filter_mode_arg).lower() not in ('false', 'none'):
        fname_base += f".overlay_{filter_mode_arg}"

    out_file = os.path.join(args.outdir, f"{fname_base}.png")

    # Use strict bounding box with padding to keep labels visible but trim whitespace
    fig.savefig(out_file, dpi=180, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return pol, os.path.abspath(out_file)


def _plot_worker(pol: str) -> Tuple[str, str]:
    if _PLOT_CONTEXT is None:
        raise RuntimeError("Plot context is not initialized")
    return _render_single_pollutant(pol, _PLOT_CONTEXT)


def _batch_mode(args):
    generated_plots = []
    generated_files = []
    preprocessed_csv_path = None
    imported_preprocessed_path = None
    emis_attrs: dict = {}
    input_basename_var = None  # Will be set after loading data

    def _finish(code: int):
        if code == 0:
            _write_settings_snapshot(
                args,
                generated_plots,
                generated_files,
                attrs=emis_attrs,
                input_basename=input_basename_var
            )
        return code

    # Optional self-test harness (runs before reading real data if requested)
    if args.self_test:
        try:
            os.makedirs(args.outdir, exist_ok=True)
            # Build small synthetic emissions DataFrame (counties mode) with two fictitious FIPS codes
            synth = pd.DataFrame({
                'FIPS': ['01001','01003','01005','01007'],
                'NOX': [5.0, 10.0, 0.0, 20.0],
                'SO2': [1.0, 0.5, 2.0, 0.0]
            })
            # Minimal square polygons (fake geometries) in WGS84
            polys = []
            for i, f in enumerate(synth['FIPS']):
                x0 = -90 + i*0.5
                y0 = 30 + i*0.5
                polys.append(Polygon([(x0,y0),(x0+0.4,y0),(x0+0.4,y0+0.4),(x0,y0+0.4)]))
            gdf = gpd.GeoDataFrame({'FIPS': synth['FIPS']}, geometry=polys, crs='+proj=longlat +datum=WGS84')
            merged = gdf.merge(synth, on='FIPS', how='left')
            for pol in ['NOX','SO2']:
                fig, ax = plt.subplots(figsize=(6,3))
                merged.plot(column=pol, ax=ax, legend=True, cmap='jet', edgecolor='black', linewidth=0.2)
                # Zoom-to-data demonstration
                if args.zoom_to_data:
                    try:
                        valid = merged[merged[pol].notna()]
                        if not valid.empty:
                            minx, miny, maxx, maxy = valid.total_bounds
                            dx = (maxx - minx) * 0.05 if maxx > minx else 0.1
                            dy = (maxy - miny) * 0.05 if maxy > miny else 0.1
                            ax.set_xlim(minx - dx, maxx + dx)
                            ax.set_ylim(miny - dy, maxy + dy)
                    except Exception:
                        pass
                ax.set_title(f"SELF-TEST {pol}")
                fig.tight_layout()
                out_path = os.path.join(args.outdir, f"selftest_{pol}.png")
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                generated_plots.append(os.path.abspath(out_path))
            logging.info("Self-test artifacts written to %s (files starting with selftest_)", args.outdir)
        except Exception:
            logging.exception("Self-test failed")
        # Continue with normal processing if real inputs provided; else exit early
        if not args.filepath:
            return _finish(0)
    
    
    # Retrieve json_payload
    json_payload = getattr(args, 'json_payload', None)

    # -----------------------------------------------------
    # Unified File Loading Logic
    # -----------------------------------------------------
    input_path = args.filepath
    
    # Check JSON filepath if still None
    if not input_path and isinstance(json_payload, dict):
        input_path = json_payload.get('arguments', {}).get('filepath')

    if not input_path:
        logging.error("Batch mode requires input file(s). Use --filepath.")
        return 2

    # Normalize input_path to list of strings
    if isinstance(input_path, (list, tuple)):
        # Already a list?
        file_list = [str(p) for p in input_path if p]
    elif isinstance(input_path, str):
        # Support semicolon splitting for multiple files passed as single string
        file_list = [p.strip() for p in input_path.split(';') if p.strip()]
    else:
        file_list = []
    
    # Resolve relative paths against JSON directory if applicable
    json_ref_dir = os.path.dirname(args.json) if getattr(args, 'json', None) else None
    resolved_files = []
    for f in file_list:
        if f and not os.path.isabs(f) and json_ref_dir:
            resolved_files.append(os.path.abspath(os.path.join(json_ref_dir, f)))
        elif f:
            resolved_files.append(os.path.abspath(f))
    
    if not resolved_files:
        logging.error("No valid input files found.")
        return 2

    # Verify existence
    missing = [f for f in resolved_files if not os.path.exists(f)]
    if missing:
        logging.error("Input file(s) not found: %s", missing)
        return 1

    msg_load = f"Loading emissions data from: {resolved_files}"
    logging.info(msg_load)
    print(msg_load)
    
    # Propagate grid/inline arguments to environment variables for reader
    # Note: NetCDF reader (including Inline proxy) is designed to prefer headers.
    # We pass STACK_GROUPS to allow Inline conversion. GRIDDESC is passed only if needed, 
    # but for NetCDF files we expect headers to define the grid.
    if getattr(args, 'stack_groups', None):
        os.environ['STACK_GROUPS'] = args.stack_groups
    
    # Note: NetCDF reader (including Inline proxy) is designed to prefer file headers.
    # We strictly rely on header attributes for NetCDF grid definitions.
    
    emis_df = None
    raw_df = None

    # Parse NetCDF dimension args
    ncf_params = {}
    
    # Time Dimension
    t_arg = getattr(args, 'ncf_tdim', 'avg')
    if str(t_arg).isdigit():
        ncf_params['tstep_idx'] = int(t_arg)
        ncf_params['tstep_op'] = 'select'
    else:
        # map common aliases
        op = str(t_arg).lower()
        if op in ('avg', 'mean', 'average'):
            ncf_params['tstep_op'] = 'mean'
        elif op in ('sum', 'total'):
            ncf_params['tstep_op'] = 'sum'
        elif op in ('max', 'maximum'):
            ncf_params['tstep_op'] = 'max'
        elif op in ('min', 'minimum'):
            ncf_params['tstep_op'] = 'min'
        else:
            # Default fallback
            ncf_params['tstep_op'] = 'mean'

    # Layer Dimension
    z_arg = getattr(args, 'ncf_zdim', '0')
    if str(z_arg).isdigit():
        ncf_params['layer_idx'] = int(z_arg)
        ncf_params['layer_op'] = 'select'
    else:
        op = str(z_arg).lower()
        ncf_params['layer_idx'] = None # Override default 0
        if op in ('avg', 'mean', 'average'):
            ncf_params['layer_op'] = 'mean'
        elif op in ('sum', 'total'):
            ncf_params['layer_op'] = 'sum'
        elif op in ('max', 'maximum'):
            ncf_params['layer_op'] = 'max'
        elif op in ('min', 'minimum'):
            ncf_params['layer_op'] = 'min'
        else:
            # Default fallback
            ncf_params['layer_idx'] = 0
            ncf_params['layer_op'] = 'select'

    # Define notify callback for batch mode to display INFO messages to screen
    def batch_notify(level, message):
        """Display loader notifications to screen in batch mode."""
        lvl = (level or 'INFO').upper()
        msg = str(message).strip()
        if lvl == 'INFO':
            print(f"INFO: {msg}")
        elif lvl == 'WARNING':
            print(f"WARNING: {msg}")
        elif lvl == 'ERROR':
            print(f"ERROR: {msg}")
        # Also log it
        getattr(logging, lvl.lower(), logging.info)(msg)

    try:
        norm_delim = normalize_delim(args.delim)
        # read_inputfile handles list of paths
        emis_df, raw_df = read_inputfile(
            fpath=resolved_files,
            sector=args.sector,
            delim=norm_delim,
            skiprows=args.skiprows,
            comment=args.comment,
            encoding=args.encoding,
            flter_col=args.filter_col,
            flter_start=args.filter_start,
            flter_end=args.filter_end,
            flter_val=getattr(args, 'filter_values', None),
            return_raw=False,
            ncf_params=ncf_params,
            notify=batch_notify
        )

    except Exception as e:
        if "No valid FIPS code columns found" in str(e):
            logging.error("Input file is not supported: No valid FIPS/Region columns found.")
            return 1
        logging.exception("Error reading emissions: %s", e)
        return 1
    
    # -----------------------------------------------------
    # End Unified Loading
    # -----------------------------------------------------

    
    if emis_df is None:
        logging.error("No emissions data available for processing.")
        return 1

    # Build FIPS column
    try:
        emis_df = get_emis_fips(emis_df)
    except ValueError as e:
        if "No valid FIPS code columns found" in str(e):
            logging.error("Input file is not supported: No valid FIPS/Region columns found.")
            return 1
        logging.exception("Error building FIPS columns")
        return 1

    # Apply units_map from JSON payload if available (overrides input data)
    # Must be done after get_emis_fips because pd.concat in get_emis_fips may drop attrs
    if isinstance(json_payload, dict):
        units_map_arg = json_payload.get('arguments', {}).get('units_map')
        if isinstance(units_map_arg, dict) and emis_df is not None:
            # Merge with existing units_map if any, preferring the argument
            existing_map = emis_df.attrs.get('units_map', {})
            if not isinstance(existing_map, dict): existing_map = {}
             
            for k, v in units_map_arg.items():
                existing_map[k] = v
             
            emis_df.attrs['units_map'] = existing_map
            logging.info("Applied units_map from configuration settings.")

    
    # Handle fill-nan option
    fill_nan_val = getattr(args, 'fill_nan', None)
    should_fill = False
    fill_number = 0.0
    
    if fill_nan_val is not None:
        if isinstance(fill_nan_val, bool) and not fill_nan_val:
            should_fill = False
        elif isinstance(fill_nan_val, str) and fill_nan_val.lower() == 'false':
            should_fill = False
        else:
            try:
                fill_number = float(fill_nan_val)
                should_fill = True
            except (ValueError, TypeError):
                should_fill = False

    if should_fill:
        try:
            # Only fill numeric columns
            num_cols = emis_df.select_dtypes(include=[np.number]).columns
            if not num_cols.empty:
                emis_df[num_cols] = emis_df[num_cols].fillna(fill_number)
                logging.info("Filled NaN values with %.4f", fill_number)
        except Exception:
            logging.exception("Failed to fill NaN values")

    # Capture emissions DataFrame attributes for later use
    emis_attrs = dict(getattr(emis_df, 'attrs', {}) or {})


    # Update input_basename; use sector if specified in arguments or in JSON payload
    start_sector = getattr(args, 'sector', None)
    json_sector = None
    if json_payload:
        json_sector = json_payload.get('arguments', {}).get('sector', None)
    
    input_basename = start_sector or json_sector or (os.path.basename(resolved_files[0]) if resolved_files else 'output')
    input_basename_var = input_basename  # Assign to nonlocal variable for _finish closure

    # Detect NetCDF input
    is_ncf_input = False
    if resolved_files and resolved_files[0].lower().endswith(('.nc', '.ncf')):
        is_ncf_input = True

    # Validate inputs based on plot type
    
    # Strict validation for re-imported grid data
    if args.pltyp == 'grid' and emis_df.attrs.get('is_preprocessed'):
        if 'GRID_RC' not in emis_df.columns:
            logging.error("Plot Type is 'grid' but input pre-processed CSV(s) missing 'GRID_RC' column. Aborting.")
            return 1

    overlay_county = None
    overlay_geom = None

    if is_ncf_input:
        # NetCDF (Grid or Inline) uses internal headers for grid definition.
        if args.griddesc or args.gridname:
            logging.warning("NetCDF input detected: Ignoring provided --griddesc/--gridname. Grid parameters will be taken from NetCDF input file headers instead.")
            
    else:
        if args.pltyp == 'grid':
            if not args.griddesc or not args.gridname:
                logging.error("Batch grid plotting (--pltyp grid) requires --griddesc and --gridname.")
                return 2
        if args.pltyp == 'county':
            if not args.county_shapefile:
                logging.error("Batch county plotting (--pltyp county) requires --county-shapefile.")
                return 2

    # Load geometry based on plot type
    if is_ncf_input:
        try:
            from ncf_processing import create_ncf_domain_gdf
            # Use the proxy file for geometry if available (ensures correct 2D grid for Inline sources)
            geom_source_path = emis_df.attrs.get('proxy_ncf_path', args.filepath)
             
            logging.info("Generating grid geometry from NetCDF file parameters (%s)...", "proxy" if geom_source_path != args.filepath else "original")
            base_geom = create_ncf_domain_gdf(geom_source_path)
            merge_on = 'GRID_RC'
        except Exception:
            logging.exception("Error extracting grid geometry from NetCDF file.")
            return 1
        
        # Overlay county if provided, relaxed requirement
        if args.county_shapefile:
            try:
                overlay_county = read_shpfile(args.county_shapefile, False)
                logging.info("Loaded county shapefile as overlay.")
            except Exception:
                logging.warning(f"Failed to load {args.county_shapefile} for overlay plotting. Ignoring.")
                overlay_county = None

    elif args.pltyp == 'grid':
        try:
            base_geom = create_domain_gdf(args.griddesc, args.gridname)
            merge_on = 'GRID_RC'
        except Exception as e:
            logging.exception(f"Error creating grid from GRIDDESC {args.griddesc}")
            return 1
        
        ## If source type = ff10_point, map latitude/longitude to grid cells
        if emis_df.attrs.get('source_type') == 'ff10_point':
            emis_df = map_latlon2grd(emis_df, base_geom)

    if args.pltyp == 'county':
        try:
            base_geom = read_shpfile(args.county_shapefile, True)
            merge_on = 'FIPS'
        except Exception as e:
            logging.exception(f"Error reading required county shapefile {args.county_shapefile}")
            return 1

    if args.county_shapefile:
        try:
            overlay_county = read_shpfile(args.county_shapefile, False)            
        except Exception:
            logging.exception(f"Failed to load {args.county_shapefile} for overlay plotting. Ignoring overlay.")
            overlay_county = None

    # Optional overlay when a shapefile is supplied (e.g., for grid runs)
    overlay_geom = None
    ov_paths = getattr(args, 'overlay_shapefile', None)
    
    if ov_paths:
        # Normalize to list (handle string, semicolon separation, lists)
        ov_file_list = []
        if isinstance(ov_paths, str):
            ov_file_list = [p.strip() for p in ov_paths.split(';') if p.strip()]
        elif isinstance(ov_paths, (list, tuple)):
            for item in ov_paths:
                if isinstance(item, str):
                    ov_file_list.extend([p.strip() for p in item.split(';') if p.strip()])
                else:
                    ov_file_list.append(str(item))
        
        # Load and combine
        loaded_parts = []
        for fpath in ov_file_list:
            try:
                # Resolve path similar to inputs (optional check omitted for brevity, rely on read_shpfile)
                part_gdf = read_shpfile(fpath, False)
                if part_gdf is not None and not part_gdf.empty:
                    loaded_parts.append(part_gdf)
            except Exception:
                logging.exception(f"Failed to load overlay shapefile: {fpath}. Skipping.")
        
        if loaded_parts:
            if len(loaded_parts) == 1:
                overlay_geom = loaded_parts[0]
            else:
                try:
                    # Combine multiple layers
                    # Use First layer's CRS as target
                    target = loaded_parts[0]
                    target_crs = target.crs
                    
                    final_list = [target]
                    for other in loaded_parts[1:]:
                        if target_crs and other.crs and other.crs != target_crs:
                            other = other.to_crs(target_crs)
                        final_list.append(other)
                    
                    overlay_geom = pd.concat(final_list, ignore_index=True)
                    logging.info("Combined %d overlay shapefiles.", len(final_list))
                except Exception:
                    logging.exception("Failed to combine multiple overlay shapefiles. Using first successfully loaded file.")
                    overlay_geom = loaded_parts[0]

    # Detect pollutants
    pollutants = detect_pollutants(emis_df)

    if not pollutants:
        logging.error("No pollutant columns detected.")
        return 1

    requested: List[str] = []
    try:
        raw_requested = getattr(args, 'pollutant_list', None)
        if raw_requested:
            seen_req = set()
            cleaned = []
            for pol in raw_requested:
                if not pol:
                    continue
                if pol not in seen_req:
                    seen_req.add(pol)
                    cleaned.append(pol)
            requested = cleaned
    except Exception:
        requested = []
    if not requested and args.pollutant:
        requested = [args.pollutant]
    if requested:
        missing = [p for p in requested if p not in pollutants]
        if missing:
            logging.error(
                "Requested pollutant(s) %s not found. Available: %s",
                ', '.join(missing),
                ', '.join(pollutants),
            )
            return 1
    os.makedirs(args.outdir, exist_ok=True)
    
    # Simple groupby and sum
    # NOTE: We aggregate into 'emis_agg' for plotting but keep 'emis_df' intact for export.
    emis_agg = None
    try:
        attrs = dict(getattr(emis_df, 'attrs', {}))
        try:
            grouped = emis_df.groupby(merge_on, dropna=False, sort=False, observed=False)
        except TypeError:
            grouped = emis_df.groupby(merge_on, sort=False, observed=False)
        if requested:
            emis_agg = grouped[requested].sum(numeric_only=True).reset_index()
        else:
            emis_agg = grouped.sum(numeric_only=True).reset_index()
        emis_agg.attrs = attrs
    except Exception:
        logging.exception("Unexpected error while aggregating emissions data")

    if merge_on == 'FIPS':
        try:
            if 'FIPS' in emis_agg.columns:
                emis_agg = emis_agg.copy()
                emis_agg['FIPS'] = coerce_merge_key(emis_agg['FIPS'], pad=6)
        except Exception:
            logging.exception("Failed to normalize emissions FIPS prior to merge")
        try:
            if 'FIPS' in base_geom.columns:
                base_geom = base_geom.copy()
                base_geom['FIPS'] = coerce_merge_key(base_geom['FIPS'], pad=6)
        except Exception:
            logging.exception("Failed to normalize geometry FIPS prior to merge")

    try:
        merged = base_geom.merge(emis_agg, on=merge_on, how='left', sort=False)
    except Exception:
        logging.exception("Failed to merge emissions with geometry")
        return 1
    
    # Fill NaNs in merged geometry if requested (fixes holes in map)
    if should_fill:
        try:
            cols_to_fill = [p for p in pollutants if p in merged.columns]
            if cols_to_fill:
                merged[cols_to_fill] = merged[cols_to_fill].fillna(fill_number)
                logging.info("Filled NaN values in merged geometry with %.4f", fill_number)
        except Exception:
            logging.exception("Failed to fill NaN values in merged dataframe")

    # Filter by overlay if requested
    filter_arg = getattr(args, 'filtered_by_overlay', None)
    
    # Resolve boolean/None/String inputs to a canonical mode string
    filter_mode = None
    if filter_arg is None:
        filter_mode = None
    elif isinstance(filter_arg, bool):
        filter_mode = 'intersect' if filter_arg else None
    elif isinstance(filter_arg, str):
        s = filter_arg.strip().lower()
        if s in ('', 'false', 'null', 'undefined', 'none'):
            filter_mode = None
        else:
            filter_mode = s
    else:
        filter_mode = str(filter_arg).strip().lower()

    if filter_mode:
        # Update args with the canonical mode so it propagates to title/filename generation
        args.filtered_by_overlay = filter_mode

        if overlay_geom is None:
            logging.warning("filtered_by_overlay='%s' requested but no overlay shapefile loaded.", filter_mode)
        else:
            try:
                logging.info("Applying overlay filter mode: '%s'", filter_mode)
                merged = apply_spatial_filter(merged, overlay_geom, filter_mode)
                 
                if filter_mode in ('clipped', 'intersect', 'within'):
                    logging.info("Filtering complete. Remaining features: %d", len(merged))

            except Exception:
                logging.exception("Failed to apply overlay filter '%s'", filter_mode)

    # Export processed emissions data to CSV (post-filtering)
    csv_from_json = False
    if isinstance(json_payload, dict):
        csv_from_json = json_payload.get('arguments', {}).get('export_csv', False)

    if (getattr(args, 'export_csv', False) or csv_from_json):
        # Create output directory if it doesn't exist
        os.makedirs(args.outdir, exist_ok=True)

        # construct filename
        fname_base = f"{input_basename}"
        
        # Add NetCDF dimension info for CSV export if applicable
        st = emis_df.attrs.get('source_type', '')
        if st and ('netcdf' in st or 'inline' in st):
            tdim = getattr(args, 'ncf_tdim', 'avg')
            zdim = getattr(args, 'ncf_zdim', '0')
            fname_base += f".t_{tdim}.z_{zdim}"

        if args.filter_col:
            fname_base += f".filterby_{args.filter_col}"
        
        # Add overlay filter mode to filename if active (using the resolved filter_mode variable)
        if filter_mode:
            fname_base += f".overlay_{filter_mode}"
            
        csv_outpath = os.path.join(args.outdir, f"{fname_base}.pivotted.csv")
        
        try:
            # If spatial filtering occurred, we MUST use 'merged' to reflect the filter.
            # However, 'merged' contains geometry and shapefile attributes.
            # 'emis_df' contains only the original data columns.
            # If filter_mode is active, we export the subset of 'merged'.
             
            if filter_mode:
                # Ensure we don't carry the geometry column into the CSV
                # Instead of exporting the aggregated 'merged' DF (which might lose columns),
                # we filter the original 'emis_df' to keep only rows present in the spatial subset.
                # We use the merge key to identify valid rows.
                try:
                    if merge_on and merge_on in merged.columns and merge_on in emis_df.columns:
                        valid_keys = merged[merge_on].unique()
                        export_df = emis_df[emis_df[merge_on].isin(valid_keys)]
                    else:
                        # Fallback: if keys don't match, export merged without geom
                        export_df = pd.DataFrame(merged.drop(columns='geometry', errors='ignore'))
                except Exception:
                    export_df = pd.DataFrame(merged.drop(columns='geometry', errors='ignore'))
                 
                logging.info("Exporting spatially filtered data (from filtered_by_overlay='%s')", filter_mode)
            else:
                # If no spatial filter, prefer the clean 'emis_df' (original behavior)
                # Note: Unlike previous versions which exported the plotting aggregation (County-level),
                # this now exports the full source-level data (emis_df) to preserve columns like SCC.
                # This results in a larger file (more rows) but retains granular detail.
                export_df = emis_df
            
            # Filter columns based on key_cols from config to reduce size
            # This ensures we export only relevant metadata and pollutants.
            if key_cols:
                # Identify columns to keep: those in key_cols, pollutants, or FIPS
                # Note: pollutants list was detected earlier
                cols_to_keep = set(key_cols)

                # If user requested specific pollutants, limit export to those ONLY IF reading pre-processed CSV.
                # If reading RAW files (FF10/List), user wants to keep ALL pollutants in the export.
                is_reused = emis_df.attrs.get('is_preprocessed', False)
                 
                if requested and is_reused:
                        cols_to_keep.update([p for p in requested if p in export_df.columns])
                else:
                        cols_to_keep.update(pollutants)

                cols_to_keep.add('FIPS') # Always keep FIPS as primary identifier
                cols_to_keep.add('GRID_RC') # Keep grid cell ID if available (from lat/lon mapping)
                 
                # Reorder columns: FIPS/GRID_RC first, then others (preserving original order)
                ordered_cols = []
                for special in ['FIPS', 'GRID_RC']:
                    if special in export_df.columns and special in cols_to_keep:
                        ordered_cols.append(special)
                 
                # Append remaining columns
                ordered_cols.extend([c for c in export_df.columns if c in cols_to_keep and c not in ordered_cols])
                 
                export_df = export_df[ordered_cols]
            
            # Collapse rows with same key columns by summing pollutant values
            # This handles cases where input had multiple rows per key (e.g. split by CAS or extraneous metadata)
            # but we are exporting a simplified view (only key_cols + pollutants).
             
            # Ensure identifying columns (FIPS, GRID_RC) are NOT treated as pollutants to be summed
            safe_identifiers = {'FIPS', 'GRID_RC', 'fips', 'grid_rc', 'ROW', 'COL', 'X', 'Y'}
             
            # Determine sum_cols as intersection of (requested or pollutants) and current columns
            # AND ensure they are numeric.
            target_pols = set(requested) if requested else set(pollutants)
             
            sum_cols = [c for c in export_df.columns if c in target_pols and c not in safe_identifiers and pd.api.types.is_numeric_dtype(export_df[c])]
            grp_cols = [c for c in export_df.columns if c not in sum_cols]
             
            if sum_cols and grp_cols:
                try:
                    # Create a copy immediately to avoid SettingWithCopyWarning and defragment
                    export_df = export_df.copy()
                     
                    # Remove explicit fillna to preserve sparse structure (reduce CSV size)
                    # export_df[sum_cols] = export_df[sum_cols].fillna(0)
                     
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Use min_count=1 to preserve NaNs where no data exists (sparse), 
                        # preventing conversion of NaNs to 0.0 which bloats CSV size.
                        # observed=True is CRITICAL to prevent expanding all Categorical combinations (MemoryError)
                        export_df = export_df.groupby(grp_cols, as_index=False, dropna=False, observed=True)[sum_cols].sum(min_count=1)
                          
                        # Remove rows where ALL sum_cols are 0.0 or NaN (no emissions)
                        # This cleans up "ghost" rows that have keys but no data.
                        if sum_cols:
                            # Check if sum along pollutant axis is effectively 0/NaN
                            # We keep rows where at least one pollutant > 0 or < 0 (non-zero)
                            # Note: is.na() check handles NaNs, == 0 handles zeros.
                            mask_valid = (export_df[sum_cols].fillna(0) != 0).any(axis=1)
                            export_df = export_df[mask_valid]
                except Exception as e:
                    logging.warning("Aggregation failed (%s); falling back to simple deduplication.", e)
                    export_df = export_df.drop_duplicates()
            else:
                export_df = export_df.drop_duplicates()

            # FIPS padding removed to preserve original input format as requested.
            # if 'FIPS' in export_df.columns:
            #    export_df['FIPS'] = coerce_merge_key(export_df['FIPS'], pad=6)

            msg = f"Writing processed data to CSV: {csv_outpath} ... (This may take a moment for large files)"
            logging.info(msg)
            print(msg)
             
            export_df.to_csv(csv_outpath, index=False)
             
            msg_done = f"Successfully exported CSV to: {csv_outpath}"
            logging.info(msg_done)
            print(msg_done)
             
            generated_files.append(os.path.abspath(csv_outpath))
        except Exception:
            logging.exception("Failed to export CSV to %s", csv_outpath) 

    # Determine projection (only use LCC when grid + GRIDDESC provided, unless --force-lcc)
    crs_proj = None; tf_fwd = None; tf_inv = None
    if args.projection == 'wgs84':
        crs_proj = None  # keep geographic
    elif is_ncf_input and args.projection != 'wgs84':
        try:
            from ncf_processing import read_ncf_grid_params
            coord_params, _ = read_ncf_grid_params(args.filepath)
            gdtyp, p_alpha, p_beta, _p_gamma, x_cent, y_cent = coord_params
            if gdtyp == 2:
                a_b = "+a=6370000.0 +b=6370000.0" if USE_SPHERICAL_EARTH else "+ellps=WGS84 +datum=WGS84"
                proj4 = f"+proj=lcc +lat_1={p_alpha} +lat_2={p_beta} +lat_0={y_cent} +lon_0={x_cent} {a_b} +x_0=0 +y_0=0 +units=m +no_defs"
                crs_proj = pyproj.CRS.from_proj4(proj4)
                tf_fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_proj, always_xy=True)
                tf_inv = pyproj.Transformer.from_crs(crs_proj, pyproj.CRS.from_epsg(4326), always_xy=True)
        except Exception:
            crs_proj = None; tf_fwd = None; tf_inv = None
    elif args.projection == 'lcc':
        # attempt grid-specific if possible, else default CONUS
        if args.griddesc and args.gridname:
            try:
                from data_processing import extract_grid
                coord_params, _ = extract_grid(args.griddesc, args.gridname)
                _, p_alpha, p_beta, _p_gamma, x_cent, y_cent = coord_params
                a_b = "+a=6370000.0 +b=6370000.0" if USE_SPHERICAL_EARTH else "+ellps=WGS84 +datum=WGS84"
                proj4 = f"+proj=lcc +lat_1={p_alpha} +lat_2={p_beta} +lat_0={y_cent} +lon_0={x_cent} {a_b} +x_0=0 +y_0=0 +units=m +no_defs"
                crs_proj = pyproj.CRS.from_proj4(proj4)
                tf_fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_proj, always_xy=True)
                tf_inv = pyproj.Transformer.from_crs(crs_proj, pyproj.CRS.from_epsg(4326), always_xy=True)
            except Exception:
                crs_proj = None; tf_fwd = None; tf_inv = None
        if crs_proj is None:
            try:
                a_b = "+a=6370000.0 +b=6370000.0" if USE_SPHERICAL_EARTH else "+ellps=WGS84 +datum=WGS84"
                proj4 = f"+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-96 {a_b} +x_0=0 +y_0=0 +units=m +no_defs"
                crs_proj = pyproj.CRS.from_proj4(proj4)
                tf_fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_proj, always_xy=True)
                tf_inv = pyproj.Transformer.from_crs(crs_proj, pyproj.CRS.from_epsg(4326), always_xy=True)
            except Exception:
                crs_proj = None; tf_fwd = None; tf_inv = None
    elif args.griddesc and args.gridname and args.pltyp != 'county':
        # For auto mode: if gridname provided, use grid projection (unless county plot)
        try:
            from data_processing import extract_grid
            coord_params, _ = extract_grid(args.griddesc, args.gridname)
            _, p_alpha, p_beta, _p_gamma, x_cent, y_cent = coord_params
            a_b = "+a=6370000.0 +b=6370000.0" if USE_SPHERICAL_EARTH else "+ellps=WGS84 +datum=WGS84"
            proj4 = f"+proj=lcc +lat_1={p_alpha} +lat_2={p_beta} +lat_0={y_cent} +lon_0={x_cent} {a_b} +x_0=0 +y_0=0 +units=m +no_defs"
            crs_proj = pyproj.CRS.from_proj4(proj4)
            tf_fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_proj, always_xy=True)
            tf_inv = pyproj.Transformer.from_crs(crs_proj, pyproj.CRS.from_epsg(4326), always_xy=True)
        except Exception:
            crs_proj = None; tf_fwd = None; tf_inv = None
    elif args.force_lcc:
        try:
            a_b = "+a=6370000.0 +b=6370000.0" if USE_SPHERICAL_EARTH else "+ellps=WGS84 +datum=WGS84"
            proj4 = f"+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-96 {a_b} +x_0=0 +y_0=0 +units=m +no_defs"
            crs_proj = pyproj.CRS.from_proj4(proj4)
            tf_fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_proj, always_xy=True)
            tf_inv = pyproj.Transformer.from_crs(crs_proj, pyproj.CRS.from_epsg(4326), always_xy=True)
        except Exception:
            crs_proj = None; tf_fwd = None; tf_inv = None
    
    # If crs_proj is None we stay in geographic (EPSG:4326) space
    # but we initialize identity transformers to allow graticule drawing via _draw_graticule
    if crs_proj is None:
        try:
            crs_proj = pyproj.CRS.from_epsg(4326)
            tf_fwd = pyproj.Transformer.from_crs(crs_proj, crs_proj, always_xy=True)
            tf_inv = pyproj.Transformer.from_crs(crs_proj, crs_proj, always_xy=True)
        except Exception:
            crs_proj = None; tf_fwd = None; tf_inv = None

    try:
        merged_plot = merged.to_crs(crs_proj) if (crs_proj is not None and getattr(merged, 'crs', None) is not None) else merged
    except Exception:
        merged_plot = merged
    try:
        batch_src = emis_df.attrs.get('source_name')
    except Exception:
        batch_src = None
    to_plot = requested
    if not to_plot:
        logging.info("Specify --pollutant in batch mode. Available:")
        for p in pollutants:
            logging.info("  %s", p)
        return _finish(0)
    context: Dict[str, Any] = {
        'args': args,
        'merged': merged,
        'merged_plot': merged_plot,
        'overlay_county': overlay_county,
        'overlay_geom': overlay_geom,
        'crs_proj': crs_proj,
        'tf_fwd': tf_fwd,
        'tf_inv': tf_inv,
        'emis_df': emis_df,
        'batch_src': batch_src,
        'input_basename': input_basename,
    }
    worker_count = 1
    requested_workers = getattr(args, 'workers', 0)
    
    # Auto-detect if workers is 0
    if requested_workers <= 0:
        if _PLOT_MP_CONTEXT is not None and len(to_plot) >= 1:
            # Default to aggressive parallel usage
            worker_count = _resolve_worker_count(len(to_plot))
        else:
            worker_count = 1
    else:
        worker_count = requested_workers
        
    if worker_count <= 1:
        for pol in to_plot:
            _, out_path = _render_single_pollutant(pol, context)
            logging.info("Wrote %s", out_path)
            generated_plots.append(out_path)
    else:
        global _PLOT_CONTEXT
        _PLOT_CONTEXT = context
        executor_kwargs = {'max_workers': worker_count}
        if _PLOT_MP_CONTEXT is not None:
            executor_kwargs['mp_context'] = _PLOT_MP_CONTEXT
        results: Dict[str, str] = {}
        try:
            with ProcessPoolExecutor(**executor_kwargs) as pool:
                futures = {pool.submit(_plot_worker, pol): pol for pol in to_plot}
                for fut in as_completed(futures):
                    pol_name, out_path = fut.result()
                    results[pol_name] = out_path
        finally:
            _PLOT_CONTEXT = None
        for pol in to_plot:
            out_path = results.get(pol)
            if out_path:
                logging.info("Wrote %s", out_path)
                generated_plots.append(out_path)
    return _finish(0)