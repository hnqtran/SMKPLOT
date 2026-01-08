"""
Batch processing module for SMKPLOT.

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
    _normalize_delim,
    country_cols,
    _coerce_merge_key,
    get_emis_fips,
    filter_dataframe_by_range,
    filter_dataframe_by_values
)

from config import USE_SPHERICAL_EARTH
from plotting import _draw_graticule


def _safe_sector_slug(sector: Optional[str]) -> str:
    text = str(sector or 'default')
    slug = ''.join(ch if (ch.isalnum() or ch in {'-', '_'}) else '_' for ch in text)
    slug = slug.strip('_') or 'default'
    return slug


def _serialize_attrs(attrs) -> dict:
    if not isinstance(attrs, dict):
        return {}
    result = {}
    for key, value in attrs.items():
        safe_key = str(key)
        try:
            json.dumps(value)
            result[safe_key] = value
        except TypeError:
            result[safe_key] = repr(value)
    return result


_PLOT_CONTEXT: Optional[Dict[str, Any]] = None
try:
    _PLOT_MP_CONTEXT = multiprocessing.get_context('fork')
except Exception:
    _PLOT_MP_CONTEXT = None


def _resolve_worker_count(num_jobs: int) -> int:
    if num_jobs <= 1:
        return 1 if num_jobs else 0
    cpu = os.cpu_count() or 1
    if cpu <= 1:
        return 1
    return min(num_jobs, max(1, cpu - 1))


def _write_settings_snapshot(args, plots, files, attrs=None):
    """Persist run configuration and generated artifacts for reuse."""
    try:
        os.makedirs(args.outdir, exist_ok=True)
        payload = {
            'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
            'arguments': {
                k: getattr(args, k) for k in vars(args)
                if k not in {'json', 'yaml', 'json_payload', 'pollutant_list', 'pollutant_first','filter_values'}
            },
            'outputs': {
                'output_directory': os.path.abspath(args.outdir),
                'plots': [os.path.abspath(p) for p in plots if p],
                'files': [os.path.abspath(p) for p in files if p],
            }
        }
        if attrs is not None:
            payload['outputs']['emis_df_attrs'] = _serialize_attrs(attrs)
        sector_slug = _safe_sector_slug(getattr(args, 'sector', None))
        
        if getattr(args, 'yaml', None):
            yaml_path = os.path.join(args.outdir, f"{sector_slug}.yaml")
            payload['outputs']['settings_yaml'] = os.path.abspath(yaml_path)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(payload, f, sort_keys=False)
            logging.info("Saved settings snapshot to %s", yaml_path)
        else:
            json_path = os.path.join(args.outdir, f"{sector_slug}.json")
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

    fig, ax = plt.subplots(figsize=(9, 5))
    data = merged[pol]
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
    try:
        cmap_name = args.cmap or 'jet'
        cmap_registry = getattr(matplotlib, 'colormaps', None)
        if cmap_registry is not None:
            cmap = cmap_registry.get_cmap(cmap_name)
        else:
            cmap = mplcm.get_cmap(cmap_name)
    except Exception:
        try:
            cmap = mplcm.get_cmap('jet')
        except Exception:
            cmap = plt.get_cmap('jet')

    bins = None
    if args.bins:
        try:
            parts = [p for p in args.bins.replace(',', ' ').split() if p]
            bins = sorted(set(float(p) for p in parts))
        except Exception:
            bins = None

    plot_kwargs = dict(
        column=pol,
        ax=ax,
        legend=True,
        cmap=cmap,
        linewidth=0.05,
        edgecolor='black',
        missing_kwds={'color': '#f0f0f0', 'edgecolor': 'none', 'label': 'No Data'},
    )

    if bins and len(bins) >= 2:
        try:
            plot_kwargs['norm'] = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
        except Exception:
            pass
        use_log = False
    else:
        use_log = args.log_scale and (data > 0).any()
        if use_log:
            positive = data[data > 0]
            if positive.empty:
                plot_kwargs['norm'] = None
            else:
                plot_kwargs['norm'] = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))

    try:
        merged_plot.plot(**plot_kwargs)
    except Exception:
        merged.plot(**plot_kwargs)

    cb_ax = None
    try:
        for candidate in fig.axes:
            if candidate is not ax:
                cb_ax = candidate
                break
    except Exception:
        cb_ax = None
    if cb_ax is not None and pol_unit:
        try:
            bbox = cb_ax.get_position()
            orient_vertical = (bbox.height >= bbox.width)
        except Exception:
            orient_vertical = True
        try:
            if orient_vertical:
                cb_ax.set_ylabel(pol_unit)
            else:
                cb_ax.set_xlabel(pol_unit)
        except Exception:
            pass

    # Overlay county shapefile boundaries if provided
    if overlay_county is not None:
        try:
            if crs_proj is not None and getattr(overlay_county, 'crs', None) is not None:
                overlay_county = overlay_county.to_crs(crs_proj)
            overlay_county.boundary.plot(ax=ax, color='black', linewidth=0.3, alpha=0.7)
        except Exception:
            logging.exception("Failed to overlay county shapefile boundaries")
    
    # Overlay shapefile boundaries if provided
    if overlay_geom is not None:
        try:
            if crs_proj is not None and getattr(overlay_geom, 'crs', None) is not None:
                overlay_geom = overlay_geom.to_crs(crs_proj)
            overlay_geom.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.7, linestyle='--')
        except Exception:
            logging.exception("Failed to overlay auxiliary shapefile boundaries")
    
    if args.zoom_to_data:
        try:
            valid = merged_plot[merged_plot[pol].notna() & (merged_plot[pol].abs() > 0)]
            if valid.empty:
                fallback = merged_plot[merged_plot[pol].notna()]
                if not fallback.empty:
                    valid = fallback
            if not valid.empty:
                minx, miny, maxx, maxy = valid.total_bounds
                pad = max(0.0, min(0.25, args.zoom_pad))
                dx = (maxx - minx) * pad if maxx > minx else 0.1
                dy = (maxy - miny) * pad if maxy > miny else 0.1
                ax.set_xlim(minx - dx, maxx + dx)
                ax.set_ylim(miny - dy, maxy + dy)
                logging.debug(
                    "Zoom-to-data applied for %s: bounds=(%.4f,%.4f,%.4f,%.4f) pad=%.3f",
                    pol,
                    minx,
                    miny,
                    maxx,
                    maxy,
                    pad,
                )
            else:
                logging.debug("Zoom-to-data skipped for %s: no valid geometries after filtering", pol)
        except Exception:
            logging.exception("Zoom-to-data failed for %s", pol)

    try:
        bins_ticks = bins or []
    except Exception:
        bins_ticks = []
    try:
        from matplotlib.ticker import FixedLocator

        if cb_ax is not None and bins_ticks:
            try:
                _norm = plot_kwargs.get('norm', None)
            except Exception:
                _norm = None
            is_log = isinstance(_norm, LogNorm)
            ticks = [t for t in bins_ticks if (t > 0)] if is_log else list(bins_ticks)
            if ticks:
                try:
                    bbox = cb_ax.get_position()
                    orient_vertical = (bbox.height >= bbox.width)
                except Exception:
                    orient_vertical = True
                if orient_vertical:
                    try:
                        cb_ax.yaxis.set_major_locator(FixedLocator(ticks))
                    except Exception:
                        pass
                else:
                    try:
                        cb_ax.xaxis.set_major_locator(FixedLocator(ticks))
                    except Exception:
                        pass
                try:
                    fig.canvas.draw_idle()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        if crs_proj is not None and tf_fwd is not None and tf_inv is not None:
            _draw_graticule(ax, tf_fwd, tf_inv, lon_step=5, lat_step=5)
            try:
                ax.set_aspect('equal', adjustable='box')
            except Exception:
                pass
    except Exception:
        pass

    ax.set_axis_off()


    stats_source = None
    if args.zoom_to_data:
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

    if stats_source is None:
        stats_source = emis_df[pol] if isinstance(emis_df, pd.DataFrame) and pol in emis_df.columns else data

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
    if not args.zoom_to_data:
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
    fig.tight_layout()

    pltyp_raw = getattr(args, 'pltyp', None)
    if pltyp_raw:
        pltyp_token = ''.join(ch if (ch.isalnum() or ch in {'-', '_'}) else '_' for ch in str(pltyp_raw))
        pltyp_suffix = f"_{pltyp_token.strip('_') or 'plot'}"
    else:
        pltyp_suffix = ''

    if args.filter_col:
        out_file = os.path.join(
            args.outdir,
            f"{input_basename}_{pol}{pltyp_suffix}.filterby_{args.filter_col}.png",
        )
    else:
        out_file = os.path.join(args.outdir, f"{input_basename}_{pol}{pltyp_suffix}.png")

    fig.savefig(out_file, dpi=180)
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

    def _finish(code: int):
        if code == 0:
            _write_settings_snapshot(
                args,
                generated_plots,
                generated_files,
                attrs=emis_attrs,
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
                        valid = merged[merged[pol].fillna(0) != 0]
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
    
    
    # Check for JSON payload in args
    json_payload = getattr(args, 'json_payload', None)

    # Check if reimporting processed data specified in arguments
    reimport_requested = getattr(args, 'reimport', False)
    # Check if reimporting processed data specified in JSON payload
    if not reimport_requested and isinstance(json_payload, dict):
        reimport_requested = bool(json_payload.get('arguments', {}).get('reimport', False))

    emis_df = None
    raw_df = None

    if reimport_requested:
        
        if not isinstance(json_payload, dict):
            logging.error("Reimport requires --json or --yaml to point to a saved settings file.")
            return 2
        
        # Locate preprocessed emission file from JSON arguments
        imported_preprocessed_path = json_payload.get('arguments', {}).get('imported_file', None)
        
        # Normalize to list
        if isinstance(imported_preprocessed_path, (str, type(None))):
             imported_paths = [imported_preprocessed_path] if imported_preprocessed_path else []
        else:
             imported_paths = list(imported_preprocessed_path)

        resolved_paths = []
        json_dir = os.path.dirname(args.json) if getattr(args, 'json', None) else None

        for path in imported_paths:
            if path and not os.path.isabs(path):
                if json_dir:
                    path = os.path.abspath(os.path.join(json_dir, path))
                else:
                    path = os.path.abspath(path)
            
            if path and not os.path.exists(path):
                logging.error("Specified imported file %s does not exist locally.", path)
                return 1
            if path:
                resolved_paths.append(path)

        if not resolved_paths:
            logging.error("Reimport requested but no preprocessed files listed in JSON outputs. Aborting.")
            return 1
        
        # All good now, read the preprocessed emissions data
        try:
            dfs = []
            for p in resolved_paths:
                dfs.append(pd.read_csv(p))
            emis_df = pd.concat(dfs, ignore_index=True)

            # Apply filtering BEFORE aggregation to ensure we don't lose granularity
            filter_col = args.filter_col
            filter_start = args.filter_start
            filter_end = args.filter_end
            filter_values = args.filter_values

            if filter_col and (filter_start is not None) and (filter_end is not None):
                try:
                    emis_df = filter_dataframe_by_range(
                        emis_df,
                        filter_col,
                        float(filter_start),
                        float(filter_end)
                    )
                    logging.info(
                        "Applied reimport filtering on column %s for range %.4g to %.4g",
                        filter_col,
                        float(filter_start),
                        float(filter_end)
                    )
                except Exception:
                    logging.exception("Failed to apply reimport filtering by range on column %s", filter_col)
            
            if filter_col and (filter_values is not None):
                try:
                    emis_df = filter_dataframe_by_values(
                        emis_df,
                        filter_col,
                        filter_values
                    )
                    logging.info(
                        "Applied reimport filtering on column %s for values %s",
                        filter_col,
                        repr(filter_values)
                    )
                except Exception:
                    logging.exception("Failed to apply reimport filtering by values on column %s", filter_col)

            # Ensure FIPS column exists for aggregation
            try:
                emis_df = get_emis_fips(emis_df)
            except Exception:
                pass

            # Aggregation logic
            group_keys = []
            if 'FIPS' in emis_df.columns:
                group_keys.append('FIPS')
            elif 'GRID_RC' in emis_df.columns:
                group_keys.append('GRID_RC')
            
            if 'country_cd' in emis_df.columns:
                group_keys.append('country_cd')
            
            if group_keys:
                # Clear cached pollutants from attrs if any
                if hasattr(emis_df, 'attrs'):
                     emis_df.attrs.pop('_detected_pollutants', None)
                
                pols = detect_pollutants(emis_df)
                if pols:
                    agg_dict = {p: 'sum' for p in pols}
                    # Handle other columns (take first)
                    for c in emis_df.columns:
                        if c not in group_keys and c not in pols:
                            agg_dict[c] = 'first'
                    
                    emis_df = emis_df.groupby(group_keys, as_index=False).agg(agg_dict)
                    logging.info("Aggregated reimported data by %s", group_keys)

        except Exception:
            logging.exception("Failed to reimport processed emissions data from %s", resolved_paths)
            return 1
        
        # Restore DataFrame attributes
        attrs_from_json = json_payload.get('arguments', {}).get('imported_attrs', None)
        if isinstance(attrs_from_json, dict):
            for key, value in attrs_from_json.items():
                emis_df.attrs[key] = value
        
        logging.info("Reimported processed emissions data from %s", resolved_paths)

        raw_df = emis_df.copy(deep=True)
        
        # Apply filtering if specified in JSON arguments
        # (Filtering has already been applied before aggregation above)
        
    else: # Not reimporting; read raw input file
        if not args.filepath:
            logging.error("Batch mode requires --filepath to specify the SMKREPORT / FF10 input file.")
            return 2
        
        try:
            norm_delim = _normalize_delim(args.delim)
            emis_df, raw_df = read_inputfile(
                fpath=args.filepath,
                sector=args.sector,
                delim=norm_delim,
                skiprows=args.skiprows,
                comment=args.comment,
                encoding=args.encoding,
                flter_col=args.filter_col,
                flter_start=args.filter_start,
                flter_end=args.filter_end,
                flter_val=getattr(args, 'filter_values', None),
            )
        except Exception as e:
            if "No valid FIPS code columns found" in str(e):
                logging.error("Input file is not supported: No valid FIPS/Region columns found.")
                return 1
            logging.exception("Error reading emissions")
            return 1
    
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

    print(emis_df.attrs)
    print(emis_df.head(5))
    
    # Capture emissions DataFrame attributes for later use
    emis_attrs = dict(getattr(emis_df, 'attrs', {}) or {})


    # Update input_basename; use sector if specified in arguments or in JSON payload
    start_sector = getattr(args, 'sector', None)
    json_sector = None
    if json_payload:
        json_sector = json_payload.get('arguments', {}).get('sector', None)
    
    input_basename = start_sector or json_sector or os.path.basename(args.filepath)


    # Export processed emissions data to CSV if requested in arguments or JSON payload
    csv_from_json = False
    if isinstance(json_payload, dict):
        csv_from_json = json_payload.get('arguments', {}).get('export_csv', False)

    if (getattr(args, 'export_csv', False) or csv_from_json):
        # Create output directory if it doesn't exist
        os.makedirs(args.outdir, exist_ok=True)
        if args.filter_col:
            csv_outpath = os.path.join(args.outdir, f"{input_basename}.filterby_{args.filter_col}.pivotted.csv")
        else:
            csv_outpath = os.path.join(args.outdir, f"{input_basename}.pivotted.csv")
        # export raw_df to CSV
        if reimport_requested:
            emis_df.to_csv(csv_outpath, index=False)
        else:
            raw_df.to_csv(csv_outpath, index=False)

        logging.info("Processed emissions data exported to %s", csv_outpath)
        generated_files.append(os.path.abspath(csv_outpath))        
    
    # Validate inputs based on plot type
    overlay_county = None
    overlay_geom = None

    if args.pltyp == 'grid':
        if not args.griddesc or not args.gridname:
            logging.error("Batch grid plotting (--pltyp grid) requires --griddesc and --gridname.")
            return 2
    if args.pltyp == 'county':
        if not args.county_shapefile:
            logging.error("Batch county plotting (--pltyp county) requires --county-shapefile.")
            return 2

    # Load geometry based on plot type
    if args.pltyp == 'grid':
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
    if args.overlay_shapefile:
        try:
            overlay_geom = read_shpfile(args.overlay_shapefile, False)
        except Exception:
            logging.exception(f"Failed to load {args.overlay_shapefile} for plotting. Ignoring overlay.")
            overlay_geom = None

    # Detect pollutants
    pollutants = detect_pollutants(emis_df)
    #print(f'List of pollutants: {pollutants}')
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
    try:
        attrs = dict(getattr(emis_df, 'attrs', {}))
        try:
            grouped = emis_df.groupby(merge_on, dropna=False, sort=False, observed=False)
        except TypeError:
            grouped = emis_df.groupby(merge_on, sort=False, observed=False)
        if requested:
            emis_df = grouped[requested].sum(numeric_only=True).reset_index()
        else:
            emis_df = grouped.sum(numeric_only=True).reset_index()
        emis_df.attrs = attrs
    except Exception:
        logging.exception("Unexpected error while aggregating emissions data")

    if merge_on == 'FIPS':
        try:
            if 'FIPS' in emis_df.columns:
                emis_df = emis_df.copy()
                emis_df['FIPS'] = _coerce_merge_key(emis_df['FIPS'], pad=6)
        except Exception:
            logging.exception("Failed to normalize emissions FIPS prior to merge")
        try:
            if 'FIPS' in base_geom.columns:
                base_geom = base_geom.copy()
                base_geom['FIPS'] = _coerce_merge_key(base_geom['FIPS'], pad=6)
        except Exception:
            logging.exception("Failed to normalize geometry FIPS prior to merge")

    try:
        merged = base_geom.merge(emis_df, on=merge_on, how='left', sort=False)
    except Exception:
        logging.exception("Failed to merge emissions with geometry")
        return 1
    
    # Export merged GeoDataFrame to file if requested
    #if (getattr(args, 'export_csv', False) or json_payload.get('arguments', {}).get('export_csv', False)):
    #    if args.filter_col:
    #        csv_outpath = os.path.join(args.outdir, f"{input_basename}.merged.filterby_{args.filter_col}.pivotted.csv")
    #    else:
    #        csv_outpath = os.path.join(args.outdir, f"{input_basename}.merged.pivotted.csv")
    #    # export merged to CSV
    #    merged.to_csv(csv_outpath, index=False)
    #    logging.info("Processed emissions data merged with geometry and exported to %s", csv_outpath)
    #    generated_files.append(os.path.abspath(csv_outpath))  

    # Determine projection (only use LCC when grid + GRIDDESC provided, unless --force-lcc)
    crs_proj = None; tf_fwd = None; tf_inv = None
    if args.projection == 'wgs84':
        crs_proj = None  # keep geographic
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
    elif args.griddesc and args.gridname:
        # For auto mode: if gridname provided, use grid projection
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
    try:
        merged_plot = merged.to_crs(crs_proj) if (crs_proj is not None and getattr(merged, 'crs', None) is not None) else merged
    except Exception:
        merged_plot = merged
    try:
        batch_src = emis_df.attrs.get('source_name')
    except Exception:
        batch_src = None
    to_plot = pollutants if args.batch_all else requested
    if not to_plot:
        logging.info("Specify --pollutant or --batch-all in batch mode. Available:")
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
    if requested_workers > 0:
        worker_count = requested_workers
    elif _PLOT_MP_CONTEXT is not None and len(to_plot) > 1:
        resolved = _resolve_worker_count(len(to_plot))
        if resolved > 1:
            worker_count = resolved
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