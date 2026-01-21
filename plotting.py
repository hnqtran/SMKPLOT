"""Plotting utilities for SMKPLOT GUI."""

import logging
import copy
import functools
import numpy as np

import pandas as pd
import geopandas as gpd
import pyproj
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.figure import Figure
from typing import Optional, List, Dict, Any, Tuple

from config import USE_SPHERICAL_EARTH


# Module-level constant for dark colormaps to avoid recreation
_DARK_CMAPS = {
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
    'jet', 'turbo', 'nipy_spectral', 'gnuplot', 'gnuplot2'
}


def _resolve_cmap_and_theme(cmap_input: Any) -> Tuple[Any, Dict[str, Any]]:
    """Determine colormap and associated styling theme. Accepts name (str) or Colormap object."""
    cmap = None
    if isinstance(cmap_input, str):
        try:
            if hasattr(plt, 'colormaps'):
                cmap = plt.colormaps.get_cmap(cmap_input)
            else:
                cmap = mplcm.get_cmap(cmap_input)
        except (AttributeError, ValueError, ImportError):
            pass
    else:
        # Assume it's a colormap object
        cmap = cmap_input

    if cmap is None:
        try:
            cmap = plt.get_cmap('jet')
        except (AttributeError, ValueError):
            cmap = mplcm.get_cmap('jet')

    # Determine theme from name
    c_name = getattr(cmap, 'name', str(cmap_input))
    _cmap_lower = str(c_name).lower()

    if _cmap_lower in _DARK_CMAPS:
        theme = {
            'county_color': '#333333',
            'overlay_color': 'cyan',
            'county_lw': 0.4,
            'overlay_lw': 0.8
        }
    else:
        theme = {
            'county_color': '#000000',
            'overlay_color': 'black',
            'county_lw': 0.4,
            'overlay_lw': 0.8
        }
    return cmap, theme


def _get_plot_kwargs(gdf, column, cmap, bins, log_scale, user_kwargs=None) -> Dict[str, Any]:
    """Construct argument dictionary for geopandas plot function."""
    kwargs = dict(
        column=column,
        legend=True,
        cmap=cmap,
        linewidth=0,
        edgecolor='none',
        missing_kwds={'color': '#f0f0f0', 'edgecolor': 'none', 'label': 'No Data'},
    )
    if user_kwargs:
        kwargs.update(user_kwargs)

    if bins and len(bins) >= 2:
        try:
            # Update cmap to support transparency for values below first bin
            cmap = copy.copy(cmap)
            cmap.set_under('none')
            try:
                cmap.set_over(cmap(cmap.N - 1))
            except Exception:
                pass
                
            kwargs['cmap'] = cmap
            kwargs['norm'] = BoundaryNorm(bins, ncolors=cmap.N, clip=False, extend='neither')
            kwargs['legend_kwds'] = {'ticks': bins, 'format': '%.10g'} 
        except (ValueError, AttributeError) as e:
            logging.warning("Failed to configure custom bins/norm: %s", e)
    else:
        data = gdf[column]
        use_log = log_scale and (data > 0).any()
        if use_log:
            positive = data[data > 0]
            if not positive.empty:
                kwargs['norm'] = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
                
    return kwargs


def _label_colorbar(ax: plt.Axes, label: str):
    """Attempt to find and label the colorbar associated with the axes."""
    if not label:
        return
        
    cb_ax = None
    if ax.figure:
        for cand in ax.figure.axes:
            if cand is not ax and cand.get_label() == '<colorbar>':
                cb_ax = cand
                break
        if not cb_ax:
            for cand in ax.figure.axes:
                if cand is not ax:
                    cb_ax = cand
                    break
                     
    if cb_ax:
        try:
            bbox = cb_ax.get_position()
            orient_vertical = (bbox.height >= bbox.width)
            if orient_vertical:
                cb_ax.set_ylabel(label)
            else:
                cb_ax.set_xlabel(label)
        except (AttributeError, ValueError):
            logging.debug("Could not set label on colorbar axis.")


def _add_overlays(ax: plt.Axes, overlay_counties, overlay_shape, crs_proj, theme):
    """Draw optional boundary layers."""
    if overlay_counties is not None:
        try:
            if crs_proj is not None and getattr(overlay_counties, 'crs', None) is not None:
                trunc_overlay = overlay_counties.to_crs(crs_proj)
            else:
                trunc_overlay = overlay_counties
            trunc_overlay.boundary.plot(
                ax=ax, 
                color=theme['county_color'], 
                linewidth=theme['county_lw'], 
                alpha=0.7
            )
        except Exception as e:
            logging.warning("Failed to overlay county shapefile: %s", e)
            
    if overlay_shape is not None:
        # Handle both single GeoDataFrame and list of GeoDataFrames
        overlay_list = overlay_shape if isinstance(overlay_shape, list) else [overlay_shape]
        
        # Color cycle for multiple overlays
        colors = ['cyan', 'magenta', 'yellow', 'red', 'lime', 'orange']
        
        for idx, shape_gdf in enumerate(overlay_list):
            try:
                if crs_proj is not None and getattr(shape_gdf, 'crs', None) is not None:
                    shape_overlay = shape_gdf.to_crs(crs_proj)
                else:
                    shape_overlay = shape_gdf
                
                # Use different color for each overlay
                color = colors[idx % len(colors)]
                
                shape_overlay.boundary.plot(
                    ax=ax, 
                    color=color, 
                    linewidth=theme['overlay_lw'], 
                    alpha=0.9, 
                    linestyle='--'
                )
            except Exception as e:
                logging.warning("Failed to overlay auxiliary shapefile %d: %s", idx, e)


def _zoom_to_extent(ax: plt.Axes, gdf, column: str, zoom_pad: float):
    """Zoom map extent to valid data bounds."""
    try:
        valid = gdf[gdf[column].notna()]
        if not valid.empty:
            minx, miny, maxx, maxy = valid.total_bounds
            pad = max(0.0, min(0.25, zoom_pad))
            dx = (maxx - minx) * pad if maxx > minx else 0.1
            dy = (maxy - miny) * pad if maxy > miny else 0.1
            ax.set_xlim(minx - dx, maxx + dx)
            ax.set_ylim(miny - dy, maxy + dy)
    except Exception:
        pass


def create_map_plot(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    ax: plt.Axes,
    cmap_name: str = 'jet',
    bins: Optional[List[float]] = None,
    log_scale: bool = False,
    unit_label: Optional[str] = None,
    overlay_counties: Optional[gpd.GeoDataFrame] = None,
    overlay_shape: Optional[gpd.GeoDataFrame] = None,
    crs_proj=None,
    tf_fwd=None,
    tf_inv=None,
    zoom_to_data: bool = True,
    zoom_pad: float = 0.05,
    **kwargs
):
    """
    Shared logic to render an emissions map plot on a given Axes.
    Returns the collection (artist) representing the main data plot.
    """
    # 1. Resolve Style
    cmap, theme = _resolve_cmap_and_theme(cmap_name)

    # 2. Build Arguments
    plot_kwargs = _get_plot_kwargs(gdf, column, cmap, bins, log_scale, user_kwargs=kwargs)
    plot_kwargs['ax'] = ax  # Ensure ax is passed explicitly

    # 3. Plot Data
    collection = None
    try:
        # GeoPandas plot returns the axes, but assumes collection is added
        gdf.plot(**plot_kwargs)
        if ax.collections:
            collection = ax.collections[-1]
    except Exception as e:
        logging.error("Map plotting failed for column '%s': %s", column, e)
        return None

    ax.set_title(title, fontsize=10)
    
    # Enable border (spines) but hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')

    # 4. Label Colorbar
    _label_colorbar(ax, unit_label)

    # 5. Overlays
    _add_overlays(ax, overlay_counties, overlay_shape, crs_proj, theme)

    # 6. Graticule


    # 7. Zoom
    if zoom_to_data:
        _zoom_to_extent(ax, gdf, column, zoom_pad)
        
    # 6. Graticule (Draw AFTER zoom to ensure grid matches visible extent)
    if crs_proj is not None and tf_fwd and tf_inv:
        _draw_graticule(ax, tf_fwd, tf_inv, lon_step=None, lat_step=None)
    return collection


@functools.lru_cache(maxsize=1)
def _default_conus_lcc_crs():
    """Return a default CONUS Lambert Conformal Conic CRS and transformers.
    Uses standard parallels 33/45 and center 40N, 96W. Honors USE_SPHERICAL_EARTH.
    Cached to avoid re-initializing transformers.
    """
    try:
        a_b = "+a=6370000.0 +b=6370000.0" if USE_SPHERICAL_EARTH else "+ellps=WGS84 +datum=WGS84"
        proj4 = (
            f"+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-96 {a_b} +x_0=0 +y_0=0 +units=m +no_defs"
        )
        crs_lcc = pyproj.CRS.from_proj4(proj4)
        tf_fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_lcc, always_xy=True)
        tf_inv = pyproj.Transformer.from_crs(crs_lcc, pyproj.CRS.from_epsg(4326), always_xy=True)
        return crs_lcc, tf_fwd, tf_inv
    except Exception:
        # Fallback to WGS84 if something goes wrong
        c = pyproj.CRS.from_epsg(4326)
        tf = pyproj.Transformer.from_crs(c, c, always_xy=True)
        return c, tf, tf

@functools.lru_cache(maxsize=32)
def _lcc_from_griddesc(griddesc_path, grid_name):
    """Build an LCC CRS from the current GRIDDESC selection if available.
    Cached to avoid repeated file I/O on GRIDDESC.
    """
    try:
        from .data_processing import extract_grid
        coord_params, _ = extract_grid(griddesc_path, grid_name)
        # coord_params: proj_type, p_alpha, p_beta, p_gamma, x_cent, y_cent
        _, p_alpha, p_beta, _p_gamma, x_cent, y_cent = coord_params
        a_b = "+a=6370000.0 +b=6370000.0" if USE_SPHERICAL_EARTH else "+ellps=WGS84 +datum=WGS84"
        proj4 = (
            f"+proj=lcc +lat_1={p_alpha} +lat_2={p_beta} +lat_0={y_cent} +lon_0={x_cent} {a_b} +x_0=0 +y_0=0 +units=m +no_defs"
        )
        crs_lcc = pyproj.CRS.from_proj4(proj4)
        tf_fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_lcc, always_xy=True)
        tf_inv = pyproj.Transformer.from_crs(crs_lcc, pyproj.CRS.from_epsg(4326), always_xy=True)
        return crs_lcc, tf_fwd, tf_inv
    except Exception:
        return None

def _plot_crs(griddesc_path, grid_name):
    """Return (crs, forward_transformer, inverse_transformer) for plotting.

    New behavior: ONLY use an LCC projection when a GRIDDESC file is provided AND a valid
    grid name is selected. Otherwise, fall back to WGS84 (EPSG:4326) so maps render        geographic coordinates. This ensures layout stability when multiple
        grids are not specified.
    """
    try:
        # Require explicit grid specification for LCC
        if griddesc_path and grid_name and grid_name not in (None, '', 'Select Grid'):
            lcc = _lcc_from_griddesc(griddesc_path, grid_name)
            if lcc:  # (crs, tf_fwd, tf_inv)
                return lcc
    except ImportError:
        logging.debug("Data Processing utilities not available for LCC grid extraction.")
    except Exception as e:
        logging.warning("Failed to derive LCC from GRIDDESC (%s/%s): %s", griddesc_path, grid_name, e)

    # Fallback: geographic WGS84 with identity transformers
    try:
        crs = pyproj.CRS.from_epsg(4326)
        tf = pyproj.Transformer.from_crs(crs, crs, always_xy=True)
        return crs, tf, tf
    except Exception:
        return None, None, None

def _draw_graticule(ax, tf_fwd: pyproj.Transformer, tf_inv: pyproj.Transformer, lon_step=None, lat_step=None, with_labels=True):
    """Draw longitude/latitude grid lines (in degrees) using the supplied transformers.

    Optimized to use vectorized pyproj transformations.
    Returns a dictionary with lists of created Line2D and Text artists for optional cleanup/redraw.
    """
    artists = {'lines': [], 'texts': []}
    if ax is None or tf_fwd is None or tf_inv is None:
        return artists
    try:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()


        # 1. Estimate Lat/Lon bounds from view extent
        # Vectorized inverse transform on boundary points
        sx = np.array([x0, x1, x1, x0, 0.5 * (x0 + x1), x1, 0.5 * (x0 + x1), x0])
        sy = np.array([y0, y0, y1, y1, y0, 0.5 * (y0 + y1), y1, 0.5 * (y0 + y1)])

        try:
            sample_lons, sample_lats = tf_inv.transform(sx, sy)
            # Filter out potentially infinite/failed transforms
            valid_mask = (np.abs(sample_lons) < 1e6) & (np.abs(sample_lats) < 1e6)
            lons = sample_lons[valid_mask]
            lats = sample_lats[valid_mask]
        except Exception:
            return artists

        if lons.size == 0 or lats.size == 0:
            return artists

        lon_min = max(-180.0, lons.min()) - 1.0
        lon_max = min(180.0, lons.max()) + 1.0
        lat_min = max(-90.0, lats.min()) - 1.0
        lat_max = min(90.0, lats.max()) + 1.0

        if lon_min >= lon_max or lat_min >= lat_max:
            return artists
            
        # Determine appropriate step size if not fixed
        # Logic: aim for ~4-8 lines in the view
        
        def _nice_step(span):
            if span <= 0: return 1.0
            # Rough target step
            target = span / 5.0
            # Snap to 1, 5, 10 or 0.1, 0.5 etc.
            power = 10 ** np.floor(np.log10(target))
            base = target / power
            # nice bases: 1, 2, 5
            if base < 1.5: step = 1.0 * power
            elif base < 3.5: step = 2.0 * power
            else: step = 5.0 * power
            return max(step, 1e-5) # Safety floor

        lon_span = lons.max() - lons.min()
        lat_span = lats.max() - lats.min()
        
        # Use provided step unless it's too sparse for the current view (zoom handling)
        actual_lon_step = lon_step
        if actual_lon_step is None or (lon_step > lon_span and lon_span > 0):
             actual_lon_step = _nice_step(lon_span)
             
        actual_lat_step = lat_step
        if actual_lat_step is None or (lat_step > lat_span and lat_span > 0):
             actual_lat_step = _nice_step(lat_span)

        # Pre-calculate precise sampling arrays
        lats_samp = np.linspace(lat_min, lat_max, 200)
        lons_samp = np.linspace(lon_min, lon_max, 200)

        # Helpers for labels
        def _fmt_deg(v, step):
            prec = 0
            if step < 1: prec = 1
            if step < 0.1: prec = 2
            if step < 0.01: prec = 3
            if step < 0.001: prec = 4
            
            val = abs(v)
            if prec == 0:
                s = f"{int(round(val))}"
            else:
                s = f"{val:.{prec}f}"
            return s

        def _lon_label(v):
            s = _fmt_deg(v, actual_lon_step)
            return f"{s}°{'W' if v < 0 else 'E'}"
            
        def _lat_label(v):
            s = _fmt_deg(v, actual_lat_step)
            return f"{s}°{'S' if v < 0 else 'N'}"

        pad_y = 0.012 * (y1 - y0)
        pad_x = 0.012 * (x1 - x0)

        # 2. Draw Longitude Lines (constant Lon, varying Lat)
        # Determine start/end aligned to step
        l_start = np.floor(lon_min / actual_lon_step) * actual_lon_step
        l_end = lon_max + actual_lon_step
        lon_range = np.arange(l_start, l_end, actual_lon_step)
        
        for lon in lon_range:
            try:
                # Vectorized transform: one lon repeated, many lats
                xs, ys = tf_fwd.transform(np.full_like(lats_samp, lon), lats_samp)

                # Plot line
                lines = ax.plot(xs, ys, color='#cccccc', linewidth=0.5, alpha=0.8, zorder=10)
                artists['lines'].extend(lines)

                # Labeling
                if with_labels:
                    mask = (xs >= x0) & (xs <= x1)
                    if np.any(mask):

                        # Find point closest to bottom edge y0
                        valid_ys = ys[mask]
                        valid_xs = xs[mask]
                        idx_b = np.argmin(np.abs(valid_ys - y0))

                        x_b = valid_xs[idx_b]
                        y_b = y0 + pad_y

                        if x0 <= x_b <= x1 and y0 <= y_b <= y1:
                            txt = ax.text(
                                x_b, y_b, _lon_label(lon),
                                color='#333333', fontsize=8,
                                ha='center', va='bottom', zorder=12,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3)
                            )
                            artists['texts'].append(txt)


            except Exception as e:
                logging.warning(f"Graticule Lon Error: {e}")

        # 3. Draw Latitude Lines (constant Lat, varying Lon)
        l_start = np.floor(lat_min / actual_lat_step) * actual_lat_step
        l_end = lat_max + actual_lat_step
        lat_range = np.arange(l_start, l_end, actual_lat_step)
        
        for lat in lat_range:
            try:
                # Vectorized transform: many lons, one lat repeated
                xs, ys = tf_fwd.transform(lons_samp, np.full_like(lons_samp, lat))

                # Plot line
                lines = ax.plot(xs, ys, color='#cccccc', linewidth=0.5, alpha=0.8, zorder=10)
                artists['lines'].extend(lines)

                # Labeling
                if with_labels:
                    mask = (ys >= y0) & (ys <= y1)
                    if np.any(mask):
                        # Find point closest to left edge x0
                        valid_xs = xs[mask]
                        valid_ys = ys[mask]
                        idx = np.argmin(np.abs(valid_xs - x0))

                        x_lab = x0 + pad_x
                        y_lab = valid_ys[idx]

                        if x0 <= x_lab <= x1 and y0 <= y_lab <= y1:
                            txt = ax.text(
                                x_lab, y_lab, _lat_label(lat),
                                color='#666666', fontsize=8,
                                ha='left', va='center', zorder=11,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5)
                            )
                            artists['texts'].append(txt)


            except Exception as e:
                logging.warning(f"Graticule Line Error: {e}")

    except Exception as e:
        logging.warning("Graticule drawing failed: %s", e)

    return artists