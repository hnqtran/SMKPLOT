"""Plotting utilities for SMKPLOT GUI."""

import logging
import copy
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
    zoom_to_data: bool = True,
    zoom_pad: float = 0.05,
):
    """
    Shared logic to render an emissions map plot on a given Axes.
    Returns the collection (artist) representing the main data plot.
    """
    
    # 1. Colormap Resolution ---------------------------------------------------
    try:
        if hasattr(plt, 'colormaps'):
            cmap = plt.colormaps.get_cmap(cmap_name)
        else:
             # Fallback for older matplotlib
             cmap = mplcm.get_cmap(cmap_name)
    except Exception:
        try:
            cmap = plt.get_cmap('jet')
        except:
            cmap = mplcm.get_cmap('jet')
    
    # Auto-contrast settings based on 'dark' colormaps
    _cmap_lower = str(cmap_name).lower()
    _dark_cmaps = {'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'turbo', 'nipy_spectral', 'gnuplot', 'gnuplot2'}
    if _cmap_lower in _dark_cmaps:
         county_color = '#aaaaaa' 
         overlay_color = 'cyan'
         county_lw = 0.3
         overlay_lw = 0.8
    else:
         county_color = '#555555'
         overlay_color = 'black'
         county_lw = 0.3
         overlay_lw = 0.8

    # 2. Plotting Arguments (Norm/Bins) ----------------------------------------
    plot_kwargs = dict(
        column=column,
        ax=ax,
        legend=True,  # Default legend (can be customized later)
        cmap=cmap,
        linewidth=0.05,
        edgecolor='black',
        missing_kwds={'color': '#f0f0f0', 'edgecolor': 'none', 'label': 'No Data'},
    )
    
    data = gdf[column]
    
    if bins and len(bins) >= 2:
        try:
            # Update cmap to support transparency for values below first bin
            cmap = copy.copy(cmap)
            cmap.set_under('none')
            try:
                cmap.set_over(cmap(cmap.N - 1))
            except Exception:
                pass
                
            plot_kwargs['cmap'] = cmap
            plot_kwargs['norm'] = BoundaryNorm(bins, ncolors=cmap.N, clip=False, extend='neither')
        except Exception:
            pass
    else:
        use_log = log_scale and (data > 0).any()
        if use_log:
            positive = data[data > 0]
            if not positive.empty:
                plot_kwargs['norm'] = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    
    # 3. Main Plot -------------------------------------------------------------
    try:
        # returns axes usually, but if cbar is off... geopandas returns Axes
        # Use plot directly
        gdf.plot(**plot_kwargs)
    except Exception as e:
        logging.warning(f"Plotting failed: {e}")
        return

    ax.set_title(title, fontsize=10)
    ax.axis('off')

    # 4. Colorbar Labeling -----------------------------------------------------
    if unit_label:
        # Find the colorbar axes attached to this axes
        cb_ax = None
        if ax.figure:
             for cand in ax.figure.axes:
                 if cand is not ax and cand.get_label() == '<colorbar>': # Heuristic
                     cb_ax = cand
                     break
             # Fallback heuristic: check position
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
                    cb_ax.set_ylabel(unit_label)
                else:
                    cb_ax.set_xlabel(unit_label)
            except Exception:
                pass

    # 5. Overlays (Counties / Shapes) ------------------------------------------
    if overlay_counties is not None:
        try:
            if crs_proj is not None and getattr(overlay_counties, 'crs', None) is not None:
                trunc_overlay = overlay_counties.to_crs(crs_proj)
            else:
                trunc_overlay = overlay_counties
            trunc_overlay.boundary.plot(ax=ax, color=county_color, linewidth=county_lw, alpha=0.7)
        except Exception:
            logging.exception("Failed to overlay county shapefile")
            
    if overlay_shape is not None:
        try:
             if crs_proj is not None and getattr(overlay_shape, 'crs', None) is not None:
                 shape_overlay = overlay_shape.to_crs(crs_proj)
             else:
                 shape_overlay = overlay_shape
             shape_overlay.boundary.plot(ax=ax, color=overlay_color, linewidth=overlay_lw, alpha=0.9, linestyle='--')
        except Exception:
             logging.exception("Failed to overlay auxiliary shapefile")

    # 6. Zoom ------------------------------------------------------------------
    if zoom_to_data:
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

def _default_conus_lcc_crs():
    """Return a default CONUS Lambert Conformal Conic CRS and transformers.
    Uses standard parallels 33/45 and center 40N, 96W. Honors USE_SPHERICAL_EARTH.
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

def _lcc_from_griddesc(griddesc_path, grid_name):
    """Build an LCC CRS from the current GRIDDESC selection if available."""
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
    grid name is selected. Otherwise, fall back to WGS84 (EPSG:4326) so maps render in
    geographic coordinates. This improves interpretability for county-only plots or when
    the user has not supplied grid context.
    """
    try:
        # Require explicit grid specification for LCC
        if griddesc_path and grid_name and grid_name not in (None, '', 'Select Grid'):
            lcc = _lcc_from_griddesc(griddesc_path, grid_name)
            if lcc:  # (crs, tf_fwd, tf_inv)
                return lcc
    except Exception:
        pass
    # Fallback: geographic WGS84 with identity transformers
    try:
        crs = pyproj.CRS.from_epsg(4326)
        tf = pyproj.Transformer.from_crs(crs, crs, always_xy=True)
        return crs, tf, tf
    except Exception:
        return None, None, None

def _draw_graticule(ax, tf_fwd: pyproj.Transformer, tf_inv: pyproj.Transformer, lon_step=5, lat_step=5, with_labels=True):
    """Draw longitude/latitude grid lines (in degrees) using the supplied transformers.

    Returns a dictionary with lists of created Line2D and Text artists for optional cleanup/redraw.
    """
    artists = {'lines': [], 'texts': []}
    if ax is None or tf_fwd is None or tf_inv is None:
        return artists
    try:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # Sample corners plus midpoints to estimate the current lon/lat bounds
        sample_x = [x0, x1, x1, x0, 0.5 * (x0 + x1), x1, 0.5 * (x0 + x1), x0]
        sample_y = [y0, y0, y1, y1, y0, 0.5 * (y0 + y1), y1, 0.5 * (y0 + y1)]
        lons = []
        lats = []
        for X, Y in zip(sample_x, sample_y):
            try:
                lon, lat = tf_inv.transform(X, Y)
                if abs(lon) < 1e6 and abs(lat) < 1e6:
                    lons.append(lon)
                    lats.append(lat)
            except Exception:
                pass
        if not lons or not lats:
            return artists
        lon_min = max(-180.0, min(lons)) - 1.0
        lon_max = min(180.0, max(lons)) + 1.0
        lat_min = max(-90.0, min(lats)) - 1.0
        lat_max = min(90.0, max(lats)) + 1.0
        if lon_min >= lon_max or lat_min >= lat_max:
            return artists

        lats_samp = np.linspace(lat_min, lat_max, 200)
        lons_samp = np.linspace(lon_min, lon_max, 200)

        def _lon_label(v: float) -> str:
            v_r = round(v)
            return f"{abs(int(v_r))}°{'W' if v_r < 0 else 'E'}"

        def _lat_label(v: float) -> str:
            v_r = round(v)
            return f"{abs(int(v_r))}°{'S' if v_r < 0 else 'N'}"

        w = x1 - x0
        h = y1 - y0
        pad_y = 0.012 * h
        pad_x = 0.012 * w

        for lon in np.arange(np.floor(lon_min / lon_step) * lon_step, lon_max + lon_step, lon_step):
            try:
                xs = []
                ys = []
                for lat in lats_samp:
                    x, y = tf_fwd.transform(lon, float(lat))
                    xs.append(x)
                    ys.append(y)
                lines = ax.plot(xs, ys, color='#cccccc', linewidth=0.5, alpha=0.8, zorder=10)
                artists['lines'].extend(lines)
                if with_labels:
                    arr_x = np.array(xs)
                    arr_y = np.array(ys)
                    mask = (arr_x >= x0) & (arr_x <= x1)
                    if mask.any():
                        idx = np.argmin(np.abs(arr_y[mask] - y0))
                        x_lab = arr_x[mask][idx]
                        y_lab = y0 + pad_y
                        if x0 <= x_lab <= x1 and y0 <= y_lab <= y1:
                            txt = ax.text(
                                x_lab,
                                y_lab,
                                _lon_label(lon),
                                color='#666666',
                                fontsize=8,
                                ha='center',
                                va='bottom',
                                zorder=11,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5),
                            )
                            artists['texts'].append(txt)
            except Exception:
                pass

        for lat in np.arange(np.floor(lat_min / lat_step) * lat_step, lat_max + lat_step, lat_step):
            try:
                xs = []
                ys = []
                for lon in lons_samp:
                    x, y = tf_fwd.transform(float(lon), lat)
                    xs.append(x)
                    ys.append(y)
                lines = ax.plot(xs, ys, color='#cccccc', linewidth=0.5, alpha=0.8, zorder=10)
                artists['lines'].extend(lines)
                if with_labels:
                    arr_x = np.array(xs)
                    arr_y = np.array(ys)
                    mask = (arr_y >= y0) & (arr_y <= y1)
                    if mask.any():
                        idx = np.argmin(np.abs(arr_x[mask] - x0))
                        y_lab = arr_y[mask][idx]
                        x_lab = x0 + pad_x
                        if x0 <= x_lab <= x1 and y0 <= y_lab <= y1:
                            txt = ax.text(
                                x_lab,
                                y_lab,
                                _lat_label(lat),
                                color='#666666',
                                fontsize=8,
                                ha='left',
                                va='center',
                                zorder=11,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5),
                            )
                            artists['texts'].append(txt)
            except Exception:
                pass
    except Exception:
        pass
    return artists