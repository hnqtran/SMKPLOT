"""
GUI components for SMKPLOT.

This module implements the Tkinter-based graphical user interface for the tool.
Features include:
- Interactive selection of input files (Emissions, Shapefiles, GRIDDESC).
- Asynchronous data loading and processing to prevent UI freezing.
- Dynamic filtering of data by sector, SCC, or other columns.
- Interactive map plotting with customizable settings (colormap, scale, bins).
- Export capabilities for plots and data.
"""

import os
import sys
import logging
import io
import threading
from functools import lru_cache
from typing import Optional, List, Dict, Any, Union, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm as mplcm
from matplotlib.colors import BoundaryNorm
import pyproj
from shapely.geometry import Polygon

from config import (
    US_STATE_FIPS_TO_NAME, DEFAULT_INPUTS_INITIALDIR, DEFAULT_SHPFILE_INITIALDIR,
    DEFAULT_ONLINE_COUNTIES_URL, load_settings, save_settings
)
from data_processing import (
    _normalize_delim, read_inputfile, read_shpfile, extract_grid, create_domain_gdf, detect_pollutants, map_latlon2grd,
    merge_emissions_with_geometry, filter_dataframe_by_range, filter_dataframe_by_values, get_emis_fips
)
from plotting import _plot_crs, _draw_graticule

# Backend selection: try Tk if DISPLAY exists, otherwise Agg
_display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
try:
    if _display:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

USING_TK = matplotlib.get_backend().lower().startswith('tk')
if USING_TK:
    try:
        import tkinter as tk  # type: ignore
        from tkinter import ttk, filedialog  # type: ignore
    except Exception:
        USING_TK = False
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # re-import safe
else:
    # Provide stubs so type checker / runtime does not fail before GUI branch
    tk = None  # type: ignore
    ttk = None  # type: ignore
    filedialog = None  # type: ignore
class EmissionGUI:
    def __init__(self, root, inputfile_path: Optional[str], counties_path: Optional[str], emissions_delim: Optional[str] = None, *, cli_args=None):
        self.root = root
        self.root.title("SMKREPORT Emission Plotter (Author: tranhuy@email.unc.edu)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # UI scaling based on screen resolution (baseline ~1600px width)
        try:
            sw = max(1, int(self.root.winfo_screenwidth()))
        except Exception:
            sw = 1600
        try:
            self._ui_scale = max(0.85, min(1.6, sw / 1600.0))
        except Exception:
            self._ui_scale = 1.0
        # helpers for scalable widget sizes (character widths)
        def _w_chars(n: int, min_chars: int = 10, max_chars: int = 120) -> int:
            try:
                return int(max(min_chars, min(max_chars, round(n * self._ui_scale))))
            except Exception:
                return n
        self._w_chars = _w_chars  # store for use in layout

        self.cli_args = cli_args
        self.json_payload = getattr(cli_args, 'json_payload', None) if cli_args else None
        self.json_path = getattr(cli_args, 'json', None) if cli_args else None
        self._json_arguments = self.json_payload.get('arguments', {}) if isinstance(self.json_payload, dict) else {}
        
        # Priotize arguments from CLI args before JSON snapshot        
        self.emissions_delim = getattr(cli_args, 'delim', None) if cli_args else None or self._json_arguments.get('delim', None)
        self.emissions_delim = _normalize_delim(self.emissions_delim)
        self.sector = getattr(cli_args, 'sector', None) if cli_args else None or self._json_arguments.get('sector', None)
        self.skiprows = getattr(cli_args, 'skiprows', None) if cli_args else None or self._json_arguments.get('skiprows', None)
        self.comment_token = getattr(cli_args, 'comment', None) if cli_args else None or self._json_arguments.get('comment', None)
        self.encoding = getattr(cli_args, 'encoding', None) if cli_args else None or self._json_arguments.get('encoding', None)

        self.filter_col = getattr(cli_args, 'filter_col', None) if cli_args else None or self._json_arguments.get('filter_col', None)
        self.filter_start = getattr(cli_args, 'filter_start', None) if cli_args else None or self._json_arguments.get('filter_start', None)
        self.filter_end = getattr(cli_args, 'filter_end', None) if cli_args else None or self._json_arguments.get('filter_end', None)
        
        raw_filter_vals = getattr(cli_args, 'filter_val', None) if cli_args else None or self._json_arguments.get('filter_val', None)
        if isinstance(raw_filter_vals, (list, tuple, set)):
            self.filter_values = [str(v) for v in raw_filter_vals if v is not None]
        elif raw_filter_vals is None:
            self.filter_values = None
        else:
            self.filter_values = [str(raw_filter_vals)]
        if not self.filter_values:
            self.filter_values = None

        # GUI state for delimiter selection (auto/comma/tab/pipe/space/other)
        self.delim_var: Optional[tk.StringVar] = None
        self.custom_delim_var: Optional[tk.StringVar] = None
        self._last_loaded_delim_state: Optional[Tuple[str, str, str]] = None
        self._merged_cache: Dict[
            Tuple,
            Tuple[gpd.GeoDataFrame, Optional[pd.DataFrame], Optional[str]],
        ] = {}
        self.emissions_df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.counties_gdf: Optional[gpd.GeoDataFrame] = None
        self.overlay_gdf: Optional[gpd.GeoDataFrame] = None
        self.grid_gdf: Optional[gpd.GeoDataFrame] = None
        self.pollutants: List[str] = []
        self.units_map: Dict[str, str] = {}
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        self.cbar_ax = None  # track colorbar axis for cleanup
        self._preview_win = None  # popup preview window
        # Interactive zoom state
        self._zoom_rect = None
        self._zoom_press = None
        self._zoom_cids = []
        # Base view (xlim, ylim) to support Reset View behavior
        self._base_view = None
        # Custom scale bar state
        self.scale_var: Optional[tk.StringVar] = None
        # Custom bins (used as colorbar ticks) state
        self.class_bins_var: Optional[tk.StringVar] = None
        # Plot-by mode: auto, county, grid
        self.plot_by_var: Optional[tk.StringVar] = None
        # SCC selection dropdown (enabled only when SCC/SCC Description exist)
        self.scc_keyword_var: Optional[tk.StringVar] = None  # legacy (unused for filtering now)
        self.scc_select_var: Optional[tk.StringVar] = None
        self._scc_display_to_code: Dict[str, str] = {}
        self._has_scc_cols: bool = False
        if USING_TK:
            try:
                self.status_var = tk.StringVar(value='Ready.')
            except Exception:
                self.status_var = None
        else:
            self.status_var = None
        self.status_label = None
        self._last_attrs_summary = ''
        self._loader_messages: List[Tuple[str, str]] = []
        self._preview_has_data = False
        self._source_type: Optional[str] = None
        self._ff10_ready = False
        self._ff10_grid_ready = False
        self._build_layout()

        # Load default settings under .config/smkgui_settings.json and apply them if needed
        settings = load_settings()
        last_paths = settings.get('last_paths', {})
        ui_state = settings.get('ui_state', {})

        # Restore UI state first

        # Setting for plot scale (linear/log)
        if self._json_arguments.get('log_scale', None) is True:
            self.scale_var.set('log')
        elif self._json_arguments.get('log_scale', None) is None:
            self.scale_var.set('linear')
        else:
            self.scale_var.set(ui_state.get('scale', 'linear'))
        
        # Settings for bins var
        if self._json_arguments.get('bins', None):
            self.class_bins_var.set(str(self._json_arguments.get('bins')))
        else:
            self.class_bins_var.set(ui_state.get('bins', ''))
            
        # Settings for cmaps
        if self._json_arguments.get('cmap', None):
            self.cmap_var.set(str(self._json_arguments.get('cmap')))
        else:
            self.cmap_var.set(ui_state.get('colormap', 'viridis'))

        # Setting for delimiter
        if self._json_arguments.get('delim', None):
            self.delim_var.set(str(self._json_arguments.get('delim')))
        else:
            self.delim_var.set(ui_state.get('delimiter', ','))            
            
        if self.custom_delim_var and ui_state.get('custom_delimiter'): self.custom_delim_var.set(ui_state['custom_delimiter'])
        if self.plot_by_var and ui_state.get('plot_by'): self.plot_by_var.set(ui_state['plot_by'])
        # Projection selection (added)
        if hasattr(self, 'projection_var') and ui_state.get('projection'):
                try:
                    self.projection_var.set(ui_state['projection'])
                except Exception:
                    pass
        if self.zoom_var and 'zoom_to_data' in ui_state: self.zoom_var.set(ui_state['zoom_to_data'])
        
        #self._on_delim_change() # Ensure custom delim entry visibility is correct

        # Restore last inputpath path but defer loading until user requests it
        self.inputfile_path = inputfile_path or getattr(cli_args, 'filepath', None) if cli_args else None or self._json_arguments.get('filepath', None) or last_paths.get('inputpath')
        self.reimport_requested = bool(getattr(cli_args, 'reimport', False)) if cli_args else self._json_arguments.get('reimport', False)
        if self.reimport_requested:
            cli_imported = getattr(cli_args, 'imported_file', None) if cli_args else None
            raw_imported = cli_imported or self._json_arguments.get('imported_file')
            
            if isinstance(raw_imported, (list, tuple)):
                self.imported_paths = [str(p) for p in raw_imported if p]
            elif raw_imported:
                self.imported_paths = [str(raw_imported)]
            else:
                self.imported_paths = []

            if not self.imported_paths:
                raise ValueError(f'Reimport specified but no imported_file provided. Aborting.')

            for p in self.imported_paths:
                if not os.path.exists(p):
                    raise ValueError(f'Reimport specified but could not locate imported_file {p}. Aborting.')
            
            self.inputfile_path = self.imported_paths[0]

        if self.inputfile_path and os.path.exists(self.inputfile_path):
            try:
                self.emis_entry.delete(0, tk.END)
                self.emis_entry.insert(0, self.inputfile_path)
            except Exception:
                pass
            
            if self.json_payload:
                self._set_status('Auto-loading data from JSON/YAML configuration...', level='INFO')
                # Schedule load to run after UI init
                self.root.after(200, lambda: self.load_inputfile(show_preview=False))
            else:
                self._set_status('Recovered previous input path. Click Preview Data to load.', level='INFO')

        # Restore and auto-load GRIDDESC path
        self.griddesc_path  = getattr(cli_args, 'griddesc', None) if cli_args else None or self._json_arguments.get('griddesc', None) or last_paths.get('griddesc') or None
        if self.griddesc_path and os.path.exists(self.griddesc_path):
            try:
                self.griddesc_entry.delete(0, tk.END)
                self.griddesc_entry.insert(0, self.griddesc_path)
            except Exception:
                pass
            self.load_griddesc()

            # Restore last selected grid name if available
            if self.grid_name_var and ui_state.get('grid_name'):
                # Check if the saved grid name is in the list of available grids
                available_grids = self.grid_name_menu['menu'].winfo_children()
                available_names = [item.cget('label') for item in available_grids]
                if ui_state['grid_name'] in available_names:
                    self.grid_name_var.set(ui_state['grid_name'])
                    self.load_grid_shape()

        # Restore last SCC selection
        if self.scc_select_var and ui_state.get('scc_selection'):
            self.scc_select_var.set(ui_state['scc_selection'])

        # Restore and auto-load Counties shapefile path (precedence: CLI arg > saved last path > online default)
        self.counties_path  = counties_path or getattr(cli_args, 'county_shapefile', None) if cli_args else None or self._json_arguments.get('county_shapefile', None) or last_paths.get('counties') or None
        if self.counties_path and os.path.exists(self.counties_path):
            self.load_shpfile()
            try:
                self.county_entry.delete(0, tk.END)
                self.county_entry.insert(0, self.counties_path)
            except Exception:
                    pass
        else:
            # Fall back to online counties
            try:
                self.use_online_counties()
            except Exception as e:
                self._notify('WARNING', 'Online Counties Not Loaded', f'Could not load default online counties: {e}')

        # Load optional overlay shapefile
        self.overlay_path  = getattr(cli_args, 'overlay_shapefile', None) if cli_args else None or self._json_arguments.get('overlay_shapefile', None)
        if self.overlay_path and os.path.exists(self.overlay_path):
            self.load_overlay()
        else:
            # Fall back to online counties
            try:
                self.use_online_counties()
            except Exception as e:
                self._notify('WARNING', 'Online Counties Not Loaded', f'Could not load default online counties: {e}')

        

        #self._preferred_grid_name = getattr(cli_args, 'gridname', None) if cli_args else None or self._json_arguments.get('gridname', None)
        #self._preferred_overlay_path = getattr(cli_args, 'overlay_shapefile', None) if cli_args else None or self._json_arguments.get('overlay_shapefile', None)
        #self._apply_json_snapshot_defaults()

    # ---- Projection helpers (LCC + graticule) ----
    def _default_conus_lcc_crs(self):
        """Return a default CONUS Lambert Conformal Conic CRS and transformers.
        Uses standard parallels 33/45 and center 40N, 96W. Honors USE_SPHERICAL_EARTH.
        """
        try:
            from config import USE_SPHERICAL_EARTH
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

    def _lcc_from_griddesc(self):
        """Build an LCC CRS from the current GRIDDESC selection if available."""
        try:
            if not self.griddesc_path:
                return None
            grid_name = self.grid_name_var.get() if getattr(self, 'grid_name_var', None) else None
            if not grid_name or grid_name == 'Select Grid':
                return None
            from .data_processing import extract_grid
            coord_params, _ = extract_grid(self.griddesc_path, grid_name)
            # coord_params: proj_type, p_alpha, p_beta, p_gamma, x_cent, y_cent
            _, p_alpha, p_beta, _p_gamma, x_cent, y_cent = coord_params
            from config import USE_SPHERICAL_EARTH
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

    def _current_lcc_crs(self):
        """Select an LCC CRS using GRIDDESC when available, else a CONUS default."""
        """(Deprecated) Retained for backward compatibility; now only used internally if needed."""
        return self._lcc_from_griddesc() or self._default_conus_lcc_crs()

    def _plot_crs(self):
        """Return (crs, forward_transformer, inverse_transformer) for plotting.

        New behavior: ONLY use an LCC projection when a GRIDDESC file is provided AND a valid
        grid name is selected. Otherwise, fall back to WGS84 (EPSG:4326) so maps render in
        geographic coordinates. This improves interpretability for county-only plots or when
        the user has not supplied grid context.
        """
        choice = None
        try:
            choice = (self.projection_var.get() if getattr(self, 'projection_var', None) else 'auto')
        except Exception:
            choice = 'auto'
        choice = (choice or 'auto').strip().lower()

        if choice == 'wgs84':
            try:
                crs = pyproj.CRS.from_epsg(4326)
                tf = pyproj.Transformer.from_crs(crs, crs, always_xy=True)
                return crs, tf, tf
            except Exception:
                return None, None, None

        if choice == 'lcc':
            candidate = self._lcc_from_griddesc() or self._default_conus_lcc_crs()
            if candidate:
                return candidate

        try:
            # Require explicit grid specification for LCC
            if self.griddesc_path and getattr(self, 'grid_name_var', None):
                gname = self.grid_name_var.get()
                if gname and gname not in (None, '', 'Select Grid'):
                    lcc = self._lcc_from_griddesc()
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

    def _draw_graticule(self, ax, tf_fwd: pyproj.Transformer, tf_inv: pyproj.Transformer, lon_step=5, lat_step=5, with_labels=True):
        """Draw longitude/latitude lines (in degrees) on the given axes using provided transformers.
        The axes must be in the projected (LCC) coordinates. Optionally label lines along edges.
        """
        from .plotting import _draw_graticule as _draw_helper

        if ax is None:
            return _draw_helper(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)

        def _remove_existing():
            artists = getattr(ax, '_smk_graticule_artists', None)
            if isinstance(artists, dict):
                for key in ('lines', 'texts'):
                    try:
                        group = artists.get(key, [])
                    except Exception:
                        group = []
                    for art in group or []:
                        try:
                            art.remove()
                        except Exception:
                            pass
            ax._smk_graticule_artists = {'lines': [], 'texts': []}  # type: ignore[attr-defined]

        def _disconnect():
            cids = getattr(ax, '_smk_graticule_cids', [])
            for cid in cids or []:
                try:
                    ax.callbacks.disconnect(cid)
                except Exception:
                    pass
            ax._smk_graticule_cids = []  # type: ignore[attr-defined]
            ax._smk_graticule_callback = None  # type: ignore[attr-defined]

        try:
            _remove_existing()
        except Exception:
            pass
        try:
            _disconnect()
        except Exception:
            pass

        artists = _draw_helper(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)
        if not isinstance(artists, dict):
            artists = {'lines': [], 'texts': []}
        ax._smk_graticule_artists = artists  # type: ignore[attr-defined]

        def _on_limits(_axis):
            try:
                _remove_existing()
                refreshed = _draw_helper(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)
                if not isinstance(refreshed, dict):
                    refreshed = {'lines': [], 'texts': []}
                ax._smk_graticule_artists = refreshed  # type: ignore[attr-defined]
                try:
                    ax.figure.canvas.draw_idle()
                except Exception:
                    pass
            except Exception:
                pass

        try:
            cid1 = ax.callbacks.connect('xlim_changed', _on_limits)
        except Exception:
            cid1 = None
        try:
            cid2 = ax.callbacks.connect('ylim_changed', _on_limits)
        except Exception:
            cid2 = None
        ax._smk_graticule_cids = [cid for cid in (cid1, cid2) if cid is not None]  # type: ignore[attr-defined]
        ax._smk_graticule_callback = _on_limits  # type: ignore[attr-defined]
        return artists

    # ---- Simple config persistence for last-used paths ----
    def _config_file(self) -> str:
        cfg_dir = os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('./'), '.config')
        try:
            os.makedirs(cfg_dir, exist_ok=True)
        except Exception:
            pass
        return os.path.join(cfg_dir, 'smkgui_settings.json')

    def _load_settings(self) -> dict:
        """Load the entire settings dictionary from the JSON config file."""
        return load_settings()

    def _save_settings(self) -> None:
        """Collect and save current GUI settings to the JSON config file."""
        settings = {
            'last_paths': {
                'inputpath': self.inputfile_path or '',
                'griddesc': self.griddesc_path or '',
                'counties': self.counties_path or '',
            },
            'ui_state': {
                'scale': self.scale_var.get() if self.scale_var else 'linear',
                'bins': self.class_bins_var.get() if self.class_bins_var else '',
                'colormap': self.cmap_var.get() if self.cmap_var else 'jet',
                'delimiter': self.delim_var.get() if self.delim_var else 'auto',
                'custom_delimiter': self.custom_delim_var.get() if self.custom_delim_var else '',
                'plot_by': self.plot_by_var.get() if self.plot_by_var else 'auto',
                'projection': self.projection_var.get() if hasattr(self, 'projection_var') and self.projection_var else 'auto',
                'grid_name': self.grid_name_var.get() if self.grid_name_var else '',
                'scc_selection': self.scc_select_var.get() if self.scc_select_var else 'All SCC',
                'zoom_to_data': self.zoom_var.get() if self.zoom_var else True,
            }
        }
        save_settings(settings)

    def _set_status(self, message: str, level: Optional[str] = None) -> None:
        """Update the GUI status bar, collapsing whitespace and capping length."""
        if not USING_TK:
            return
        try:
            if self.status_var is None:
                self.status_var = tk.StringVar(master=self.root, value='')
            prefix = f"{level.upper()}: " if level else ''
            collapsed = ' '.join(message.strip().split()) if message else ''
            self.status_var.set((prefix + collapsed)[:512])
            if self.status_label is not None:
                self.status_label.update_idletasks()
        except Exception:
            pass

    def _format_attrs_for_display(self, source=None, *, max_units: int = 12, max_chars: int = 4000) -> str:
        """Return a human-friendly summary of DataFrame attrs for GUI display."""
        attrs: Dict[str, object] = {}
        if source is None and isinstance(self.emissions_df, pd.DataFrame):
            try:
                attrs = dict(getattr(self.emissions_df, 'attrs', {}))
            except Exception:
                attrs = {}
        elif isinstance(source, pd.DataFrame):
            try:
                attrs = dict(getattr(source, 'attrs', {}))
            except Exception:
                attrs = {}
        elif isinstance(source, dict):
            attrs = dict(source)
        if not attrs:
            return ''
        lines: List[str] = []
        for key, value in attrs.items():
            if key == 'units_map' and isinstance(value, dict):
                pairs: List[str] = []
                for idx, (k, v) in enumerate(value.items()):
                    if idx >= max_units:
                        pairs.append('...')
                        break
                    pairs.append(f"{k}: {v}")
                lines.append(f"{key}: " + ', '.join(pairs))
            elif isinstance(value, dict):
                snippets: List[str] = []
                for idx, (k, v) in enumerate(value.items()):
                    if idx >= max_units:
                        snippets.append('...')
                        break
                    snippets.append(f"{k}: {v}")
                lines.append(f"{key}: " + ', '.join(snippets))
            elif isinstance(value, (list, tuple)) and len(value) > max_units:
                subset = ', '.join(str(v) for v in value[:max_units])
                lines.append(f"{key}: {subset}, ... ({len(value)} items)")
            else:
                lines.append(f"{key}: {value}")
        text = '\n'.join(lines)
        if len(text) > max_chars:
            text = text[:max_chars - 3] + '...'
        return text

    def _loader_notify(self, level: str, message: str) -> None:
        lvl = (level or 'INFO').upper()
        msg = ' '.join(str(message).strip().split()) if message else ''
        if not msg:
            return
        self._loader_messages.append((lvl, msg))
        if len(self._loader_messages) > 10:
            self._loader_messages = self._loader_messages[-10:]
        self._set_status(msg, level=lvl)
        if USING_TK and getattr(self, 'preview', None) is not None and not getattr(self, '_preview_has_data', False):
            self._render_loader_messages()

    def _render_loader_messages(self) -> None:
        if not USING_TK or getattr(self, 'preview', None) is None:
            return
        try:
            self.preview.delete('1.0', tk.END)
            if not self._loader_messages:
                self.preview.insert(tk.END, "Load Messages\n(no recent messages)\n")
            else:
                self.preview.insert(tk.END, "Load Messages\n")
                for lvl, msg in self._loader_messages:
                    self.preview.insert(tk.END, f"[{lvl}] {msg}\n")
            self.preview.see(tk.END)
            self._preview_has_data = False
        except Exception:
            pass

    def _append_loader_messages_to_preview(self) -> None:
        if (not USING_TK) or getattr(self, 'preview', None) is None or not self._loader_messages:
            return
        try:
            self.preview.insert(tk.END, "\n\nLoad Messages\n")
            for lvl, msg in self._loader_messages:
                self.preview.insert(tk.END, f"[{lvl}] {msg}\n")
            self.preview.see(tk.END)
        except Exception:
            pass

    def _on_close(self):
        """Harden shutdown: save settings, close plots, and force exit."""
        try:
            self._save_settings()
        except Exception:
            pass
        try:
            # Close all matplotlib figures to release memory and resources
            plt.close('all')
        except Exception:
            pass
        try:
            # Explicitly destroy the main window
            self.root.destroy()
        except Exception:
            pass
        # Force exit to ensure the process terminates, especially if non-daemon threads are running.
        # This helps return control to the terminal promptly.
        sys.exit(0)

    # ---- Unified notification / logging helper ----
    def _notify(self, level: str, title: str, message: str, exc: Optional[Exception] = None, *, popup: bool = True):
        """Log the message and show a GUI dialog (if Tk available). Levels: INFO, WARNING, ERROR.
        If an exception object is supplied, logs stack trace at ERROR level."""
        lvl = level.upper()
        if exc is not None and lvl == 'ERROR':
            logging.error("%s: %s", title, message, exc_info=exc)
        else:
            log_fn = getattr(logging, lvl.lower(), logging.info)
            log_fn("%s: %s", title, message)
        summary = message.strip().splitlines()[0] if message else ''
        if summary:
            self._set_status(summary, level=lvl)
        # Show dialog if Tk is active
        if popup and USING_TK:
            try:
                from tkinter import messagebox
                if lvl == 'INFO':
                    messagebox.showinfo(title, message)
                elif lvl == 'WARNING':
                    messagebox.showwarning(title, message)
                elif lvl == 'ERROR':
                    messagebox.showerror(title, message)
            except Exception:
                pass  # Suppress any UI errors after logging

    def _build_layout(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.grid(row=0, column=0, sticky='nsew')
        # keep a reference to the main frame for later configuration
        self.frm = frm
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # SMOKE report + delimiter controls
        ttk.Label(frm, text="SMOKE Report / FF10 Input:").grid(row=0, column=0, sticky='w')
        self.emis_entry = ttk.Entry(frm, width=self._w_chars(60))
        self.emis_entry.grid(row=0, column=1, sticky='we')
        ttk.Button(frm, text="Browse", command=self.browse_inputfile).grid(row=0, column=2, padx=4)
        # Delimiter state (widgets will be placed in button row)
        self.delim_var = tk.StringVar()
        self.custom_delim_var = tk.StringVar()
        # Determine initial delimiter token from provided CLI arg
        def _initial_delim_token(raw: Optional[str]) -> str:
            if not raw:
                return 'auto'
            raw_norm = _normalize_delim(raw)
            if raw_norm == ',' : return 'comma'
            if raw_norm == ';' : return 'semicolon'
            if raw_norm == '\t' : return 'tab'
            if raw_norm == '|' : return 'pipe'
            if raw_norm == ' ' : return 'space'
            # treat as custom
            self.custom_delim_var.set(raw_norm)
            return 'other'
        self.delim_var.set(_initial_delim_token(self.emissions_delim))

        # Counties file
        ttk.Label(frm, text="Counties Shapefile / Zip / URL:").grid(row=1, column=0, sticky='w')
        self.county_entry = ttk.Entry(frm, width=self._w_chars(60))
        self.county_entry.grid(row=1, column=1, sticky='we')
        ttk.Button(frm, text="Browse", command=self.browse_shpfile).grid(row=1, column=2, padx=4)
        # Online year selector and button
        self.counties_year_var = tk.StringVar(value='2020')
        ttk.OptionMenu(frm, self.counties_year_var, '2020', '2020', '2023', command=lambda *_: self.use_online_counties()).grid(row=1, column=3, sticky='we')

        # GRIDDESC file
        ttk.Label(frm, text="GRIDDESC File (optional):").grid(row=2, column=0, sticky='w')
        self.griddesc_entry = ttk.Entry(frm, width=self._w_chars(60))
        self.griddesc_entry.grid(row=2, column=1, sticky='we')
        ttk.Button(frm, text="Browse", command=self.browse_griddesc).grid(row=2, column=2, padx=4)

        # Grid Name selector
        self.grid_name_var = tk.StringVar()
        self.grid_name_menu = ttk.OptionMenu(frm, self.grid_name_var, "Select Grid", command=lambda *_: self.load_grid_shape())
        self.grid_name_menu.grid(row=2, column=3, sticky='we')

        # Pollutant selector
        ttk.Label(frm, text="Pollutant:").grid(row=3, column=0, sticky='w')
        self.pollutant_var = tk.StringVar()
        self.pollutant_menu = ttk.OptionMenu(frm, self.pollutant_var, None)
        self.pollutant_menu.grid(row=3, column=1, sticky='we')

        # (Scale and Zoom moved to button row to avoid horizontal expansion)
        # Custom bins state (widgets placed later in button row)
        self.class_bins_var = tk.StringVar(value='')
        # Colormap selector state (widget placed later in button row)
        self.cmap_var = tk.StringVar(value='jet')
        cmap_choices = [
            'jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo',
            'Blues','Greens','Reds','Purples','Oranges','Greys',
            'YlGn','YlGnBu','YlOrRd','OrRd','PuBuGn'
        ]
        self._cmap_choices = cmap_choices

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=4, column=0, columnspan=4, pady=6, sticky='we')
        # Column stretching: let the bins entry grow
        for c in (11,):
            try:
                btn_frame.columnconfigure(c, weight=1)
            except Exception:
                pass
        self.plot_btn = ttk.Button(btn_frame, text="Plot", command=self.plot)
        self.plot_btn.grid(row=0, column=0, padx=4, sticky='w')
        ttk.Button(btn_frame, text="Preview Data", command=self.preview_data).grid(row=0, column=1, padx=8, sticky='w')
        # Place Delim controls
        self.delim_label = ttk.Label(btn_frame, text="Delim:")
        self.delim_label.grid(row=0, column=2, sticky='e', padx=(12,2))
        self.delim_menu_widget = ttk.OptionMenu(btn_frame, self.delim_var, self.delim_var.get(), 'auto','comma','semicolon','tab','pipe','space','other', command=lambda *_: self._on_delim_change())
        self.delim_menu_widget.grid(row=0, column=3, sticky='w')
        self.custom_delim_entry = ttk.Entry(btn_frame, width=self._w_chars(6, min_chars=4, max_chars=10), textvariable=self.custom_delim_var)
        self.custom_delim_entry.grid(row=0, column=4, sticky='w', padx=(4,0))
        # Hide initially unless other selected
        if self.delim_var.get() != 'other':
            self.custom_delim_entry.grid_remove()
        # Bind events to reload if custom delim changes (focus out / return)
        def _reload_event(*_):
            if not self.inputfile_path:
                return
            current_state = self._current_delimiter_state()
            if current_state == self._last_loaded_delim_state:
                return
            self.load_inputfile(show_preview=True)
        self.custom_delim_entry.bind('<FocusOut>', _reload_event)
        self.custom_delim_entry.bind('<Return>', _reload_event)
        # Zoom to Data next to Delim
        self.zoom_var = tk.BooleanVar(value=True)
        self.zoom_check = ttk.Checkbutton(btn_frame, text='Zoom to Data', variable=self.zoom_var)
        self.zoom_check.grid(row=0, column=5, sticky='w', padx=(12,0))
        # Scale next to Zoom to Data and before Bins
        self.scale_label = ttk.Label(btn_frame, text="Scale:")
        self.scale_label.grid(row=0, column=6, sticky='e', padx=(12,2))
        self.scale_var = tk.StringVar(value='linear')
        self.scale_menu_widget = ttk.OptionMenu(btn_frame, self.scale_var, 'linear', 'linear', 'log')
        self.scale_menu_widget.grid(row=0, column=7, sticky='w')
        # Plot-by control (Auto/County/Grid)
        self.plotby_label = ttk.Label(btn_frame, text='Plot by:')
        self.plotby_label.grid(row=0, column=8, sticky='e', padx=(12,2))
        self.plot_by_var = tk.StringVar(value='auto')
        self.plotby_menu_widget = ttk.OptionMenu(btn_frame, self.plot_by_var, 'auto', 'auto', 'county', 'grid')
        self.plotby_menu_widget.grid(row=0, column=9, sticky='w')
        # Projection selection (Auto / WGS84 / LCC)
        self.proj_label = ttk.Label(btn_frame, text='Proj:')
        self.proj_label.grid(row=0, column=10, sticky='e', padx=(12,2))
        self.projection_var = tk.StringVar(value='auto')
        self.proj_menu_widget = ttk.OptionMenu(btn_frame, self.projection_var, 'auto', 'auto', 'wgs84', 'lcc')
        self.proj_menu_widget.grid(row=0, column=11, sticky='w')
        # Bins and Colormap controls
        self.bins_label = ttk.Label(btn_frame, text='Bins:')
        self.bins_label.grid(row=0, column=12, sticky='e', padx=(12,2))
        self.bins_entry = ttk.Entry(btn_frame, width=self._w_chars(28, min_chars=18, max_chars=40), textvariable=self.class_bins_var)
        self.bins_entry.grid(row=0, column=13, sticky='we')
        self.cmap_label = ttk.Label(btn_frame, text='Colormap:')
        self.cmap_label.grid(row=0, column=14, sticky='e', padx=(12,2))
        self.cmap_menu_widget = ttk.OptionMenu(btn_frame, self.cmap_var, self.cmap_var.get(), *self._cmap_choices)
        self.cmap_menu_widget.grid(row=0, column=15, sticky='w')

        # SCC selection dropdown (replaces free-text keyword)
        self.scc_select_var = tk.StringVar(value='All SCC')
        self.scc_label = ttk.Label(btn_frame, text='SCC:')
        # Wider combobox to show long SCC descriptions
        self.scc_entry = ttk.Combobox(btn_frame, textvariable=self.scc_select_var, values=[], width=self._w_chars(80, min_chars=40, max_chars=120), state='disabled')
        # place after colormap
        self.scc_label.grid(row=0, column=16, sticky='e', padx=(12,2))
        self.scc_entry.grid(row=0, column=17, sticky='w')
        # initially disabled until we detect SCC columns
        try:
            self.scc_entry.state(['disabled'])
            self.scc_label.state(['disabled'])
        except Exception:
            pass

        # Responsive layout: reflow Bins/Colormap/SCC onto second row when window is narrow
        self._btn_frame = btn_frame
        self._layout_mode = None  # 'wide' or 'compact'
        try:
            self.root.bind('<Configure>', self._on_resize)
        except Exception:
            pass

        # Text preview widget (persistent)
        self.preview = tk.Text(frm, height=10, width=90)
        self.preview.grid(row=5, column=0, columnspan=4, pady=4, sticky='nsew')

        # Embedded plot frame (row 6) - map will display here (pop-out windows draw their own figs)
        self.plot_frame = ttk.Frame(frm)
        self.plot_frame.grid(row=6, column=0, columnspan=4, sticky='nsew', pady=(4, 0))

        if USING_TK:
            try:
                if self.status_var is None:
                    self.status_var = tk.StringVar(master=self.root, value='Ready.')
            except Exception:
                self.status_var = None
            if self.status_var is not None:
                self.status_label = ttk.Label(frm, textvariable=self.status_var, relief='sunken', anchor='w')
                self.status_label.grid(row=7, column=0, columnspan=4, sticky='we', pady=(4, 0))

        # Configure stretch rows/columns for the main frame
        try:
            frm.rowconfigure(5, weight=1)   # text preview
            frm.rowconfigure(6, weight=3)   # plot area larger
            frm.rowconfigure(7, weight=0)
            frm.columnconfigure(1, weight=1)
        except Exception:
            pass

    def _configure_button_layout(self, mode: str):
        # Adjust grid positions depending on mode
        if mode == 'wide':
            try:
                self._btn_frame.columnconfigure(13, weight=1)
                for c in (1, 17):
                    self._btn_frame.columnconfigure(c, weight=0)
            except Exception:
                pass
            # Put all on row 0
            for widget, col in [
                (self.proj_label, 10), (self.proj_menu_widget, 11),
                (self.bins_label, 12), (self.bins_entry, 13), (self.cmap_label, 14), (self.cmap_menu_widget, 15),
                (self.scc_label, 16), (self.scc_entry, 17)
            ]:
                try:
                    widget.grid_configure(row=0, column=col)
                except Exception:
                    pass
        else:  # compact
            try:
                # Make bins entry and scc entry stretch on compact row
                self._btn_frame.columnconfigure(3, weight=1)
                self._btn_frame.columnconfigure(7, weight=1)
                # Reset weights for columns used in wide mode
                for c in (13, 17):
                    self._btn_frame.columnconfigure(c, weight=0)
            except Exception:
                pass
            # Move Bins/Colormap/SCC to row 1
            mapping = [
                (self.proj_label, 0), (self.proj_menu_widget, 1),
                (self.bins_label, 2), (self.bins_entry, 3),
                (self.cmap_label, 4), (self.cmap_menu_widget, 5),
                (self.scc_label, 6), (self.scc_entry, 7),
            ]
            for widget, col in mapping:
                try:
                    # Allow both entry widgets to stretch horizontally
                    sticky_val = 'we' if widget in (self.bins_entry, self.scc_entry) else 'w'
                    widget.grid_configure(row=1, column=col, sticky=sticky_val)
                except Exception:
                    pass

    def _on_resize(self, event=None):
        try:
            w = int(self.root.winfo_width())
        except Exception:
            w = 1400
        # Choose a breakpoint around 1440px (scaled)
        try:
            threshold = int(1440 * getattr(self, '_ui_scale', 1.0))
        except Exception:
            threshold = 1440
        mode = 'compact' if w < threshold else 'wide'
        if mode != self._layout_mode:
            self._configure_button_layout(mode)
            self._layout_mode = mode
        # No widget creation here; just reflow handled by _configure_button_layout

    def browse_inputfile(self):
        # Prefer directory of current/last SMKREPORT/FF10 if available
        init_dir = None
        try:
            # Load fresh settings to get the most recent path
            settings = load_settings()
            last_smk = (settings.get('last_paths') or {}).get('inputpath')
            cand = self.inputfile_path or last_smk
            if cand and os.path.exists(cand):
                init = os.path.dirname(cand)
                if os.path.isdir(init):
                    init_dir = init
        except Exception:
            init_dir = None
        if init_dir is None:
            init_dir = DEFAULT_INPUTS_INITIALDIR if os.path.isdir(DEFAULT_INPUTS_INITIALDIR) else None
        path = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("CSV/List", "*.csv *.txt *.lst"), ("All", "*.*")])
        if not path:
            return
        self.emis_entry.delete(0, tk.END)
        self.emis_entry.insert(0, path)
        self.inputfile_path = path
        self.load_inputfile(show_preview=True)

    def browse_griddesc(self):
        path = filedialog.askopenfilename(filetypes=[("All files", "*.*")])
        if not path:
            return
        self.griddesc_entry.delete(0, tk.END)
        self.griddesc_entry.insert(0, path)
        self.griddesc_path = path
        self.load_griddesc()

    def browse_shpfile(self):
        init_dir = DEFAULT_SHPFILE_INITIALDIR if os.path.isdir(DEFAULT_SHPFILE_INITIALDIR) else None
        path = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("Shapefile / Geopackage / Zip", "*.shp *.gpkg *.zip"), ("All", "*.*")])
        if not path:
            return
        self.county_entry.delete(0, tk.END)
        self.county_entry.insert(0, path)
        self.counties_path = path
        self.load_shpfile()

    def use_online_counties(self):
        """Set and load the default online US counties shapefile. User selection can override later."""
        try:
            year = self.counties_year_var.get()
        except Exception:
            year = '2020'
        from config import _online_counties_url
        url = _online_counties_url(year) if year else DEFAULT_ONLINE_COUNTIES_URL
        self.counties_path = url
        try:
            self.county_entry.delete(0, tk.END)
            self.county_entry.insert(0, self.counties_path)
        except Exception:
            pass
        self.load_shpfile()

    def _current_delimiter(self) -> Optional[str]:
        """Return the effective delimiter based on GUI selections.
        'auto' returns None so pandas does its normal parsing (or uses provided default).
        """
        if not self.delim_var:
            return self.emissions_delim  # fallback (pre-GUI call)
        token = self.delim_var.get()
        if token == 'auto':
            return None
        mapping = {
            'comma': ',',
            'semicolon': ';',
            'tab': '\t',
            'pipe': '|',
            'space': ' ',
        }
        if token == 'other':
            val = self.custom_delim_var.get() if self.custom_delim_var else ''
            return val or None
        return mapping.get(token, None)

    def _current_delimiter_state(self) -> Tuple[str, str, str]:
        if not self.delim_var:
            fallback = self.emissions_delim if self.emissions_delim is not None else '__AUTO__'
            return ('', '', fallback)
        token = self.delim_var.get() or ''
        custom = self.custom_delim_var.get() if self.custom_delim_var else ''
        effective = self._current_delimiter()
        effective_repr = effective if effective is not None else '__AUTO__'
        return (token, custom or '', effective_repr)

    def _invalidate_merge_cache(self) -> None:
        try:
            self._merged_cache.clear()
        except Exception:
            self._merged_cache = {}

    def _on_delim_change(self):
        # Show or hide custom entry
        if self.delim_var.get() == 'other':
            self.custom_delim_entry.grid()
            self.custom_delim_entry.focus_set()
        else:
            self.custom_delim_entry.grid_remove()
        # Reload emissions automatically if a file is present
        if self.inputfile_path:
            current_state = self._current_delimiter_state()
            if current_state != self._last_loaded_delim_state:
                self.load_inputfile(show_preview=True)

    def _parse_bins(self) -> List[float]:
        """Parse custom bins from the GUI field (comma or space separated)."""
        raw = (self.class_bins_var.get() if self.class_bins_var else '').strip()
        if not raw:
            return []
        try:
            parts = [p for p in raw.replace(',', ' ').split() if p]
            vals = sorted(set(float(p) for p in parts))
            return vals
        except Exception:
            self._notify('WARNING', 'Bins Parse', 'Could not parse custom bins; expected numbers separated by comma or space.')
            return []

    def _apply_json_snapshot_defaults(self) -> None:
        
        print(self._json_arguments)
        
        if not self._json_arguments:
            return

        

    def _finalize_loaded_emissions(self, *, show_preview: bool, source_label: Optional[str] = None, scc_data=None) -> None:
        if not isinstance(self.emissions_df, pd.DataFrame):
            self._notify('ERROR', 'Invalid Data', 'Emissions dataset is not a DataFrame; cannot continue.')
            return

        self._invalidate_merge_cache()

        try:
            self._source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type')
        except Exception:
            self._source_type = None
        self._ff10_ready = bool(self._source_type == 'ff10_point')

        try:
            if self.inputfile_path:
                self._save_settings()
        except Exception:
            pass

        if self._ff10_ready and self.grid_gdf is not None:
            self._ensure_ff10_grid_mapping()

        self._update_scc_widgets(scc_data=scc_data)

        self.pollutants = detect_pollutants(self.emissions_df)
        try:
            self.units_map = dict(self.emissions_df.attrs.get('units_map', {}))
        except Exception:
            self.units_map = {}
        if not self.pollutants:
            self._notify('WARNING', 'No Pollutants', 'No pollutant columns detected.')
            return

        menu = self.pollutant_menu["menu"]
        try:
            menu.delete(0, "end")
        except Exception:
            pass
        PER_COLUMN = 25
        for idx, p in enumerate(self.pollutants):
            kw = {}
            if idx > 0 and (idx % PER_COLUMN) == 0:
                kw['columnbreak'] = 1
            try:
                menu.add_command(label=p, command=lambda v=p: self.pollutant_var.set(v), **kw)
            except Exception:
                pass
        try:
            self.pollutant_var.set(self.pollutants[0])
        except Exception:
            pass

        attrs_summary = self._format_attrs_for_display(self.emissions_df)
        self._last_attrs_summary = attrs_summary

        summary_parts: List[str] = []
        label = source_label
        if label is None and self.inputfile_path:
            label = os.path.basename(self.inputfile_path)
        if label:
            summary_parts.append(label)
        try:
            summary_parts.append(f"{len(self.emissions_df):,} rows")
        except Exception:
            pass
        try:
            summary_parts.append(f"{len(self.emissions_df.columns)} columns")
        except Exception:
            pass
        try:
            if self.pollutants:
                summary_parts.append(f"{len(self.pollutants)} pollutants")
        except Exception:
            pass
        summary_text = ' • '.join(part for part in summary_parts if part)
        if summary_text:
            self._set_status(f"Loaded {summary_text}", level='INFO')

        if attrs_summary and USING_TK and not show_preview:
            try:
                self.preview.delete('1.0', tk.END)
                self.preview.insert(tk.END, "Attributes\n")
                self.preview.insert(tk.END, attrs_summary)
                self._append_loader_messages_to_preview()
                self._preview_has_data = True
            except Exception:
                pass

        if show_preview:
            self.preview_data()

    def load_inputfile(self, show_preview: bool = True):
        try:
            self._loader_messages.clear()
        except Exception:
            self._loader_messages = []
        self._preview_has_data = False
        self._ff10_grid_ready = False

        # Capture UI state for thread
        try:
            effective_delim = self._current_delimiter()
            current_delim_state = self._current_delimiter_state()
        except Exception:
            effective_delim = None
            current_delim_state = None

        self._set_status("Loading data...", level="INFO")
        
        threading.Thread(
            target=self._load_inputfile_worker, 
            args=(show_preview, effective_delim, current_delim_state), 
            daemon=True
        ).start()

    def _load_inputfile_worker(self, show_preview, effective_delim, current_delim_state):
        try:
            if self.reimport_requested:

                try:
                    if hasattr(self, 'imported_paths') and self.imported_paths:
                        dfs = []
                        for p in self.imported_paths:
                            dfs.append(pd.read_csv(p))
                        emis_df = pd.concat(dfs, ignore_index=True)

                        # Apply filtering BEFORE aggregation
                        if self.filter_col and (self.filter_start is not None or self.filter_end is not None):
                            try:
                                emis_df = filter_dataframe_by_range(emis_df, self.filter_col, self.filter_start, self.filter_end)
                            except Exception:
                                self.root.after(0, lambda: self._notify('WARNING', 'Filter Range Failed', 'Could not apply range filter from JSON snapshot.', popup=False))
                        
                        if self.filter_col and self.filter_values:
                            try:
                                emis_df = filter_dataframe_by_values(emis_df, self.filter_col, self.filter_values)
                            except Exception:
                                self.root.after(0, lambda: self._notify('WARNING', 'Filter Values Failed', 'Could not apply discrete filter from JSON snapshot.', popup=False))

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
                            if hasattr(emis_df, 'attrs'):
                                emis_df.attrs.pop('_detected_pollutants', None)
                            
                            pols = detect_pollutants(emis_df)
                            if pols:
                                agg_dict = {p: 'sum' for p in pols}
                                for c in emis_df.columns:
                                    if c not in group_keys and c not in pols:
                                        agg_Dict[c] = 'first'
                                
                                emis_df = emis_df.groupby(group_keys, as_index=False).agg(agg_dict)
                    else:
                        emis_df = pd.read_csv(self.inputfile_path)
                except Exception as exc:
                    self.root.after(0, lambda: self._notify('ERROR', 'Reimport Load Error', f"Failed to load processed emissions data: {exc}", exc=exc))
                    return

                attrs = self._json_arguments.get('imported_attrs')
                if isinstance(attrs, dict):
                    for key, value in attrs.items():
                        try:
                            emis_df.attrs[key] = value
                        except Exception:
                            pass

                emis_df = get_emis_fips(emis_df)
                #raw_df = emis_df.copy(deep=True)

                filtered = emis_df
                # Filtering already applied before aggregation
                
                self.emissions_df = filtered
                self.raw_df = self.emissions_df.copy(deep=True)
                self._ff10_grid_ready = False

                # Pre-compute SCC data in thread
                scc_data = self._compute_scc_data(self.raw_df)

                self.root.after(0, lambda: self._loader_notify('INFO', f"Reimported processed emissions file: {os.path.basename(self.inputfile_path)}"))
                self.reimport_requested = False
                self.root.after(0, lambda: self._finalize_loaded_emissions(show_preview=show_preview, scc_data=scc_data))

            else:
                try:
                    # Thread-safe notify wrapper
                    def safe_notify(level, message):
                        self.root.after(0, lambda: self._loader_notify(level, message))

                    emissions_df, raw_df = read_inputfile(
                        self.inputfile_path,
                        sector=self.sector,
                        delim=effective_delim,
                        skiprows=self.skiprows,
                        comment=self.comment_token,
                        encoding=self.encoding,
                        flter_col=self.filter_col,
                        flter_start=self.filter_start,
                        flter_end=self.filter_end,
                        flter_val=self.filter_values,
                        notify=safe_notify,
                    )
                except Exception as e:
                    err_msg = str(e)
                    if "No valid FIPS code columns found" in err_msg:
                        self.root.after(0, lambda: self._notify('ERROR', 'Input Not Supported', 'The input file format is not supported (missing FIPS/Region columns).', exc=None))
                    else:
                        self.root.after(0, lambda e=e: self._notify('ERROR', 'Emissions Load Error', str(e), exc=e))
                    return
                
                self.emissions_df = emissions_df
                self.raw_df = raw_df
                if effective_delim is not None:
                    self.emissions_delim = effective_delim
                self._last_loaded_delim_state = current_delim_state
                
                # Pre-compute SCC data in thread
                scc_data = self._compute_scc_data(self.raw_df)
                
                self.root.after(0, lambda: self._finalize_loaded_emissions(show_preview=show_preview, scc_data=scc_data))
        
        except Exception as e:
             self.root.after(0, lambda: self._notify('ERROR', 'Async Load Error', str(e), exc=e))

    def _compute_scc_data(self, df):
        has = False
        items = []
        scc_map = {}
        try:
            if isinstance(df, pd.DataFrame):
                lower = {c.lower() for c in df.columns}
                has = ('scc' in lower) or any(c in lower for c in ['scc description','scc_description'])
                if has:
                    # Build unique list of SCC + description
                    cmap = {c.lower(): c for c in df.columns}
                    scc_col = cmap.get('scc')
                    desc_col = cmap.get('scc description') or cmap.get('scc_description')
                    scc_series = df[scc_col].astype(str) if scc_col else pd.Series(['']*len(df))
                    desc_series = df[desc_col].astype(str) if desc_col else pd.Series(['']*len(df))
                    # Use code_key as primary key (prefer SCC, fallback to description)
                    code_key = scc_series.where(scc_series.str.strip() != '', other=desc_series)
                    display = code_key.str.strip()
                    with_desc = desc_series.str.strip()
                    # Combine display with description when available and not duplicate
                    display = display.where(with_desc == '' , other=(code_key.str.strip() + ' – ' + with_desc))
                    # Create unique by code_key keeping first description
                    tmp = pd.DataFrame({'code': code_key.fillna(''), 'disp': display.fillna('')})
                    tmp = tmp.drop_duplicates(subset=['code'])
                    # Sort by code for stability
                    try:
                        tmp = tmp.sort_values('code')
                    except Exception:
                        pass
                    items = [('All SCC', '')] + [(row.disp if row.disp else row.code, row.code) for row in tmp.itertuples(index=False)]
                    scc_map = {d: c for d, c in items}
        except Exception:
            has = False
            items = []
            scc_map = {}
        return has, items, scc_map

    def _update_scc_widgets(self, scc_data=None):
        if scc_data:
            has, items, scc_map = scc_data
        else:
            has, items, scc_map = self._compute_scc_data(self.raw_df)
        
        self._has_scc_cols = bool(has)
        if has:
            self._scc_display_to_code = scc_map
            try:
                self.scc_entry['values'] = [d for d, _ in items]
                self.scc_select_var.set('All SCC')
            except Exception:
                pass
        
        # Enable/disable widgets accordingly
        try:
            if self._has_scc_cols:
                self.scc_entry.state(['!disabled'])
                self.scc_label.state(['!disabled'])
            else:
                self.scc_entry.state(['disabled'])
                self.scc_label.state(['disabled'])
                if self.scc_select_var:
                    self.scc_select_var.set('All SCC')
                self._scc_display_to_code = {}
        except Exception:
            pass

    def load_griddesc(self):
        if not self.griddesc_path:
            return
        self._ff10_grid_ready = False
        try:
            grid_names = extract_grid(self.griddesc_path, grid_id=None)
            menu = self.grid_name_menu["menu"]
            menu.delete(0, "end")
            for name in grid_names:
                # Set variable and trigger load_grid_shape() when user selects
                try:
                    menu.add_command(label=name, command=tk._setit(self.grid_name_var, name, lambda *_: self.load_grid_shape()))  # type: ignore[attr-defined]
                except Exception:
                    menu.add_command(label=name, command=lambda v=name: (self.grid_name_var.set(v), self.load_grid_shape()))
            # Do not auto-create domain shapefile; wait until user selects a grid name
            try:
                self.grid_name_var.set("Select Grid")
            except Exception:
                pass
            # Clear any existing grid geometry to avoid stale merges
            self.grid_gdf = None
            self._invalidate_merge_cache()
        except Exception as e:
            self._notify('ERROR', 'GRIDDESC Load Error', str(e), exc=e)

    def load_grid_shape(self):
        if not self.griddesc_path or not self.grid_name_var.get():
            return
        self._ff10_grid_ready = False
        try:
            self.grid_gdf = create_domain_gdf(self.griddesc_path, self.grid_name_var.get())
            self._notify('INFO', 'Grid Loaded', f"Successfully created grid shape for '{self.grid_name_var.get()}'.", popup=False)
            self._invalidate_merge_cache()
            self._ensure_ff10_grid_mapping()
        except Exception as e:
            self._notify('ERROR', 'Grid Creation Error', str(e), exc=e)

    def load_shpfile(self):
        try:
            self.counties_gdf = read_shpfile(self.counties_path, True)
            # Save updated path on successful load
            try:
                self._save_settings()
            except Exception:
                pass
            self._invalidate_merge_cache()
        except Exception as e:
            self._notify('ERROR', 'Counties Load Error', str(e), exc=e)
            return

    def load_overlay(self):
        try:
            self.overlay_gdf = read_shpfile(self.overlay_path, False)
        except Exception as e:
            self._notify('ERROR', 'Overlay Shapefile Load Error', str(e), exc=e)
            return

    def _merged(self, plot_by_mode=None, scc_selection=None, scc_code_map=None, notify=None) -> Optional[gpd.GeoDataFrame]:
        if self.emissions_df is None:
            return None

        def _do_notify(level, title, msg, exc=None):
            if notify:
                notify(level, title, msg, exc)
            else:
                self._notify(level, title, msg, exc=exc)

        mode = (plot_by_mode if plot_by_mode is not None else (self.plot_by_var.get().lower() if self.plot_by_var else 'auto'))
        base_gdf: Optional[gpd.GeoDataFrame] = None
        merge_on: Optional[str] = None
        geometry_tag = None
        try:
            source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type') if isinstance(self.emissions_df, pd.DataFrame) else None
        except Exception:
            source_type = None

        sel_display = scc_selection if scc_selection is not None else (self.scc_select_var.get() if self.scc_select_var else 'All SCC')
        sel_code = ''
        code_map = scc_code_map if scc_code_map is not None else self._scc_display_to_code
        if code_map:
            try:
                sel_code = code_map.get(sel_display, '') or ''
            except Exception:
                sel_code = ''
        use_scc_filter = bool(self._has_scc_cols and sel_code)

        if mode == 'grid':
            if self.grid_gdf is None:
                _do_notify('WARNING', 'Grid not loaded', 'Select a GRIDDESC and Grid Name first to build the grid geometry.')
                return None
            self._ensure_ff10_grid_mapping(notify_success=False)
            base_gdf = self.grid_gdf
            merge_on = 'GRID_RC'
            geometry_tag = 'grid'
        elif mode == 'county':
            if self.counties_gdf is None:
                _do_notify('WARNING', 'Counties not loaded', 'Load a counties shapefile or use the online counties option.')
                return None
            base_gdf = self.counties_gdf
            merge_on = 'FIPS'
            geometry_tag = 'county'
        else:
            if self.grid_gdf is not None:
                if 'GRID_RC' not in getattr(self.emissions_df, 'columns', []):
                    self._ensure_ff10_grid_mapping(notify_success=False)
                if 'GRID_RC' in getattr(self.emissions_df, 'columns', []):
                    base_gdf = self.grid_gdf
                    merge_on = 'GRID_RC'
                    geometry_tag = 'grid'
            if base_gdf is None and self.counties_gdf is not None:
                base_gdf = self.counties_gdf
                merge_on = 'FIPS'
                if not merge_on or merge_on.lower() not in {c.lower() for c in self.emissions_df.columns}:
                    merge_on = 'FIPS'
                geometry_tag = 'county'
            if base_gdf is None:
                _do_notify('WARNING', 'No suitable geometry', 'Could not find a suitable shapefile (Counties or Grid) for the loaded emissions data.')
                return None

        if merge_on and isinstance(base_gdf, gpd.GeoDataFrame):
            if merge_on not in base_gdf.columns:
                if merge_on.lower() == 'region_cd' and 'FIPS' in base_gdf.columns:
                    try:
                        base_gdf = base_gdf.copy()
                        base_gdf['region_cd'] = base_gdf['FIPS']
                    except Exception:
                        pass
                else:
                    _do_notify('WARNING', 'Geometry Missing Column', f"Selected geometry layer lacks '{merge_on}' column.")
                    return None

        pol_tuple = tuple(self.pollutants or [])
        if not pol_tuple:
            try:
                pol_tuple = tuple(detect_pollutants(self.emissions_df))
            except Exception:
                pol_tuple = ()

        cache_key = (
            geometry_tag or mode,
            merge_on or '',
            id(base_gdf) if base_gdf is not None else 0,
            id(self.emissions_df) if isinstance(self.emissions_df, pd.DataFrame) else 0,
            id(self.raw_df) if isinstance(self.raw_df, pd.DataFrame) else 0,
            sel_code if use_scc_filter else '',
            pol_tuple,
        )
        cached = self._merged_cache.get(cache_key)
        if cached is not None:
            cached_merged, cached_prepared, cached_key = cached
            try:
                merged_view = cached_merged.copy()
            except Exception:
                merged_view = cached_merged
            try:
                if cached_prepared is not None:
                    merged_view.attrs['__prepared_emis'] = cached_prepared
            except Exception:
                pass
            try:
                if cached_key:
                    merged_view.attrs['__merge_key'] = cached_key
            except Exception:
                pass
            return merged_view

        emis_for_merge: Optional[pd.DataFrame]
        emis_for_merge = self.emissions_df if isinstance(self.emissions_df, pd.DataFrame) else None
        if emis_for_merge is None:
            _do_notify('ERROR', 'No Emissions Data', 'Emissions dataset is not loaded or invalid.')
            return None

        raw_df_for_scc = None
        if use_scc_filter and isinstance(self.raw_df, pd.DataFrame):
            cmap = {c.lower(): c for c in self.raw_df.columns}
            scc_col = cmap.get('scc')
            desc_col = cmap.get('scc description') or cmap.get('scc_description')
            if scc_col and scc_col in self.raw_df.columns:
                try:
                    mask = self.raw_df[scc_col].astype(str) == sel_code
                except Exception:
                    mask = pd.Series([False] * len(self.raw_df), index=self.raw_df.index)
            elif desc_col and desc_col in self.raw_df.columns:
                try:
                    mask = self.raw_df[desc_col].astype(str) == sel_code
                except Exception:
                    mask = pd.Series([False] * len(self.raw_df), index=self.raw_df.index)
            else:
                mask = pd.Series([False] * len(self.raw_df), index=self.raw_df.index)
            raw_df_for_scc = self.raw_df.loc[mask].copy()
            if raw_df_for_scc.empty:
                _do_notify('WARNING', 'SCC filter matched 0 rows', f'No rows matched SCC: {sel_display}. Showing all SCC.')
                raw_df_for_scc = None

        if merge_on not in emis_for_merge.columns or use_scc_filter:
            try:
                raw_df = raw_df_for_scc if use_scc_filter else self.raw_df  # type: ignore[attr-defined]
            except Exception:
                raw_df = None
            if isinstance(raw_df, pd.DataFrame) and merge_on in raw_df.columns:
                pols = list(self.pollutants or detect_pollutants(raw_df))
                if pols:
                    try:
                        subset = raw_df[[merge_on] + pols]
                        try:
                            grouped = subset.groupby(merge_on, dropna=False, sort=False)
                        except TypeError:
                            grouped = subset.groupby(merge_on, sort=False)
                        agg = grouped.sum(numeric_only=True).reset_index()
                        try:
                            agg.attrs = dict(getattr(emis_for_merge, 'attrs', {}))
                        except Exception:
                            pass
                        emis_for_merge = agg
                    except Exception:
                        pass

        if merge_on not in emis_for_merge.columns:
            if merge_on == 'GRID_RC':
                _do_notify('WARNING', 'No GRID_RC in data', 'Plot by Grid requires emissions with X/Y Cell (GRID_RC).')
            else:
                if source_type in ('ff10_point', 'ff10_nonpoint'):
                    _do_notify('WARNING', 'No region_cd in data', 'FF10 data requires region_cd for county plots.')
                else:
                    _do_notify('WARNING', 'No FIPS in data', 'Plot by County requires emissions with FIPS codes.')
            return None

        try:
            merged, prepared_emis = merge_emissions_with_geometry(
                emis_for_merge,
                base_gdf,
                merge_on,
                sort=False,
                copy_geometry=False,
            )
        except Exception as exc:
            _do_notify('ERROR', 'Geometry Merge Failed', str(exc), exc=exc)
            return None

        try:
            matched_vals = prepared_emis[merge_on].dropna()
            merged['__has_emissions'] = merged[merge_on].isin(matched_vals)
        except Exception:
            try:
                merged['__has_emissions'] = merged[merge_on].notna()
            except Exception:
                merged['__has_emissions'] = False

        try:
            cache_df = merged.copy()
        except Exception:
            cache_df = merged
        self._merged_cache[cache_key] = (cache_df, prepared_emis, merge_on)
        return merged

    def _setup_hover(self, merged: gpd.GeoDataFrame, pollutant: str, ax=None, lonlat_transformer: Optional[pyproj.Transformer] = None) -> None:
        """Enhance status bar to show pollutant at cursor and WGS84 lon/lat by overriding format_coord.
        Pass the GeoDataFrame in the same CRS as the axes (reprojected if needed), and optionally a transformer
        that converts axes coordinates to WGS84 lon/lat for display.
        """
        try:
            from shapely.geometry import Point  # type: ignore
        except Exception:
            return
        gdf = merged
        try:
            sindex = gdf.sindex
        except Exception:
            sindex = None

        def _fmt(x: float, y: float) -> str:
            # Build lon/lat if possible; otherwise fall back to axes coords
            base = None
            try:
                if lonlat_transformer is not None:
                    lon, lat = lonlat_transformer.transform(x, y)
                    if abs(lon) < 1e6 and abs(lat) < 1e6:
                        base = f"lon={lon:.4f}°, lat={lat:.4f}°"
                elif getattr(gdf, 'crs', None) is not None:
                    try:
                        # If geometries are already geographic
                        is_geog = gdf.crs.is_geographic if hasattr(gdf.crs, 'is_geographic') else False
                        if is_geog:
                            base = f"lon={x:.4f}°, lat={y:.4f}°"
                    except Exception:
                        pass
            except Exception:
                base = None
            if base is None:
                base = f"x={x:.4f}, y={y:.4f}"
            try:
                pt = Point(x, y)
                if sindex is not None:
                    cand_idx = list(sindex.intersection((x, y, x, y)))
                else:
                    # Fallback to brute-force if no sindex
                    cand_idx = range(len(gdf))
                for idx in cand_idx:
                    row = gdf.iloc[idx]
                    geom = row.geometry
                    if geom is not None and geom.contains(pt):
                        parts = [base]
                        fips = row.get('FIPS')
                        region_cd = row.get('region_cd') or row.get('REGION_CD')
                        gridrc = row.get('GRID_RC')
                        if isinstance(fips, (str, int)):
                            parts.append(f"FIPS={str(fips).zfill(6) if str(fips).isdigit() else str(fips)}")
                        elif isinstance(region_cd, (str, int, float)):
                            region_str = str(region_cd)
                            parts.append(f"region_cd={region_str.zfill(6) if region_str.isdigit() else region_str}")
                        elif isinstance(gridrc, (str, int)):
                            parts.append(f"GRID_RC={gridrc}")
                        val = row.get(pollutant)
                        if not pd.isna(val):
                            try:
                                parts.append(f"{pollutant}={float(val):.4g}")
                            except Exception:
                                parts.append(f"{pollutant}={val}")
                        return ", ".join(parts)
            except Exception:
                pass
            return base

        try:
            target_ax = ax if ax is not None else self.ax
            if target_ax is not None:
                target_ax.format_coord = _fmt
        except Exception:
            pass

    # ---- Rectangle zoom (left-drag zoom in, right-drag zoom out) ----
    def _install_box_zoom(self):
        import matplotlib.patches as mpatches

        # Clean previous connections/rectangle
        self._remove_box_zoom()

        ax = self.ax
        fig = self.fig
        if ax is None or fig is None:
            return

        rect = mpatches.Rectangle((0, 0), 0, 0, fill=False, ec='red', lw=1.2, zorder=9999)
        ax.add_patch(rect)
        rect.set_visible(False)
        self._zoom_rect = rect
        self._zoom_press = None

        def on_press(event):
            if event.inaxes != ax:
                return
            if event.button not in (1, 3):  # 1=left (zoom in), 3=right (zoom out)
                return
            # Don't interfere with built-in toolbar modes (zoom/pan)
            try:
                if getattr(self, 'toolbar', None) is not None and getattr(self.toolbar, 'mode', '') == 'pan/zoom':
                    return
            except Exception:
                pass
            self._zoom_press = (event.xdata, event.ydata, event.button)
            # Set rectangle edge color based on zoom direction
            rect.set_edgecolor('red' if event.button == 1 else 'blue')
            rect.set_linestyle('-')
            rect.set_linewidth(1.2)
            rect.set_visible(True)
            rect.set_xy((event.xdata, event.ydata))
            rect.set_width(0)
            rect.set_height(0)
            fig.canvas.draw_idle()

        def on_motion(event):
            if self._zoom_press is None or event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
            try:
                if getattr(self, 'toolbar', None) is not None and getattr(self.toolbar, 'mode', '') == 'pan/zoom':
                    return
            except Exception:
                pass
            x0, y0, btn = self._zoom_press
            x1, y1 = event.xdata, event.ydata
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            rect.set_xy((xmin, ymin))
            rect.set_width(max(0.0, xmax - xmin))
            rect.set_height(max(0.0, ymax - ymin))
            fig.canvas.draw_idle()

        def on_release(event):
            if self._zoom_press is None:
                return
            try:
                if getattr(self, 'toolbar', None) is not None and getattr(self.toolbar, 'mode', '') == 'pan/zoom':
                    # Hide rectangle and abort when toolbar is in a mode
                    rect.set_visible(False)
                    fig.canvas.draw_idle()
                    self._zoom_press = None
                    return
            except Exception:
                pass
            # Derive end coordinates even if mouse was released outside the axes
            x0, y0, btn = self._zoom_press
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                x1, y1 = event.xdata, event.ydata
            else:
                # Fallback to rectangle's current extents
                try:
                    x1 = rect.get_x() + rect.get_width()
                    y1 = rect.get_y() + rect.get_height()
                except Exception:
                    x1, y1 = x0, y0
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            # Hide the rubber band
            rect.set_visible(False)
            fig.canvas.draw_idle()
            self._zoom_press = None

            # Ignore tiny drags
            if abs(xmax - xmin) < 1e-12 or abs(ymax - ymin) < 1e-12:
                return

            if btn == 1:
                # Zoom in: fit to box
                # Push current view to toolbar history before changing
                try:
                    if getattr(self, 'toolbar', None) is not None:
                        self.toolbar.push_current()  # type: ignore[attr-defined]
                except Exception:
                    pass
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            else:
                # Zoom out: use the box as a reference to compute a scale factor
                cur_xmin, cur_xmax = ax.get_xlim()
                cur_ymin, cur_ymax = ax.get_ylim()
                cur_w = max(1e-12, cur_xmax - cur_xmin)
                cur_h = max(1e-12, cur_ymax - cur_ymin)
                box_w = max(1e-12, xmax - xmin)
                box_h = max(1e-12, ymax - ymin)
                # Scale factor: how much of current view the box represents
                sx = min(10.0, max(1e-6, box_w / cur_w))
                sy = min(10.0, max(1e-6, box_h / cur_h))
                s = min(sx, sy)
                # New size grows inversely with s (smaller box -> larger zoom-out)
                new_w = cur_w / s
                new_h = cur_h / s
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                # Clamp zoom-out to the original/base view when known
                try:
                    if hasattr(self, '_base_view') and self._base_view:
                        (bx0, bx1), (by0, by1) = self._base_view
                        base_w = max(1e-12, bx1 - bx0)
                        base_h = max(1e-12, by1 - by0)
                        if new_w >= base_w or new_h >= base_h:
                            ax.set_xlim(bx0, bx1)
                            ax.set_ylim(by0, by1)
                            fig.canvas.draw_idle()
                            return
                        # Keep view within base bounds
                        x0_new = min(max(bx0, cx - new_w / 2.0), bx1 - new_w)
                        y0_new = min(max(by0, cy - new_h / 2.0), by1 - new_h)
                        ax.set_xlim(x0_new, x0_new + new_w)
                        ax.set_ylim(y0_new, y0_new + new_h)
                        fig.canvas.draw_idle()
                        return
                except Exception:
                    pass
                # Push current view to toolbar history before changing
                try:
                    if getattr(self, 'toolbar', None) is not None:
                        self.toolbar.push_current()  # type: ignore[attr-defined]
                except Exception:
                    pass
                ax.set_xlim(cx - new_w / 2.0, cx + new_w / 2.0)
                ax.set_ylim(cy - new_h / 2.0, cy + new_h / 2.0)
            fig.canvas.draw_idle()

        cid1 = self.canvas.mpl_connect('button_press_event', on_press)
        cid2 = self.canvas.mpl_connect('motion_notify_event', on_motion)
        cid3 = self.canvas.mpl_connect('button_release_event', on_release)
        self._zoom_cids = [cid1, cid2, cid3]

    def _remove_box_zoom(self):
        # Disconnect events and remove rectangle if present
        if getattr(self, 'canvas', None) is not None:
            try:
                for cid in self._zoom_cids:
                    self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._zoom_cids = []
        if self._zoom_rect is not None:
            try:
                self._zoom_rect.remove()
            except Exception:
                pass
            self._zoom_rect = None

    def _lazy_load_inputfile(self, *, show_preview: bool = False) -> bool:
        """Load the remembered input file on demand when data is not yet in memory."""
        if isinstance(self.emissions_df, pd.DataFrame):
            return True
        path = (self.inputfile_path or '').strip()
        if not path:
            return False
        try:
            self.load_inputfile(show_preview=show_preview)
        except Exception as exc:
            self._notify('ERROR', 'Auto Load Failed', str(exc), exc=exc)
            return False
        return isinstance(self.emissions_df, pd.DataFrame)

    def _enable_plot_btn(self):
        if getattr(self, 'plot_btn', None):
            try:
                self.plot_btn.state(['!disabled'])
            except Exception:
                pass

    def plot(self):
        if self.emissions_df is None:
            self._lazy_load_inputfile(show_preview=False)
        if self.emissions_df is None:
            self._notify('WARNING', 'Missing Data', 'Load smkreport and shapefile first.')
            return
        pollutant = self.pollutant_var.get()
        if not pollutant and getattr(self, 'pollutants', None):
            try:
                first_pol = next(iter(self.pollutants))
            except Exception:
                first_pol = ''
            if first_pol:
                self.pollutant_var.set(first_pol)
                pollutant = first_pol
        if not pollutant:
            self._notify('WARNING', 'Select Pollutant', 'No pollutant selected.')
            return
        
        # Capture UI state
        plot_by_mode = self.plot_by_var.get().lower() if self.plot_by_var else 'auto'
        scc_selection = self.scc_select_var.get() if self.scc_select_var else 'All SCC'
        scc_code_map = self._scc_display_to_code.copy() if self._scc_display_to_code else {}
        
        # Determine plotting CRS (LCC only when GRIDDESC + grid name provided; else WGS84)
        try:
            plot_crs_info = self._plot_crs()
        except Exception:
            plot_crs_info = (None, None, None)

        self._set_status("Preparing plot data...", level="INFO")
        
        if getattr(self, 'plot_btn', None):
            try:
                self.plot_btn.state(['disabled'])
            except Exception:
                pass

        threading.Thread(
            target=self._plot_worker, 
            args=(pollutant, plot_by_mode, scc_selection, scc_code_map, plot_crs_info), 
            daemon=True
        ).start()

    def _plot_worker(self, pollutant, plot_by_mode, scc_selection, scc_code_map, plot_crs_info):
        try:
            # Thread-safe notify wrapper
            def safe_notify(level, title, msg, exc=None):
                self.root.after(0, lambda: self._notify(level, title, msg, exc=exc))

            merged = self._merged(
                plot_by_mode=plot_by_mode, 
                scc_selection=scc_selection, 
                scc_code_map=scc_code_map,
                notify=safe_notify
            )
            
            if merged is None:
                self.root.after(0, lambda: self._notify('WARNING', 'Missing Data', 'Load smkreport and shapefile first.'))
                self.root.after(0, self._enable_plot_btn)
                return

            # Reproject
            plot_crs, tf_fwd, tf_inv = plot_crs_info
            try:
                merged_plot = merged.to_crs(plot_crs) if plot_crs is not None and getattr(merged, 'crs', None) is not None else merged
            except Exception:
                merged_plot = merged
            
            self.root.after(0, lambda: self._finalize_plot(merged, merged_plot, pollutant, plot_crs_info))
            
        except Exception as e:
            self.root.after(0, lambda: self._notify('ERROR', 'Plot Prep Error', str(e), exc=e))
            self.root.after(0, self._enable_plot_btn)

    def _finalize_plot(self, merged, merged_plot, pollutant, plot_crs_info):
        self._enable_plot_btn()
        self._set_status("Rendering plot...", level="INFO")
        # Save current settings on successful plot generation
        self._save_settings()

        # Lazy import for embedding
        from matplotlib.figure import Figure
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except Exception as e:
            self._notify('ERROR', 'Backend Error', f'Failed to load TkAgg backend: {e}', exc=e)
            return

        # Create a new pop-out window for this plot
        win = tk.Toplevel(self.root)
        win.title("Map: plotting...")
        plot_container = ttk.Frame(win)
        plot_container.pack(side='top', fill='both', expand=True)
        # Scale figure size modestly with UI scale
        try:
            base_w, base_h = 9.0, 5.0
            fig_w = max(6.0, min(16.0, base_w * getattr(self, '_ui_scale', 1.0)))
            fig_h = max(3.5, min(10.0, base_h * getattr(self, '_ui_scale', 1.0)))
        except Exception:
            fig_w, fig_h = 9.0, 5.0
        local_fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        local_ax = local_fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(local_fig, master=plot_container)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        # Toolbar (remove Zoom tool)
        try:
            try:
                import PIL.ImageTk  # type: ignore  # noqa: F401
            except Exception:
                pass
            BaseNav = NavigationToolbar2Tk
            class _NoZoomToolbar(BaseNav):
                toolitems = tuple(t for t in getattr(BaseNav, 'toolitems', []) if t and t[0] != 'Zoom')
            toolbar = _NoZoomToolbar(canvas, plot_container, pack_toolbar=False)
        except Exception:
            from matplotlib.backends._backend_tk import NavigationToolbar2Tk as _Nav
            class _TextToolbar(_Nav):
                def _Button(self, text, image_file, toggle=False, command=None):
                    if text == 'Zoom':
                        return tk.Button(master=self, text='')
                    if toggle:
                        var = tk.IntVar(master=self, value=0)
                        btn = tk.Checkbutton(master=self, text=text, indicatoron=0, variable=var, command=command)
                        def _select():
                            try: var.set(1)
                            except Exception: pass
                        def _deselect():
                            try: var.set(0)
                            except Exception: pass
                        btn.select = _select  # type: ignore[attr-defined]
                        btn.deselect = _deselect  # type: ignore[attr-defined]
                    else:
                        btn = tk.Button(master=self, text=text, command=command)
                    btn.pack(side=tk.LEFT)
                    return btn
            toolbar = _TextToolbar(canvas, plot_container, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side='top', fill='x')
        cbar_ax = None

        # Export controls (extent checkbox + button)
        try:
            export_frame = ttk.Frame(plot_container)
            export_frame.pack(side='top', fill='x', pady=(2, 0))
            extent_only_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(export_frame, text='Current extent only', variable=extent_only_var).pack(side='left', padx=(4, 8))
            def _export_layer():
                try:
                    # Choose file path and format by extension
                    path = filedialog.asksaveasfilename(defaultextension='.gpkg',
                                                        filetypes=[('GeoPackage', '*.gpkg'), ('ESRI Shapefile', '*.shp'), ('All', '*.*')])
                except Exception as e:
                    self._notify('ERROR', 'Export Error', f'Could not open save dialog: {e}', exc=e)
                    return
                if not path:
                    return
                try:
                    gdf_src = merged_plot if 'merged_plot' in locals() and isinstance(merged_plot, gpd.GeoDataFrame) else merged
                except Exception:
                    gdf_src = merged
                gdf_to_write = gdf_src
                # Subset to current extent if requested
                if extent_only_var.get():
                    try:
                        from shapely.geometry import box as _box
                        xmin, xmax = local_ax.get_xlim(); ymin, ymax = local_ax.get_ylim()
                        bbox_geom = _box(min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))
                        try:
                            sidx = gdf_src.sindex
                        except Exception:
                            sidx = None
                        if sidx is not None:
                            try:
                                cand = list(sidx.intersection(bbox_geom.bounds))
                                gdf_to_write = gdf_src.iloc[cand]
                            except Exception:
                                gdf_to_write = gdf_src
                        try:
                            gdf_to_write = gdf_to_write[gdf_to_write.geometry.intersects(bbox_geom)]
                        except Exception:
                            pass
                    except Exception:
                        pass
                # Keep only key id and selected pollutant to keep schema small
                try:
                    cols = ['geometry']
                    for key in ('FIPS', 'region_cd', 'GRID_RC'):
                        if key in gdf_to_write.columns:
                            cols.append(key)
                    pol = pollutant
                    if pol in gdf_to_write.columns:
                        cols.append(pol)
                    gdf_out = gdf_to_write[cols].copy()
                except Exception:
                    gdf_out = gdf_to_write
                # Normalize dtypes for file drivers (avoid pandas nullable Int64)
                try:
                    for c in gdf_out.columns:
                        if c != 'geometry' and str(gdf_out[c].dtype).lower().startswith('int'):  # pandas nullable int
                            gdf_out[c] = gdf_out[c].astype('float64')
                except Exception:
                    pass
                # Determine driver from extension
                p_lower = path.lower()
                driver = 'GPKG' if p_lower.endswith('.gpkg') else 'ESRI Shapefile'
                # Layer name for gpkg
                layer = None
                if driver == 'GPKG':
                    try:
                        import re as _re
                        nm = pol if 'pol' in locals() else 'emissions'
                        layer = _re.sub(r'[^A-Za-z0-9_]+', '_', str(nm))[:63] or 'layer'
                    except Exception:
                        layer = 'layer'
                try:
                    if driver == 'GPKG':
                        gdf_out.to_file(path, layer=layer, driver=driver)
                    else:
                        # Ensure .shp extension
                        if not p_lower.endswith('.shp'):
                            path = path + '.shp'
                        gdf_out.to_file(path, driver=driver)
                    self._notify('INFO', 'Export Complete', f'Exported {len(gdf_out)} features to {path}')
                except Exception as e:
                    self._notify('ERROR', 'Export Error', f'Failed to export: {e}', exc=e)
            ttk.Button(export_frame, text='Export Layer…', command=_export_layer).pack(side='left')
        except Exception:
            pass

        # Keep references on the instance to prevent GC of handlers
        self._pop_fig = local_fig
        self._pop_ax = local_ax
        self._pop_canvas = canvas
        self._pop_toolbar = toolbar

        plot_crs, tf_fwd, tf_inv = plot_crs_info if plot_crs_info else (None, None, None)
        data = merged[pollutant]
        positive = data[data > 0].dropna()
        vmin = float(positive.min()) if not positive.empty else 0.0
        vmax_val = data.max(skipna=True)
        vmax = float(vmax_val) if pd.notna(vmax_val) else 0.0

        # Colormap selection
        cmap_name = self.cmap_var.get() if hasattr(self, 'cmap_var') and self.cmap_var else 'jet'
        try:
            cmap_registry = getattr(matplotlib, 'colormaps', None)
            if cmap_registry is not None:
                cmap = cmap_registry.get_cmap(cmap_name)
            else:
                cmap = mplcm.get_cmap(cmap_name)
        except Exception:
            # Final fallback
            try:
                cmap = mplcm.get_cmap('jet')
            except Exception:
                cmap = plt.get_cmap('jet')
        
        # Decide discrete vs continuous based on custom bins
        bins = self._parse_bins()
        if len(bins) >= 2:
            # Discrete mapping using boundaries; ignore log scale for coloring
            norm = BoundaryNorm(bins, ncolors=cmap.N, clip=True)
        else:
            # Continuous; honor linear/log selection
            if self.scale_var.get() == 'log' and vmax > 0 and vmin > 0:
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = None

        # Plotting
        # Capture axes before plotting so we can detect the newly created colorbar axis
        pre_axes = set(local_fig.axes)
        merged_plot.plot(
            column=pollutant,
            ax=local_ax,
            legend=True,
            cmap=cmap,
            linewidth=0.05,
            edgecolor='black',
            missing_kwds={
                'color': '#f0f0f0',
                'edgecolor': 'none',
                'label': 'No Data'
            },
            norm=norm
        )
        # Identify newly created colorbar axis (any new axis besides main)
        post_axes = set(local_fig.axes)
        new_axes = [a for a in (post_axes - pre_axes) if a is not local_ax]
        if new_axes:
            cbar_ax = new_axes[0]
            # Label colorbar with units if available
            try:
                unit = self.units_map.get(pollutant)
                if unit:
                    cbar_ax.set_ylabel(unit)
            except Exception:
                pass
            # Apply custom bins as colorbar ticks (if any)
            try:
                from matplotlib.ticker import FixedLocator
                bins_ticks = bins or []
                if bins_ticks:
                    # Show all provided bin edges as ticks; do not clamp to data range.
                    # For log colorbars (not used with BoundaryNorm here), we'd restrict to >0.
                    is_log = isinstance(norm, LogNorm)
                    ticks = [t for t in bins_ticks if (t > 0) ] if is_log else list(bins_ticks)
                    if ticks:
                        # Determine orientation by axis box shape
                        try:
                            bbox = cbar_ax.get_position()
                            orient_vertical = (bbox.height >= bbox.width)
                        except Exception:
                            orient_vertical = True
                        if orient_vertical:
                            try:
                                cbar_ax.yaxis.set_major_locator(FixedLocator(ticks))
                            except Exception:
                                pass
                        else:
                            try:
                                cbar_ax.xaxis.set_major_locator(FixedLocator(ticks))
                            except Exception:
                                pass
                        try:
                            local_fig.canvas.draw_idle()
                        except Exception:
                            pass
            except Exception:
                pass
        # If plotting by grid and counties layer is available, overlay county boundaries for context
        try:
            mode = (self.plot_by_var.get().lower() if getattr(self, 'plot_by_var', None) else 'auto')
        except Exception:
            mode = 'auto'
        if self.counties_gdf is not None:
            try:
                overlay = self.counties_gdf
                try:
                    overlay = overlay.to_crs(plot_crs) if plot_crs is not None and getattr(overlay, 'crs', None) is not None else overlay
                except Exception:
                    pass
                overlay.plot(ax=local_ax, facecolor='none', edgecolor='black', linewidth=0.2, alpha=0.6)
            except Exception:
                pass
        if self.overlay_gdf is not None:
            try:
                overlay = self.overlay_gdf
                try:
                    overlay = overlay.to_crs(plot_crs) if plot_crs is not None and getattr(overlay, 'crs', None) is not None else overlay
                except Exception:
                    pass
                overlay.plot(ax=local_ax, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.6)
            except Exception:
                pass            
        # Optional zoom to data extent (non-NA values)
        if self.zoom_var.get():
            candidates = None
            values_num = None
            if pollutant in merged_plot.columns:
                try:
                    values_num = pd.to_numeric(merged_plot[pollutant], errors='coerce')
                except Exception:
                    values_num = None
            if values_num is not None:
                try:
                    # Modified to include zero values in zoom extent
                    valid_mask = values_num.notna()
                    if '__has_emissions' in merged_plot.columns:
                        try:
                            has_mask = merged_plot['__has_emissions'].fillna(False)
                            valid_mask = valid_mask | has_mask
                        except Exception:
                            pass
                    if getattr(valid_mask, 'any', lambda: False)():
                        try:
                            candidates = merged_plot[valid_mask.fillna(False)]
                        except Exception:
                            candidates = merged_plot[valid_mask]
                except Exception:
                    candidates = None
            if (candidates is None or candidates.empty) and '__has_emissions' in merged_plot.columns:
                try:
                    has_mask = merged_plot['__has_emissions'].fillna(False)
                    subset = merged_plot[has_mask]
                    if not subset.empty:
                        candidates = subset
                except Exception:
                    pass
            try:
                bounds_df = candidates
                if bounds_df is None or bounds_df.empty:
                    bounds_df = merged_plot
                if bounds_df is None:
                    bounds_df = pd.DataFrame()
            except Exception:
                bounds_df = merged_plot if merged_plot is not None else pd.DataFrame()
            try:
                bounds_df = bounds_df[bounds_df.geometry.notna()]
            except Exception:
                pass
            try:
                bounds_df = bounds_df[~bounds_df.geometry.is_empty]
            except Exception:
                pass
            if bounds_df is not None and not getattr(bounds_df, 'empty', True):
                try:
                    minx, miny, maxx, maxy = bounds_df.total_bounds
                    dx = (maxx - minx) * 0.02 if maxx > minx else 0.1
                    dy = (maxy - miny) * 0.02 if maxy > miny else 0.1
                    local_ax.set_xlim(minx - dx, maxx + dx)
                    local_ax.set_ylim(miny - dy, maxy + dy)
                except Exception:
                    pass
        # Draw graticule lines on projected map
        try:
            if plot_crs is not None and tf_fwd is not None and tf_inv is not None and not plot_crs.is_geographic:
                # Slightly behind the polygons (zorder=0 in draw function)
                self._draw_graticule(local_ax, tf_fwd, tf_inv, lon_step=5, lat_step=5)
                # Equal aspect to preserve shapes visually
                try:
                    local_ax.set_aspect('equal', adjustable='box')
                except Exception:
                    pass
        except Exception:
            pass
        # Record base view for this window and wire Home
        try:
            base_view = (local_ax.get_xlim(), local_ax.get_ylim())
            try:
                base_view_holder = {}
                base_view_holder['limits'] = base_view  # keep stats baseline aligned with Home view
            except Exception:
                pass
            # Save on instance for helper-based zoom to clamp against
            try:
                self._base_view = base_view
            except Exception:
                pass
            def _home(*_a, **_k):
                try:
                    xlim, ylim = base_view
                    local_ax.set_xlim(*xlim)
                    local_ax.set_ylim(*ylim)
                    canvas.draw_idle()
                except Exception:
                    pass
            try:
                # Bind to this window's toolbar, not any class-level toolbar
                toolbar.home = _home  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass
        # Title with source name if available
        try:
            src = self.emissions_df.attrs.get('source_name') if self.emissions_df is not None else None
        except Exception:
            src = None
        # Build title and optional SCC subtitle
        main_title = f"{pollutant} emissions from {src}" if src else f"{pollutant} emission"
        try:
            sel_disp = self.scc_select_var.get() if getattr(self, 'scc_select_var', None) else 'All SCC'
        except Exception:
            sel_disp = 'All SCC'
        if sel_disp and sel_disp != 'All SCC':
            # Append as subtitle on a new line
            local_ax.set_title(f"{main_title}\nSCC: {sel_disp}")
        else:
            local_ax.set_title(main_title)
        # Install hover/status text formatter with emission values and WGS84 lon/lat
        try:
            self._setup_hover(merged_plot, pollutant, ax=local_ax, lonlat_transformer=tf_inv)
        except Exception:
            pass
        # Install rectangle zoom handlers (left-drag zoom in, right-drag zoom out) for this window
        try:
            import matplotlib.patches as mpatches
            rect = mpatches.Rectangle((0, 0), 0, 0, fill=False, ec='red', lw=1.2, zorder=9999)
            local_ax.add_patch(rect)
            rect.set_visible(False)
            zoom_press = {'state': None}
            def on_press(event):
                if event.inaxes != local_ax:
                    return
                if event.button not in (1, 3):
                    return
                try:
                    if getattr(toolbar, 'mode', '') == 'pan/zoom':
                        return
                except Exception:
                    pass
                zoom_press['state'] = (event.xdata, event.ydata, event.button)
                rect.set_edgecolor('red' if event.button == 1 else 'blue')
                rect.set_linestyle('-')
                rect.set_linewidth(1.2)
                rect.set_visible(True)
                rect.set_xy((event.xdata, event.ydata))
                rect.set_width(0)
                rect.set_height(0)
                local_fig.canvas.draw_idle()
            def on_motion(event):
                st = zoom_press['state']
                if st is None or event.inaxes != local_ax or event.xdata is None or event.ydata is None:
                    return
                try:
                    if getattr(toolbar, 'mode', ''):
                        return
                except Exception:
                    pass
                x0, y0, btn = st
                x1, y1 = event.xdata, event.ydata
                xmin, xmax = sorted([x0, x1])
                ymin, ymax = sorted([y0, y1])
                rect.set_xy((xmin, ymin))
                rect.set_width(max(0.0, xmax - xmin))
                rect.set_height(max(0.0, ymax - ymin))
                local_fig.canvas.draw_idle()
            def on_release(event):
                st = zoom_press['state']
                if st is None:
                    return
                try:
                    if getattr(toolbar, 'mode', '') == 'pan/zoom':
                        rect.set_visible(False)
                        local_fig.canvas.draw_idle()
                        zoom_press['state'] = None
                        return
                except Exception:
                    pass
                x0, y0, btn = st
                # Accept release even if outside the axes; use rectangle extents
                if event.inaxes == local_ax and event.xdata is not None and event.ydata is not None:
                    x1, y1 = event.xdata, event.ydata
                else:
                    try:
                        x1 = rect.get_x() + rect.get_width()
                        y1 = rect.get_y() + rect.get_height()
                    except Exception:
                        x1, y1 = x0, y0
                xmin, xmax = sorted([x0, x1])
                ymin, ymax = sorted([y0, y1])
                rect.set_visible(False)
                local_fig.canvas.draw_idle()
                zoom_press['state'] = None
                if abs(xmax - xmin) < 1e-12 or abs(ymax - ymin) < 1e-12:
                    return
                try:
                    toolbar.push_current()  # type: ignore[attr-defined]
                except Exception:
                    pass
                if btn == 1:
                    local_ax.set_xlim(xmin, xmax)
                    local_ax.set_ylim(ymin, ymax)
                else:
                    cur_xmin, cur_xmax = local_ax.get_xlim()
                    cur_ymin, cur_ymax = local_ax.get_ylim()
                    cur_w = max(1e-12, cur_xmax - cur_xmin)
                    cur_h = max(1e-12, cur_ymax - cur_ymin)
                    box_w = max(1e-12, xmax - xmin)
                    box_h = max(1e-12, ymax - ymin)
                    sx = min(10.0, max(1e-6, box_w / cur_w))
                    sy = min(10.0, max(1e-6, box_h / cur_h))
                    s = min(sx, sy)
                    new_w = cur_w / s
                    new_h = cur_h / s
                    cx = 0.5 * (xmin + xmax)
                    cy = 0.5 * (ymin + ymax)
                    # Clamp against base view if available
                    try:
                        (bx0, bx1), (by0, by1) = base_view
                        base_w = max(1e-12, bx1 - bx0)
                        base_h = max(1e-12, by1 - by0)
                        if new_w >= base_w or new_h >= base_h:
                            local_ax.set_xlim(bx0, bx1)
                            local_ax.set_ylim(by0, by1)
                        else:
                            x0_new = min(max(bx0, cx - new_w / 2.0), bx1 - new_w)
                            y0_new = min(max(by0, cy - new_h / 2.0), by1 - new_h)
                            local_ax.set_xlim(x0_new, x0_new + new_w)
                            local_ax.set_ylim(y0_new, y0_new + new_h)
                    except Exception:
                        local_ax.set_xlim(cx - new_w / 2.0, cx + new_w / 2.0)
                        local_ax.set_ylim(cy - new_h / 2.0, cy + new_h / 2.0)
                local_fig.canvas.draw_idle()
        except Exception:
            pass
        # Connect the handlers to the canvas so zoom works
        try:
            cid_press = canvas.mpl_connect('button_press_event', on_press)
            cid_move = canvas.mpl_connect('motion_notify_event', on_motion)
            cid_release = canvas.mpl_connect('button_release_event', on_release)
            # Store cids on the window for cleanup
            try:
                setattr(win, '_zoom_cids', [cid_press, cid_move, cid_release])
            except Exception:
                pass
            # Ensure we disconnect on close to avoid leaks
            def _on_plot_close():
                # Disconnect zoom event handlers and destroy window
                try:
                    for cid in getattr(win, '_zoom_cids', []) or []:
                        try:
                            canvas.mpl_disconnect(cid)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    artists = getattr(local_ax, '_smk_graticule_artists', None)
                    if isinstance(artists, dict):
                        for key in ('lines', 'texts'):
                            for art in artists.get(key, []) or []:
                                try:
                                    art.remove()
                                except Exception:
                                    pass
                    local_ax._smk_graticule_artists = {'lines': [], 'texts': []}  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    for cid in getattr(local_ax, '_smk_graticule_cids', []) or []:
                        try:
                            local_ax.callbacks.disconnect(cid)
                        except Exception:
                            pass
                    local_ax._smk_graticule_cids = []  # type: ignore[attr-defined]
                    local_ax._smk_graticule_callback = None  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    win.destroy()
                except Exception:
                    pass
            try:
                win.protocol("WM_DELETE_WINDOW", _on_plot_close)
            except Exception:
                pass
            # Use the plotted GeoDataFrame (in projected CRS when applicable)
            gdf_for_stats = merged_plot
            prepared_for_stats = merged.attrs.get('__prepared_emis') if hasattr(merged, 'attrs') else None
            merge_key_name = merged.attrs.get('__merge_key') if hasattr(merged, 'attrs') else None
            if (
                isinstance(prepared_for_stats, pd.DataFrame)
                and isinstance(merge_key_name, str)
                and merge_key_name in prepared_for_stats.columns
            ):
                stats_source = prepared_for_stats
            else:
                stats_source = merged
                prepared_for_stats = None
                merge_key_name = None
            pol = pollutant
            unit = None
            try:
                unit = self.units_map.get(pol)
            except Exception:
                unit = None
            # Text artist anchored to bottom-left inside axes
            stats_text = local_ax.text(
                0.01, 0.01,
                "",
                transform=local_ax.transAxes,
                ha='left', va='bottom', fontsize=9, color='#333333', zorder=2,
                bbox=dict(facecolor='white', edgecolor='#aaaaaa', alpha=0.7, pad=2)
            )

            def _calc_stats(raw_values):
                if raw_values is None:
                    return None
                series = raw_values if isinstance(raw_values, pd.Series) else pd.Series(raw_values)
                numeric = pd.to_numeric(series, errors='coerce').dropna()
                if numeric.empty:
                    return None
                arr = numeric.to_numpy(dtype=float, copy=False)
                return (
                    float(arr.min()),
                    float(arr.mean()),
                    float(arr.max()),
                    float(arr.sum()),
                )

            def _format_stats(stats_tuple):
                def _fmt(value):
                    return f"{value:.4g}" if np.isfinite(value) else "--"

                if stats_tuple is None:
                    base = "Min: --  Mean: --  Max: --  Sum: --"
                else:
                    vmin, vmean, vmax, vsum = stats_tuple
                    base = (
                        f"Min: {_fmt(vmin)}  Mean: {_fmt(vmean)}  "
                        f"Max: {_fmt(vmax)}  Sum: {_fmt(vsum)}"
                    )
                if unit:
                    return f"{base} {unit}"
                return base

            try:
                global_series = stats_source[pol]
            except Exception:
                global_series = None
            global_stats = _calc_stats(global_series)
            base_view_holder: Dict[str, Optional[Tuple[Tuple[float, float], Tuple[float, float]]]] = {'limits': None}

            def _view_matches_base(xlim, ylim, tol=1e-6):
                base = base_view_holder.get('limits')
                if base is None:
                    return True
                (bx0, bx1), (by0, by1) = base
                return (
                    abs(xlim[0] - bx0) <= tol * max(1.0, abs(bx0)) and
                    abs(xlim[1] - bx1) <= tol * max(1.0, abs(bx1)) and
                    abs(ylim[0] - by0) <= tol * max(1.0, abs(by0)) and
                    abs(ylim[1] - by1) <= tol * max(1.0, abs(by1))
                )

            def _update_stats_for_view():
                if global_stats is None:
                    stats_text.set_text(_format_stats(None))
                    try:
                        local_fig.canvas.draw_idle()
                    except Exception:
                        pass
                    return
                xlim = local_ax.get_xlim()
                ylim = local_ax.get_ylim()
                if base_view_holder['limits'] is None:
                    base_view_holder['limits'] = (xlim, ylim)
                    stats_text.set_text(_format_stats(global_stats))
                    try:
                        local_fig.canvas.draw_idle()
                    except Exception:
                        pass
                    return
                if _view_matches_base(xlim, ylim):
                    stats_text.set_text(_format_stats(global_stats))
                    try:
                        local_fig.canvas.draw_idle()
                    except Exception:
                        pass
                    return
                try:
                    from shapely.geometry import box as _box
                except Exception:
                    stats_text.set_text(_format_stats(global_stats))
                    try:
                        local_fig.canvas.draw_idle()
                    except Exception:
                        pass
                    return
                if not (xlim[0] < xlim[1] and ylim[0] < ylim[1]):
                    stats_text.set_text(_format_stats(global_stats))
                    try:
                        local_fig.canvas.draw_idle()
                    except Exception:
                        pass
                    return
                bbox_geom = _box(min(xlim), min(ylim), max(xlim), max(ylim))
                try:
                    sidx = gdf_for_stats.sindex
                except Exception:
                    sidx = None
                if sidx is not None:
                    try:
                        idx = list(sidx.intersection(bbox_geom.bounds))
                        sub = gdf_for_stats.iloc[idx]
                    except Exception:
                        sub = gdf_for_stats
                else:
                    sub = gdf_for_stats
                try:
                    sub = sub[sub.geometry.intersects(bbox_geom)]
                except Exception:
                    pass
                subset_values = None
                if (
                    prepared_for_stats is not None
                    and merge_key_name
                    and hasattr(sub, 'columns')
                    and merge_key_name in sub.columns
                    and merge_key_name in prepared_for_stats.columns
                ):
                    try:
                        keys = sub[merge_key_name]
                        keys = keys[keys.notna()]
                        if not keys.empty:
                            subset_values = prepared_for_stats.loc[
                                prepared_for_stats[merge_key_name].isin(keys.unique()),
                                pol,
                            ]
                    except Exception:
                        subset_values = None
                if subset_values is None:
                    if sub is not None and hasattr(sub, 'index'):
                        try:
                            subset_values = merged.loc[sub.index, pol]
                        except Exception:
                            try:
                                subset_values = sub.get(pol) if hasattr(sub, 'get') else None
                            except Exception:
                                subset_values = None
                    elif sub is not None and hasattr(sub, 'get'):
                        subset_values = sub.get(pol)
                subset_stats = _calc_stats(subset_values)
                if subset_stats is None:
                    subset_stats = global_stats
                stats_text.set_text(_format_stats(subset_stats))
                try:
                    local_fig.canvas.draw_idle()
                except Exception:
                    pass
            # Connect to axis limit changes
            cbids = []
            try:
                cbids.append(local_ax.callbacks.connect('xlim_changed', lambda ax: _update_stats_for_view()))
            except Exception:
                pass
            try:
                cbids.append(local_ax.callbacks.connect('ylim_changed', lambda ax: _update_stats_for_view()))
            except Exception:
                pass
            # Run once initially
            _update_stats_for_view()
            # Ensure cleanup on window close
            try:
                if hasattr(win, '_zoom_cids'):
                    # We already have a close handler; wrap it to include stats disconnect
                    old_handler = win.protocol("WM_DELETE_WINDOW")
                else:
                    old_handler = None
            except Exception:
                old_handler = None
            def _close_with_stats_cleanup():
                try:
                    for cid in cbids:
                        try:
                            local_ax.callbacks.disconnect(cid)
                        except Exception:
                            pass
                finally:
                    if callable(old_handler):
                        try:
                            old_handler()
                        except Exception:
                            try:
                                win.destroy()
                            except Exception:
                                pass
                    else:
                        try:
                            win.destroy()
                        except Exception:
                            pass
            try:
                win.protocol("WM_DELETE_WINDOW", _close_with_stats_cleanup)
            except Exception:
                pass
        except Exception:
            pass
        # Finalize window title
        try:
            if src:
                win.title(f"Map: {pollutant} • {src}")
            else:
                win.title(f"Map: {pollutant}")
        except Exception:
            pass
        
        self._set_status("Plot complete.", level="INFO")

    def preview_data(self):
        if self.emissions_df is None:
            self._lazy_load_inputfile(show_preview=False)
        if self.emissions_df is None:
            self._notify('INFO', 'No Data', 'Load a SMOKE report / FF10 input file first to preview.')
            return
        # Select the raw parsed dataset (pre-filter, pre-aggregation) if available
        df_for_info = self._get_raw_for_summary()
        
        # Update the legacy text box with columns for quick glance (from raw dataset if present)
        try:
            # Only display original input columns, hiding derived columns such as FIPS/GRID_RC
            try:
                orig_cols = self.emissions_df.attrs.get('original_columns')  # type: ignore[attr-defined]
            except Exception:
                orig_cols = None
            cols_iter = [c for c in (orig_cols or list(df_for_info.columns)) if c in df_for_info.columns]
            buf = io.StringIO()
            buf.write("Report Columns (original only; units shown when available)\n")
            cols_with_units = []
            for c in cols_iter:
                unit = self.units_map.get(c)
                if unit:
                    cols_with_units.append(f"{c} ({unit})")
                else:
                    cols_with_units.append(str(c))
            buf.write(", ".join(cols_with_units))
            self.preview.delete('1.0', tk.END)
            self.preview.insert(tk.END, buf.getvalue())
            meta_text = self._last_attrs_summary or self._format_attrs_for_display(self.emissions_df)
            if meta_text:
                self.preview.insert(tk.END, "\n\nAttributes\n")
                self.preview.insert(tk.END, meta_text)
            self._append_loader_messages_to_preview()
            self._preview_has_data = True
        except Exception:
            pass
        # Open a popup window with a tabular view
        try:
            # Pass the raw df; the popup will filter to original columns as well
            self._open_preview_window(raw=df_for_info)
        except Exception as e:
            self._notify('ERROR', 'Preview Error', f'Failed to open preview table: {e}', exc=e)

    def _open_preview_window(self, raw: Optional[pd.DataFrame] = None):
        if not USING_TK:
            return
        df = raw if isinstance(raw, pd.DataFrame) else self.emissions_df
        if df is None:
            return
        # Restrict to only original input columns if available
        try:
            orig_cols = self.emissions_df.attrs.get('original_columns') if self.emissions_df is not None else None  # type: ignore[attr-defined]
            if isinstance(orig_cols, (list, tuple)) and orig_cols:
                keep_cols = [c for c in orig_cols if c in df.columns]
                if keep_cols:
                    df = df[keep_cols]
        except Exception:
            pass
        # Close existing preview if open
        if self._preview_win is not None:
            try:
                self._preview_win.destroy()
            except Exception:
                pass
            self._preview_win = None
        win = tk.Toplevel(self.root)
        win.title("Data Preview (first 200 rows)")
        try:
            # Base size ~1000x480 scaled
            base_w, base_h = 1000, 480
            w = int(max(800, min(1800, base_w * getattr(self, '_ui_scale', 1.0))))
            h = int(max(420, min(1200, base_h * getattr(self, '_ui_scale', 1.0))))
            win.geometry(f"{w}x{h}")
            try:
                self.root.minsize(800, 500)
            except Exception:
                pass
        except Exception:
            win.geometry("1000x480")
        self._preview_win = win
        def _on_close():
            try:
                win.destroy()
            finally:
                self._preview_win = None
        win.protocol("WM_DELETE_WINDOW", _on_close)

        # Top info label
        info = ttk.Label(win, text=f"Rows: {len(df):,}    Columns: {len(df.columns):,}")
        info.pack(side='top', anchor='w', padx=8, pady=4)

        # Summary controls frame
        sumfrm = ttk.Frame(win)
        sumfrm.pack(side='top', fill='x', padx=8, pady=(0, 6))
        ttk.Label(sumfrm, text='Summarize by:').pack(side='left')
        self.summary_group_var = tk.StringVar(value='county')
        # Options: county (FIPS), state (STATEFP from FIPS), scc (SCC), grid (GRID_RC)
        group_choices = ['county', 'state', 'scc', 'grid']
        self.summary_group_combo = ttk.Combobox(sumfrm, textvariable=self.summary_group_var, values=group_choices, width=12, state='readonly')
        self.summary_group_combo.pack(side='left', padx=(6, 12))
        ttk.Button(sumfrm, text='Preview Summary', command=lambda: self._on_preview_summary()).pack(side='left', padx=(0, 6))
        ttk.Button(sumfrm, text='Export CSV', command=lambda: self._on_export_summary()).pack(side='left')

        # Frame for tree + scrollbars
        frm = ttk.Frame(win)
        frm.pack(side='top', fill='both', expand=True)
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)

        # Treeview setup
        columns = [str(c) for c in df.columns]
        tree = ttk.Treeview(frm, columns=columns, show='headings')
        vsb = ttk.Scrollbar(frm, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(frm, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        # Get column headers with units (fix: avoid duplicate names)
        units_map = getattr(df, 'attrs', {}).get('units_map', {})
        columns_with_units = []
        for c in columns:
            unit = units_map.get(c, None)
            # Only append units for pollutant columns, and only if not already present
            if unit and c not in ('STATEFP', 'STATE_NAME', 'FIPS', 'GRID_RC', 'SCC', 'SCC_DESCRIPTION') and unit not in c:
                columns_with_units.append(f"{c} ({unit})")
            else:
                columns_with_units.append(c)
        
        # Determine reasonable column widths from a sample
        sample = df.head(200)
        def _fmt_cell(v):
            if pd.isna(v):
                return ""
            if isinstance(v, float):
                try:
                    return f"{v:.6g}"
                except Exception:
                    return str(v)
            return str(v)
        # Precompute width per column
        max_len = {}
        for c, c_unit in zip(columns, columns_with_units):
            header_len = len(c_unit)
            cell_max = header_len
            try:
                for v in sample[c].head(50):
                    cell_max = max(cell_max, len(_fmt_cell(v)))
            except Exception:
                pass
            max_len[c] = min(60, max(8, cell_max))  # char units
        # Configure headings and widths
        for c, c_unit in zip(columns, columns_with_units):
            tree.heading(c, text=c_unit)
            width_px = int(max_len[c] * 8 + 20)
            anchor = 'e' if pd.api.types.is_numeric_dtype(df[c]) else 'w'
            tree.column(c, width=width_px, anchor=anchor, stretch=True, minwidth=60)

        # Insert rows
        for i, (_, row) in enumerate(sample.iterrows()):
            values = [_fmt_cell(row[c]) for c in columns]
            tree.insert('', 'end', iid=str(i), values=values)

    def _get_raw_for_summary(self) -> Optional[pd.DataFrame]:
        try:
            if isinstance(self.raw_df, pd.DataFrame):
                return self.raw_df
            return self.emissions_df
        except Exception:
            return self.emissions_df

    def _ensure_ff10_grid_mapping(self, *, notify_success: bool = True) -> None:
        """Populate GRID_RC for FF10 point datasets once grid geometry is available."""
        if not (self._ff10_ready and isinstance(self.emissions_df, pd.DataFrame)):
            return
        if self.grid_gdf is None or not isinstance(self.grid_gdf, gpd.GeoDataFrame):
            return
        if self._ff10_grid_ready:
            return
        try:
            mapped = map_latlon2grd(self.emissions_df, self.grid_gdf)
        except ValueError as exc:
            self._notify('WARNING', 'FF10 Grid Mapping', str(exc))
            return
        except Exception as exc:
            self._notify('ERROR', 'FF10 Grid Mapping Failed', str(exc), exc=exc)
            return

        if not isinstance(mapped, pd.DataFrame) or 'GRID_RC' not in mapped.columns:
            self._notify('WARNING', 'FF10 Grid Mapping', 'Failed to assign GRID_RC values to FF10 points.')
            return

        self.emissions_df = mapped
        if isinstance(self.raw_df, pd.DataFrame):
            try:
                self.raw_df = map_latlon2grd(self.raw_df, self.grid_gdf)
            except Exception:
                # best-effort for raw preview; failures are non-critical here
                pass
        self._ff10_grid_ready = True
        self._invalidate_merge_cache()
        if notify_success:
            self._notify('INFO', 'FF10 Grid Mapping', 'Mapped FF10 point records to grid cells.', popup=False)

    def _build_summary(self, mode: str) -> pd.DataFrame:
        raw = self._get_raw_for_summary()
        if raw is None or not isinstance(raw, pd.DataFrame):
            raise ValueError('No data available to summarize.')
        # Determine pollutant columns from the raw dataframe
        try:
            pols = detect_pollutants(raw)
        except Exception:
            pols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if not pols:
            raise ValueError('No pollutant numeric columns to summarize.')
        lower_map = {c.lower(): c for c in raw.columns}
        mode_l = (mode or '').strip().lower()
        group_cols: List[str] = []
        # Prepare optional name enrichment columns
        add_cols: Dict[str, pd.Series] = {}
        # County
        source_type = None
        try:
            if isinstance(self.emissions_df, pd.DataFrame):
                source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type')
        except Exception:
            source_type = None
        if mode_l == 'county':
            if 'fips' in lower_map:
                key = lower_map['fips']
            elif 'region_cd' in lower_map:
                key = lower_map['region_cd']
            else:
                raise ValueError("Cannot summarize by county: Missing FIPS/region_cd column.")
            group_cols = [key]
            # Derive STATEFP and names when possible
            try:
                fips_series = raw[key].astype(str).str.zfill(6)
                add_cols['COUNTRY_ID'] = fips_series.str[0]
                add_cols['STATEFP'] = fips_series.str[1:3]
                add_cols['STATE_NAME'] = add_cols['STATEFP'].map(US_STATE_FIPS_TO_NAME)
            except Exception:
                pass
            # If COUNTY or COUNTY_NAME column exists in input, carry a representative name per FIPS
            try:
                county_name_col = lower_map.get('county name') or lower_map.get('county')
                if county_name_col and county_name_col in raw.columns:
                    # Take the most frequent name per FIPS to avoid duplicates
                    tmp_cn = raw[[key, county_name_col]].copy()
                    # Simple approach: first non-null per FIPS
                    name_map = tmp_cn.dropna(subset=[county_name_col]).drop_duplicates(subset=[key]).set_index(key)[county_name_col]
                    add_cols['COUNTY_NAME'] = raw[key].map(name_map)
            except Exception:
                pass
        # State (from first two digits of FIPS)
        elif mode_l == 'state':
            fips_col = None
            if 'fips' in lower_map:
                fips_col = lower_map['fips']
            elif 'region_cd' in lower_map:
                fips_col = lower_map['region_cd']
            if not fips_col or fips_col not in raw.columns:
                raise ValueError("Cannot summarize by state: FIPS/region_cd not available to derive STATEFP.")
            tmp = raw.copy()
            # FIPS is typically 6 digits (C+SS+CCC). We want SS (chars 1-3).
            # If it happens to be 5 digits (SSCCC), zfill(6) -> 0SSCCC, so 1-3 is still SS.
            tmp['STATEFP'] = tmp[fips_col].astype(str).str.strip().str.zfill(6).str[1:3]
            group_cols = ['STATEFP']
            raw = tmp
            # Add state name
            try:
                raw['STATE_NAME'] = raw['STATEFP'].map(US_STATE_FIPS_TO_NAME)
                add_cols['STATE_NAME'] = raw['STATE_NAME']
            except Exception:
                pass
        # SCC (optionally with description)
        elif mode_l == 'scc':
            scc_col = lower_map.get('scc')
            if not scc_col:
                raise ValueError("Cannot summarize by SCC: 'SCC' column not found.")
            desc_col = lower_map.get('scc description')
            # Ensure SCC and optional description are treated as strings
            try:
                raw[scc_col] = raw[scc_col].astype(str)
            except Exception:
                pass
            if desc_col and desc_col in raw.columns:
                try:
                    raw[desc_col] = raw[desc_col].astype(str)
                except Exception:
                    pass
            group_cols = [scc_col] + ([desc_col] if desc_col and desc_col in raw.columns else [])
        # Grid: summarize by pair (X Cell, Y Cell), not by GRID_RC
        elif mode_l == 'grid':
            self._ensure_ff10_grid_mapping(notify_success=False)
            xcell_col = lower_map.get('x cell')
            ycell_col = lower_map.get('y cell')
            if xcell_col in raw.columns and ycell_col in raw.columns:
                # Use existing X/Y Cell columns; ensure standardized names for output
                tmp = raw.copy()
                try:
                    tmp['X Cell'] = pd.to_numeric(tmp[xcell_col], errors='coerce').astype('Int64')
                    tmp['Y Cell'] = pd.to_numeric(tmp[ycell_col], errors='coerce').astype('Int64')
                except Exception:
                    tmp['X Cell'] = tmp[xcell_col]
                fips_series = raw[fips_col].astype(str).str.zfill(6)
                add_cols['COUNTRY_ID'] = fips_series.str[0]
                add_cols['STATEFP'] = fips_series.str[1:3]
                group_cols = ['X Cell', 'Y Cell']
            elif 'GRID_RC' in raw.columns:
                # Parse GRID_RC into X/Y Cell integers
                tmp = raw.copy()
                try:
                    parts = tmp['GRID_RC'].astype(str).str.split('_', n=1, expand=True)
                    tmp['ROW'] = pd.to_numeric(parts[0], errors='coerce').astype('Int64')
                    tmp['COL'] = pd.to_numeric(parts[1], errors='coerce').astype('Int64')
                except Exception:
                    # Fallback to strings if parsing failed
                    tmp['ROW'] = parts[0] if 'parts' in locals() else ''
                    tmp['COL'] = parts[1] if 'parts' in locals() else ''
                # Standardize to X/Y Cell output labels
                tmp['X Cell'] = tmp['COL']
                tmp['Y Cell'] = tmp['ROW']
                raw = tmp
                group_cols = ['X Cell', 'Y Cell']
            elif lower_map.get('row') in raw.columns and lower_map.get('col') in raw.columns:
                # Use ROW/COL but present as X/Y Cell
                rcol = lower_map.get('row')
                ccol = lower_map.get('col')
                tmp = raw.copy()
                try:
                    tmp['X Cell'] = pd.to_numeric(tmp[ccol], errors='coerce').astype('Int64')
                    tmp['Y Cell'] = pd.to_numeric(tmp[rcol], errors='coerce').astype('Int64')
                except Exception:
                    tmp['X Cell'] = tmp[ccol]
                    tmp['Y Cell'] = tmp[rcol]
                raw = tmp
                group_cols = ['X Cell', 'Y Cell']
            else:
                raise ValueError("Cannot summarize by grid: need X Cell/Y Cell, GRID_RC, or ROW/COL.")
        else:
            raise ValueError(f"Unknown summarize mode: {mode}")
        # Perform aggregation
        cols = [c for c in group_cols if c in raw.columns]
        if not cols:
            raise ValueError('No valid grouping columns available.')
        subset = raw[cols + pols]
        try:
            grouped = subset.groupby(cols, dropna=False, sort=False)
        except TypeError:
            grouped = subset.groupby(cols, sort=False)
        out = grouped.sum(numeric_only=True).reset_index()
        # Append enrichment columns if present
        try:
            for name, series in add_cols.items():
                if name not in out.columns and series is not None:
                    # reduce to one value per group by taking first matching value
                    # build mapping from grouping key(s) to name
                    if len(cols) == 1:
                        m = raw[[cols[0]]].assign(__val=series).drop_duplicates(subset=[cols[0]]).set_index(cols[0])['__val']
                        out[name] = out[cols[0]].map(m)
                    else:
                        # For multi-key groups (e.g., SCC+desc), skip enrichment here
                        pass
        except Exception:
            pass
        # County-mode specific cleanup and ordering
        if mode_l == 'county':
            # Normalize key column to 'FIPS'
            key_col = cols[0] if cols else None
            if key_col and key_col in out.columns and key_col != 'FIPS':
                try:
                    out = out.rename(columns={key_col: 'FIPS'})
                except Exception:
                    pass
            # Ensure FIPS is 6 characters (e.g. '037001' instead of '37001')
            if 'FIPS' in out.columns:
                try:
                    if 'COUNTRY_ID' in out.columns:
                        fips_s = out['FIPS'].astype(str)
                        cid_s = out['COUNTRY_ID'].astype(str)
                        out['FIPS'] = np.where(fips_s.str.len() == 5, cid_s + fips_s, fips_s.str.zfill(6))
                    else:
                        out['FIPS'] = out['FIPS'].astype(str).str.zfill(6)
                except Exception:
                    pass
            # Drop STATEFP if present
            for drop_col in ['STATEFP']:
                if drop_col in out.columns:
                    try:
                        out = out.drop(columns=[drop_col])
                    except Exception:
                        pass
            # Drop X/Y Cell columns if present (various casings)
            for c in list(out.columns):
                cl = str(c).lower()
                if cl in ('x cell', 'y cell'):
                    try:
                        out = out.drop(columns=[c])
                    except Exception:
                        pass
            # Reorder columns: FIPS, STATE_NAME, COUNTY_NAME, then the rest
            preferred = ['FIPS', 'STATE_NAME', 'COUNTY_NAME']
            present_pref = [c for c in preferred if c in out.columns]
            rest = [c for c in out.columns if c not in present_pref]
            try:
                out = out[present_pref + rest]
            except Exception:
                pass
    # State-mode specific cleanup and ordering
        elif mode_l == 'state':
            # Drop X/Y Cell columns if somehow present
            for c in list(out.columns):
                cl = str(c).lower()
                if cl in ('x cell', 'y cell'):
                    try:
                        out = out.drop(columns=[c])
                    except Exception:
                        pass
            # Reorder columns: STATEFP, STATE_NAME, then the rest
            preferred = ['STATEFP', 'STATE_NAME']
            present_pref = [c for c in preferred if c in out.columns]
            rest = [c for c in out.columns if c not in present_pref]
            try:
                out = out[present_pref + rest]
            except Exception:
                pass
        # Grid-mode cleanup
        elif mode_l == 'grid':
            # Drop GRID_RC if present and keep standardized X/Y Cell columns at the front
            if 'GRID_RC' in out.columns:
                try:
                    out = out.drop(columns=['GRID_RC'])
                except Exception:
                    pass
            preferred = ['X Cell', 'Y Cell']
            present_pref = [c for c in preferred if c in out.columns]
            rest = [c for c in out.columns if c not in present_pref]
            try:
                out = out[present_pref + rest]
            except Exception:
                pass
        return out

    def _on_preview_summary(self):
        try:
            mode = self.summary_group_var.get() if hasattr(self, 'summary_group_var') else 'county'
            df = self._build_summary(mode)
        except Exception as e:
            self._notify('ERROR', 'Summary Error', str(e), exc=e)
            return
        # Show summary top rows in a simple popup
        try:
            win = tk.Toplevel(self.root)
            win.title(f"Summary preview by {mode}")
            # dimensions
            win.geometry("900x500")
            lbl = ttk.Label(win, text=f"Rows: {len(df):,}    Columns: {len(df.columns):,}")
            lbl.pack(side='top', anchor='w', padx=8, pady=4)
            frm = ttk.Frame(win)
            frm.pack(side='top', fill='both', expand=True)
            frm.rowconfigure(0, weight=1)
            frm.columnconfigure(0, weight=1)
            columns = [str(c) for c in df.columns]
            tree = ttk.Treeview(frm, columns=columns, show='headings')
            vsb = ttk.Scrollbar(frm, orient='vertical', command=tree.yview)
            hsb = ttk.Scrollbar(frm, orient='horizontal', command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            tree.grid(row=0, column=0, sticky='nsew')
            vsb.grid(row=0, column=1, sticky='ns')
            hsb.grid(row=1, column=0, sticky='ew')
            # Build headings with units and compute widths to avoid squeezed headers
            headings = []
            for c in columns:
                label = c
                try:
                    unit = self.units_map.get(c) if hasattr(self, 'units_map') and isinstance(self.units_map, dict) else None
                    if unit and c not in ('STATEFP', 'STATE_NAME', 'FIPS', 'GRID_RC', 'SCC', 'SCC_DESCRIPTION') and unit not in c:
                        label = f"({unit})"
                        headings.append(f"{c} ({unit})")
                    else:
                        headings.append(c)
                except Exception:
                    pass

            # Determine widths from header and a small data sample
            sample = df.head(200)
            def _fmt(v):
                if pd.isna(v):
                    return ''
                if isinstance(v, float):
                    try:
                        return f"{v:.6g}"
                    except Exception:
                        return str(v)
                return str(v)
            max_len = {}
            for c, label in zip(columns, headings):
                # Wider base for long string columns like SCC description
                base_min = 18 if c.lower() in ('scc description','scc_description') else 12
                head_len = len(label)
                cell_max = head_len
                try:
                    for v in sample[c].head(50):
                        cell_max = max(cell_max, len(_fmt(v)))
                except Exception:
                    pass
                max_len[c] = min(80, max(base_min, cell_max))
            for c, label in zip(columns, headings):
                tree.heading(c, text=label)
                width_px = int(max_len[c] * 8 + 24)
                tree.column(c, width=width_px, stretch=True)
            # insert sample
            def _fmt(v):
                if pd.isna(v):
                    return ''
                if isinstance(v, float):
                    try:
                        return f"{v:.6g}"
                    except Exception:
                        return str(v)
                return str(v)
            for i, (_, row) in enumerate(sample.iterrows()):
                values = [_fmt(row[c]) for c in columns]
                tree.insert('', 'end', iid=str(i), values=values)
        except Exception:
            pass

    def _on_export_summary(self):
        try:
            mode = self.summary_group_var.get() if hasattr(self, 'summary_group_var') else 'county'
            df = self._build_summary(mode)
        except Exception as e:
            self._notify('ERROR', 'Summary Error', str(e), exc=e)
            return
        # Ask file path
        try:
            path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        except Exception as e:
            self._notify('ERROR', 'Save Dialog Error', str(e), exc=e)
            return
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            self._notify('INFO', 'Exported', f'Summary saved to {path}')
        except Exception as e:
            self._notify('ERROR', 'Export Error', f'Failed saving CSV: {e}', exc=e)