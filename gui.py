#!/usr/bin/env python3
"""
Native PySide6 GUI for SMKPLOT.

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

This module implements the Native Qt-based graphical user interface for the tool.
Refactored from gui_qt.py/gui_tk.py to remove Tkinter compatibility layer.

Original Author: tranhuy@email.unc.edu
"""

import os
import sys

# Increase recursion limit for complex matplotlb/cartopy operations
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

import copy
import logging
import threading
import traceback
import multiprocessing
from typing import Optional, List, Dict, Any, Union, Tuple
from functools import partial

import pandas as pd
import numpy as np

import geopandas as gpd
from shapely.geometry import Point, box
import pyproj
if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    # Import logging here if needed, or print to stderr
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    sys.stderr.write("WARNING: No DISPLAY detected. Running in headless mode (qt-test). No window will be shown.\n")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, 
    QFileDialog, QMessageBox, QProgressBar, QTabWidget, 
    QSplitter, QFrame, QSizePolicy, QScrollArea, QGridLayout,
    QMenu, QMenuBar, QStatusBar, QListWidget, QTextEdit,
    QLayout, QTreeWidget, QTreeWidgetItem, QStyle, QListView,
    QDockWidget, QToolBar, QDialog, QDialogButtonBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QGroupBox, QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import (
    Qt, Signal, Slot, QObject, QThread, QTimer, QSize, QEvent, QSettings
)
from PySide6.QtGui import (
    QAction, QIcon, QFont, QIntValidator, QDoubleValidator, 
    QTextCursor, QColor
)

import pandas as pd
import numpy as np
import geopandas as gpd
import pyproj
import shapely
import matplotlib
# matplotlib.use('qtagg') # Let MPL auto-detect
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib import cm as mplcm
import matplotlib.patches as mpatches

# Local imports
try:
    from config import (
        US_STATE_FIPS_TO_NAME, DEFAULT_INPUTS_INITIALDIR, DEFAULT_SHPFILE_INITIALDIR,
        DEFAULT_ONLINE_COUNTIES_URL
    )
    from utils import normalize_delim, is_netcdf_file
    from data_processing import (
        read_inputfile, read_shpfile, extract_grid, create_domain_gdf, 
        detect_pollutants, map_latlon2grd, merge_emissions_with_geometry, 
        filter_dataframe_by_range, filter_dataframe_by_values, get_emis_fips, 
        apply_spatial_filter
    )
    from plotting import _plot_crs, _draw_graticule as _draw_graticule_fn, create_map_plot
    from ncf_processing import get_ncf_dims, create_ncf_domain_gdf, read_ncf_grid_params
except ImportError:
    # Fallback for when running directly
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import *
    from utils import *
    from data_processing import *
    from plotting import *
    from ncf_processing import *

class LogHandler(logging.Handler):
    """Custom logging handler that emits a signal locally."""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.callback:
                self.callback(msg)
        except (RuntimeError, AttributeError):
            pass # Avoid crash if UI deleted during logging
        except Exception:
            pass

class TableWindow(QMainWindow):
    """Modern window to display a DataFrame in a searchable, sortable table."""
    def __init__(self, df, title="Data Preview", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 700)
        
        # UI Setup
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header Area
        header_layout = QHBoxLayout()
        info = f"Showing {min(len(df), 2000):,} of {len(df):,} rows | {len(df.columns)} Columns"
        self.info_lbl = QLabel(info)
        self.info_lbl.setStyleSheet("font-weight: bold; color: #4b5563;")
        header_layout.addWidget(self.info_lbl)
        
        header_layout.addStretch()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter rows...")
        self.search_input.setFixedWidth(200)
        self.search_input.textChanged.connect(self._filter_table)
        header_layout.addWidget(QLabel("Search:"))
        header_layout.addWidget(self.search_input)
        
        layout.addLayout(header_layout)
        
        # Table Setup
        df_show = df.head(2000) # Increased limit for modern systems
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setRowCount(len(df_show))
        self.table.setColumnCount(len(df_show.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df_show.columns])
        
        # Populate
        self._all_items = []
        for i, row in enumerate(df_show.itertuples(index=False)):
            row_items = []
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.table.setItem(i, j, item)
                row_items.append(item)
            self._all_items.append(row_items)
        
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setMinimumSectionSize(80)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget { border: 1px solid #d1d5db; background-color: #ffffff; gridline-color: #f3f4f6; }
            QHeaderView::section { background-color: #f9fafb; padding: 4px; border: 1px solid #e5e7eb; font-weight: bold; }
        """)
        
        layout.addWidget(self.table)
        
        # Footer
        footer_layout = QHBoxLayout()
        btn_copy = QPushButton("Copy Selected")
        btn_copy.clicked.connect(self._copy_selection)
        footer_layout.addWidget(btn_copy)
        
        footer_layout.addStretch()
        
        btn_export = QPushButton("Export to CSV")
        btn_export.setObjectName("primaryBtn")
        btn_export.clicked.connect(lambda: self.export_csv(df))
        footer_layout.addWidget(btn_export)
        
        layout.addLayout(footer_layout)

    def _filter_table(self, text):
        text = text.lower()
        for i in range(self.table.rowCount()):
            match = False
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                if item and text in item.text().lower():
                    match = True
                    break
            self.table.setRowHidden(i, not match)

    def _copy_selection(self):
        selected = self.table.selectedRanges()
        if not selected: return
        s = ""
        for r in range(selected[0].topRow(), selected[0].bottomRow() + 1):
            for c in range(selected[0].leftColumn(), selected[0].rightColumn() + 1):
                item = self.table.item(r, c)
                s += (item.text() if item else "") + "\t"
            s = s.strip() + "\n"
        QApplication.clipboard().setText(s)

    def export_csv(self, df):
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Export Successful", f"Data saved to: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save CSV: {str(e)}")

class MetadataWindow(QDialog):
    """Shows raw metadata (attrs) of a DataFrame."""
    def __init__(self, data_obj, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Metadata / Attributes")
        self.resize(600, 500)
        layout = QVBoxLayout(self)
        
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Monospace", 9))
        
        summary = "--- Dataset Metadata ---\n"
        if hasattr(data_obj, 'attrs'):
            for k, v in data_obj.attrs.items():
                if k == 'per_file_summaries': continue
                summary += f"{k}: {v}\n"
        else:
            summary += "No attributes found.\n"
            
        # Special handling for per-file summaries
        summaries = data_obj.attrs.get('per_file_summaries')
        if summaries:
            summary += "\n\n--- Individual File Statistics ---\n"
            for i, s in enumerate(summaries):
                summary += f"\nFile {i+1}: {s.get('source')}\n"
                pols = s.get('pollutants', {})
                if not pols:
                    summary += "  (No pollutant statistics found for this file)\n"
                for p, stats in pols.items():
                    summary += f"  - {p}:\n"
                    summary += f"    Sum: {stats['sum']:.4g}, Max: {stats['max']:.4g}, Mean: {stats['mean']:.4g}, Count: {stats['count']}\n"
                
                # Also list unique attributes of this file if they differ
                # (For now just show some relevant ones)
                f_attrs = s.get('attrs', {})
                if f_attrs:
                    summary += "    Metadata: "
                    meta_bits = []
                    for k in ['source_type', 'is_netcdf', 'units_map']:
                        if k in f_attrs: meta_bits.append(f"{k}={f_attrs[k]}")
                    summary += (", ".join(meta_bits) if meta_bits else "None") + "\n"

        if hasattr(data_obj, 'columns'):
            summary += "\n--- Columns ---\n"
            summary += "\n".join(list(data_obj.columns))
            
        self.text.setText(summary)
        layout.addWidget(self.text)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

class DetailedStatsWindow(QDialog):
    """Shows a detailed table of statistics per file and combined."""
    def __init__(self, emissions_df, pollutant, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Detailed Statistics: {pollutant}")
        self.resize(800, 400)
        layout = QVBoxLayout(self)
        
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QLabel
        
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        # Prepare data
        summaries = emissions_df.attrs.get('per_file_summaries', [])
        
        # Combined stats
        combined = {}
        try:
            # First try the currently matched data (filtered by SCC/PlotBy)
            # If we are in the main GUI, we might want the global stats of the DF
            vals = pd.to_numeric(emissions_df[pollutant], errors='coerce').dropna()
            if not vals.empty:
                combined = {
                    'source': 'TOTAL (Combined)',
                    'sum': float(vals.sum()),
                    'max': float(vals.max()),
                    'mean': float(vals.mean()),
                    'count': len(vals)
                }
        except Exception: pass
        
        rows = []
        for s in summaries:
            pstats = s.get('pollutants', {}).get(pollutant)
            if pstats:
                rows.append({
                    'source': os.path.basename(str(s.get('source'))),
                    'sum': pstats['sum'],
                    'max': pstats['max'],
                    'mean': pstats['mean'],
                    'count': pstats['count']
                })
        
        if combined:
            rows.append(combined)
            
        if not rows:
            layout.addWidget(QLabel(f"No individual file statistics available for {pollutant}.\nEnsure you have loaded multiple files via a list or configuration."))
        else:
            self.table.setRowCount(len(rows))
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(['Source', 'Sum', 'Max', 'Mean', 'Count'])
            
            for i, r in enumerate(rows):
                self.table.setItem(i, 0, QTableWidgetItem(r['source']))
                self.table.setItem(i, 1, QTableWidgetItem(f"{r['sum']:.4g}"))
                self.table.setItem(i, 2, QTableWidgetItem(f"{r['max']:.4g}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{r['mean']:.4g}"))
                self.table.setItem(i, 4, QTableWidgetItem(str(r['count'])))
            
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

class PlotWindow(QMainWindow):
    """Pop-out window for a specific plot."""
    def __init__(self, gdf, column, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Plot: {column}")
        self.resize(1000, 800)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        from plotting import create_map_plot, _label_colorbar
        ax = self.figure.add_subplot(111)
        try:
            ax.set_aspect('equal', adjustable='box')
        except: pass
        
        # Plotting parameters
        cmap = meta.get('cmap', 'viridis')
        use_log = meta.get('use_log', True)
        bins = meta.get('bins', [])
        unit = meta.get('unit', '')
        vmin = meta.get('vmin')
        vmax = meta.get('vmax')
        
        # Render the map
        create_map_plot(
            gdf=gdf,
            column=column,
            title=f"{column} Emissions",
            ax=ax,
            cmap_name=cmap,
            bins=bins,
            log_scale=use_log,
            unit_label=unit,
            crs_proj=gdf.crs,
            vmin=vmin,
            vmax=vmax
        )
        _label_colorbar(ax, unit)
        
        # Calculate stats for the title
        try:
            vals = pd.to_numeric(gdf[column], errors='coerce').dropna()
            if not vals.empty:
                s_min, s_max = vals.min(), vals.max()
                s_mean, s_sum = vals.mean(), vals.sum()
                stats_str = f"Min: {s_min:.4g}  Mean: {s_mean:.4g}  Max: {s_max:.4g}  Sum: {s_sum:.4g} {unit}"
                ax.set_title(f"{column} Emissions\n{stats_str}", fontsize=12)
        except: pass
            
        self.canvas.draw()

class NativeEmissionGUI(QMainWindow):
    """
    Main Application Window (Native PySide6 implementation).
    Inherits from QMainWindow but remains strictly compatible with
    legacy Tkinter-based entry points via manual signal handling.
    """
    # Signals for cross-thread communication
    notify_signal = Signal(str, str) # level, message
    update_log_signal = Signal(str)
    data_loaded_signal = Signal(list, bool) # pollutants, is_netcdf
    plot_ready_signal = Signal(object, str, dict) # gdf, column, meta
    
    def __init__(self, 
                 inputfile_path: Optional[str] = None, 
                 counties_path: Optional[str] = None, 
                 emissions_delim: Optional[str] = None, 
                 grid_name: Optional[str] = None, 
                 griddesc: Optional[str] = None, 
                 pollutant: Optional[str] = None, 
                 json_payload: Any = None,
                 cli_args: Any = None, 
                 app_version: str = "2.0"):
        super().__init__()

        # --- Monkey-patch for Graticule Recursion Guard ---
        # Since we cannot touch plotting.py directly, we wrap the function here.
        import plotting
        original_draw_graticule = plotting._draw_graticule
        
        def guarded_draw_graticule(*args, **kwargs):
            ax = args[0] if args else kwargs.get('ax')
            if ax is None: return original_draw_graticule(*args, **kwargs)
            
            # Use a specialized guard to prevent infinite recursion during zoom events
            if getattr(ax, '_smk_drawing_graticule', False):
                return None
            ax._smk_drawing_graticule = True
            try:
                # Force autoscale off during graticule drawing to prevent zoom loops
                original_autoscale = ax.get_autoscale_on()
                ax.set_autoscale_on(False)
                res = original_draw_graticule(*args, **kwargs)
                ax.set_autoscale_on(original_autoscale)
                return res
            finally:
                ax._smk_drawing_graticule = False
        
        plotting._draw_graticule = guarded_draw_graticule
        
        # Update local alias to ensure internal methods use the guarded version
        import sys
        module = sys.modules[__name__]
        setattr(module, '_draw_graticule_fn', guarded_draw_graticule)
        # --------------------------------------------------

        self.setWindowTitle(f"SMKPLOT v{app_version} (Native Qt) (Author: tranhuy@email.unc.edu)")
        self.resize(1600, 900)
        
        # --- State Variables ---
        # Handle being passed from smkplot.py (cli_args or json_payload as Namespace/dict)
        self.cli_args = cli_args or json_payload
        
        # We want _json_arguments to be a flat dict of parameters
        self._json_arguments = {}
        try:
            if isinstance(self.cli_args, dict):
                # If passed a dict, it might be the raw config or a param dict
                payload = self.cli_args.get('json_payload', self.cli_args)
                if isinstance(payload, dict):
                    # Try to flatten if it has an 'arguments' key
                    self._json_arguments = payload.get('arguments', payload).copy() if isinstance(payload.get('arguments'), dict) else payload.copy()
                else:
                    self._json_arguments = payload.__dict__.copy() if hasattr(payload, '__dict__') else {}
            elif self.cli_args is not None:
                # It's likely an argparse Namespace
                self._json_arguments = self.cli_args.__dict__.copy()
                # If it has a raw payload, merge in any missing keys
                raw_payload = getattr(self.cli_args, 'json_payload', None)
                if isinstance(raw_payload, dict):
                    flat_json = raw_payload.get('arguments', raw_payload)
                    if isinstance(flat_json, dict):
                        # Merge CLI values take precedence
                        merged = flat_json.copy()
                        merged.update(self._json_arguments)
                        self._json_arguments = merged
        except Exception as e:
            logging.warning(f"Parameter extraction failed: {e}. Using defaults.")
            self._json_arguments = {} 

        self.inputfile_path = inputfile_path or getattr(self.cli_args, 'filepath', None) or self._json_arguments.get('filepath')
        self.counties_path = counties_path or getattr(self.cli_args, 'county_shapefile', None) or self._json_arguments.get('county_shapefile')
        self.griddesc_path = griddesc or getattr(self.cli_args, 'griddesc', None) or self._json_arguments.get('griddesc')
        self.grid_name = grid_name or getattr(self.cli_args, 'gridname', None) or self._json_arguments.get('gridname')
        self.emissions_delim = emissions_delim or getattr(self.cli_args, 'delim', None)
        
        # Store initial pollutant request
        self.preselected_pollutant = pollutant or getattr(self.cli_args, 'pollutant_first', None) or getattr(self.cli_args, 'pollutant', None) or self._json_arguments.get('pollutant')
        
        # User selections
        self.input_files_list = [] # For multiple file support
        self.pollutants = []
        self.units_map = {}
        
        # DataFrames
        self.emissions_df = None
        self.raw_df = None
        self.grid_gdf = None # Equivalent to self.grid_gdf
        self.counties_gdf = None # Equivalent to self.counties_gdf
        self.overlay_gdf = None
        self.filter_gdf = None
        self._merged_gdf = None
        self._hover_enabled = True
        self._last_stats_calc = None
        
        # Syncing variables with gui_qt.py for higher fidelity logic replication
        self._loader_messages = []
        self._ff10_ready = False
        self._ff10_grid_ready = False
        self._merged_cache = {}
        self._plot_windows = []
        self._zoom_press = None
        self._zoom_cids = []
        self._base_view = None
        self.initial_filter_overlay = None
        self.filter_values = {}
        self.ncf_tdim = 1
        self.ncf_zdim = 1
        self._last_attrs_summary = ""

        # Animation / Time Nav State
        self._t_data_cache = None
        self._t_idx = 0
        self._is_showing_agg = False
        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(lambda: self._step_time(1))
        
        # Add a debouncer for plot title/stats updates to maintain responsiveness
        self._title_timer = QTimer()
        self._title_timer.setSingleShot(True)
        self._title_timer.timeout.connect(self._exec_title_update)
        self._current_ax_for_title = None

        # UI State
        self.class_bins_var = "" 
        self._last_loaded_delim_state = None
        self._scc_display_to_code = {} # Mapping for display-to-code filtering
        self._data_collection = None # Matplotlib collection
        self._table_window = None # To hold reference to summary table window
        
        # --- Apply Modern Theme ---
        self._apply_styles()

        # --- Initialize UI ---
        self._init_ui()
        
        # --- Logger Setup ---
        self.log_handler = LogHandler(self._append_log)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        self.log_handler.setFormatter(fmt)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)
        
        # Connect signals
        self.notify_signal.connect(self._handle_notification)
        self.update_log_signal.connect(self.log_text.append)
        self.data_loaded_signal.connect(self._post_load_update)
        self.plot_ready_signal.connect(self._render_plot_on_main)
        
        # --- Auto-Load if arguments present ---
        QTimer.singleShot(100, self._startup_load)

    def _apply_styles(self):
        """Apply a high-contrast, polished theme that ensures readability."""
        style = """
            /* Base Styles */
            QWidget { 
                font-family: 'Segoe UI', 'Roboto', 'Ubuntu', sans-serif; 
                font-size: 11px; 
                color: #212529; 
                background: transparent;
            }
            
            QMainWindow, QDialog, QMessageBox {
                background-color: #f0f2f5;
            }

            /* Containers */
            QGroupBox {
                font-weight: bold; 
                border: 1px solid #d1d5db;
                border-radius: 6px; 
                margin-top: 20px; 
                padding-top: 15px;
                background-color: #ffffff;
                color: #374151;
            }
            QGroupBox::title {
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
                color: #3b82f6;
            }
            
            /* Inputs - Guaranteed Contrast */
            QLineEdit, QComboBox, QSpinBox, QTextEdit, QPlainTextEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 3px 5px;
                min-height: 22px;
            }
            QLineEdit:focus, QComboBox:focus { 
                border: 1.5px solid #3b82f6; 
                background-color: #fcfdfe;
            }
            
            QComboBox::drop-down { border-left: 0px; }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #111827;
                selection-background-color: #3b82f6;
                selection-color: #ffffff;
                border: 1px solid #d1d5db;
            }

            /* Buttons */
            QPushButton {
                background-color: #ffffff;
                color: #374151;
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #f9fafb; border-color: #9ca3af; }
            QPushButton:pressed { background-color: #f3f4f6; }
            
            QPushButton#primaryBtn {
                background-color: #2563eb;
                color: #ffffff !important;
                border: 1px solid #1e3a8a;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton#primaryBtn:hover { background-color: #1d4ed8; }
            QPushButton#primaryBtn:disabled { background-color: #93c5fd; color: #f3f4f6; }

            /* Tabs */
            QTabWidget::pane { 
                border: 1px solid #d1d5db; 
                top: -1px; 
                background: #ffffff; 
                border-radius: 0 0 6px 6px; 
            }
            QTabBar::tab {
                background: #e5e7eb; border: 1px solid #d1d5db;
                padding: 6px 12px; margin-right: 1px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                color: #4b5563;
            }
            QTabBar::tab:selected { 
                background: #ffffff; border-bottom-color: #ffffff; 
                font-weight: bold; color: #2563eb; 
            }
            
            /* Logs & Progress */
            QTextEdit#log_text { background-color: #ffffff; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px; }
            QProgressBar {
                background-color: #f3f4f6; 
                border: 1px solid #d1d5db;
                border-radius: 10px; 
                text-align: center; 
                color: #1f2937;
                font-weight: bold;
                height: 18px;
            }
            QProgressBar::chunk { background-color: #3b82f6; border-radius: 8px; }
            
            /* Status Bar */
            QStatusBar { background-color: #ffffff; border-top: 1px solid #e5e7eb; color: #4b5563; }
        """
        self.setStyleSheet(style)
        if QApplication.instance():
            QApplication.instance().setStyleSheet(style)

    def _init_ui(self):
        """Initialize the complete UI layout with tightened margins and a modern look."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Use Splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # --- Left Control Panel ---
        left_panel = QWidget()
        left_panel.setMinimumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        self.control_layout = left_layout 
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        
        # Tabbed config
        self.tabs = QTabWidget()
        
        # Page 1: Data Source
        page_source = QWidget(); l_source = QVBoxLayout(page_source)
        l_source.setContentsMargins(6, 6, 6, 6)
        l_source.setSpacing(2)
        self._init_inputs_section(l_source)
        self._init_variable_section(l_source)
        l_source.addStretch()
        self.tabs.addTab(page_source, "Source")
        
        # Page 2: Filtering
        page_filter = QWidget(); l_filter = QVBoxLayout(page_filter)
        l_filter.setContentsMargins(6, 6, 6, 6)
        self._init_filter_section(l_filter)
        l_filter.addStretch()
        self.tabs.addTab(page_filter, "Filter")
        
        # Page 3: Map View
        page_visuals = QWidget(); l_visuals = QVBoxLayout(page_visuals)
        l_visuals.setContentsMargins(6, 6, 6, 6)
        l_visuals.setSpacing(2)
        self._init_plot_settings_section(l_visuals)
        
        # Reset View helper
        btn_reset = QPushButton("Reset View to All Data")
        btn_reset.setIcon(self.style().standardIcon(QStyle.SP_FileDialogToParent))
        btn_reset.clicked.connect(self.reset_home_view)
        l_visuals.addWidget(btn_reset)
        
        self._init_animation_section(l_visuals)
        l_visuals.addStretch()
        self.tabs.addTab(page_visuals, "View")
        
        # Page 4: Analysis
        page_stats = QWidget(); l_stats = QVBoxLayout(page_stats)
        l_stats.setContentsMargins(6, 6, 6, 6)
        self._init_summary_section(l_stats)
        self._init_stats_panel(l_stats)
        l_stats.addStretch()
        self.tabs.addTab(page_stats, "Stats")

        left_layout.addWidget(self.tabs)
        
        # Always-visible footer buttons
        footer = QFrame()
        footer.setFrameShape(QFrame.NoFrame)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(2, 4, 2, 4)
        footer_layout.setSpacing(10)
        
        self.btn_main_plot = QPushButton("GENERATE PLOT")
        self.btn_main_plot.setObjectName("primaryBtn")
        self.btn_main_plot.setMinimumHeight(40)
        self.btn_main_plot.clicked.connect(self.update_plot)
        footer_layout.addWidget(self.btn_main_plot, 1)

        btn_exp_all = QPushButton("Quick CSV")
        btn_exp_all.setToolTip("Export plot data to CSV immediately")
        btn_exp_all.clicked.connect(self.export_data)
        footer_layout.addWidget(btn_exp_all, 0)
        
        left_layout.addWidget(footer)
        
        btn_export_all = QPushButton("Export Tool")
        btn_export_all.setMinimumHeight(45)
        btn_export_all.clicked.connect(self.export_data)
        footer_layout.addWidget(btn_export_all, 1)
        
        left_layout.addWidget(footer)
        
        self.main_splitter.addWidget(left_panel)

        # --- Right Panel (Plot + Logs) ---
        right_splitter = QSplitter(Qt.Vertical)
        
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, plot_container)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        self.plot_controls_frame = QFrame()
        self.plot_controls_frame.setVisible(False)
        self.plot_controls_frame.setFrameShape(QFrame.StyledPanel)
        self.pc_layout = QHBoxLayout(self.plot_controls_frame)
        self.pc_layout.setContentsMargins(2, 2, 2, 2)
        plot_layout.addWidget(self.plot_controls_frame)

        right_splitter.addWidget(plot_container)
        
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setFrameShape(QFrame.NoFrame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False); self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(3)
        self.status_label = QLabel("Ready")
        
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(self.progress_bar)
        log_layout.addWidget(self.status_label)
        
        right_splitter.addWidget(log_group)
        right_splitter.setStretchFactor(0, 5)
        right_splitter.setStretchFactor(1, 1)
        
        self.main_splitter.addWidget(right_splitter)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 5)
        self.main_splitter.setSizes([550, 1050])
        
        main_layout.addWidget(self.main_splitter)

    def _init_inputs_section(self, parent_layout=None):
        group = QGroupBox("1. Input Data")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setSpacing(4)
        form = QFormLayout()
        form.setVerticalSpacing(4)
        form.setLabelAlignment(Qt.AlignRight)
        
        # Emissions
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Paste emissions file path(s) here...")
        self.txt_input.editingFinished.connect(self._on_input_path_edit)
        form.addRow("Emissions:", self.txt_input)
        
        btns = QHBoxLayout()
        btns.setSpacing(4)
        btn_load = QPushButton("Add File(s)")
        btn_load.clicked.connect(self.select_input_file)
        btns.addWidget(btn_load, 2)
        
        btn_preview = QPushButton("Preview")
        btn_preview.clicked.connect(self.preview_data)
        btns.addWidget(btn_preview, 1)
        
        btn_meta = QPushButton("Meta")
        btn_meta.clicked.connect(self.show_metadata)
        btns.addWidget(btn_meta, 1)

        btn_stats = QPushButton("Stats")
        btn_stats.clicked.connect(self.show_detailed_stats)
        btns.addWidget(btn_stats, 1)

        layout.addLayout(form)
        layout.addLayout(btns)
        
        # Settings 
        settings_form = QFormLayout()
        settings_form.setLabelAlignment(Qt.AlignRight)
        
        self.cmb_delim = QComboBox()
        self.cmb_delim.addItems(['auto', 'comma', 'semicolon', 'tab', 'pipe', 'space', 'other'])
        self.cmb_delim.currentTextChanged.connect(self._on_delim_toggle)
        
        delim_layout = QHBoxLayout()
        delim_layout.addWidget(self.cmb_delim)
        self.txt_custom_delim = QLineEdit()
        self.txt_custom_delim.setPlaceholderText("Other...")
        self.txt_custom_delim.setFixedWidth(50)
        self.txt_custom_delim.setVisible(False)
        delim_layout.addWidget(self.txt_custom_delim)
        settings_form.addRow("Delim:", delim_layout)
        
        # Counties
        self.txt_counties = QLineEdit()
        self.txt_counties.setPlaceholderText("Standard or online counties...")
        self.txt_counties.editingFinished.connect(self._on_counties_path_edit)
        settings_form.addRow("Counties:", self.txt_counties)
        
        cnt_btns = QHBoxLayout()
        btn_counties = QPushButton("Browse Shp")
        btn_counties.clicked.connect(self.select_county_file)
        cnt_btns.addWidget(btn_counties)
        
        self.cmb_online_year = QComboBox()
        self.cmb_online_year.addItems(['2020', '2023'])
        cnt_btns.addWidget(QLabel("Yr:"))
        cnt_btns.addWidget(self.cmb_online_year)
        btn_online = QPushButton("Fetch")
        btn_online.clicked.connect(self.use_online_counties)
        cnt_btns.addWidget(btn_online)
        settings_form.addRow("", cnt_btns)
        
        # GRIDDESC
        self.txt_griddesc = QLineEdit()
        self.txt_griddesc.setPlaceholderText("GRIDDESC path...")
        self.txt_griddesc.editingFinished.connect(self._on_griddesc_path_edit)
        settings_form.addRow("GRIDDESC:", self.txt_griddesc)
        
        grid_row = QHBoxLayout()
        btn_griddesc = QPushButton("Browse")
        btn_griddesc.clicked.connect(self.select_griddesc_file)
        grid_row.addWidget(btn_griddesc)
        
        self.cmb_gridname = QComboBox()
        self.cmb_gridname.addItem("Select Grid")
        self.cmb_gridname.currentTextChanged.connect(self._grid_name_changed)
        grid_row.addWidget(self.cmb_gridname, 1)
        settings_form.addRow("", grid_row)
        
        # NetCDF Controls
        self.ncf_frame = QFrame()
        ncf_lyt = QHBoxLayout(self.ncf_frame)
        ncf_lyt.setContentsMargins(0,0,0,0)
        self.cmb_ncf_layer = QComboBox()
        self.cmb_ncf_layer.currentIndexChanged.connect(lambda: self.load_input_file(self.input_files_list))
        self.cmb_ncf_time = QComboBox()
        self.cmb_ncf_time.currentIndexChanged.connect(lambda: self.load_input_file(self.input_files_list))
        ncf_lyt.addWidget(QLabel("Lay:"))
        ncf_lyt.addWidget(self.cmb_ncf_layer)
        ncf_lyt.addWidget(QLabel("TS:"))
        ncf_lyt.addWidget(self.cmb_ncf_time)
        self.ncf_frame.setVisible(False)
        settings_form.addRow("NCF Dims:", self.ncf_frame)
        
        extra_lyt = QHBoxLayout()
        self.spin_skip = QSpinBox()
        extra_lyt.addWidget(QLabel("Skip:"))
        extra_lyt.addWidget(self.spin_skip)
        self.txt_comment = QLineEdit("#")
        self.txt_comment.setFixedWidth(30)
        extra_lyt.addWidget(QLabel("Com:"))
        extra_lyt.addWidget(self.txt_comment)
        settings_form.addRow("Parse Opts:", extra_lyt)
        
        layout.addLayout(settings_form)
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_variable_section(self, parent_layout=None):
        group = QGroupBox("2. Variable Selection")
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setVerticalSpacing(4)
        layout.setLabelAlignment(Qt.AlignRight)
        
        pol_layout = QHBoxLayout()
        pol_layout.setSpacing(4)
        self.cmb_pollutant = QComboBox()
        self.cmb_pollutant.currentIndexChanged.connect(self._pollutant_changed)
        pol_layout.addWidget(self.cmb_pollutant, 3)
        
        self.txt_pol_search = QLineEdit()
        self.txt_pol_search.setPlaceholderText("Search...")
        self.txt_pol_search.setMinimumWidth(80)
        self.txt_pol_search.textChanged.connect(self._filter_pollutant_list)
        pol_layout.addWidget(self.txt_pol_search, 1)
        layout.addRow("Pollutant:", pol_layout)
        
        self.lbl_units = QLabel("Unit: -")
        self.lbl_units.setStyleSheet("color: #2563eb; font-weight: bold; font-size: 10px;")
        layout.addRow("", self.lbl_units)
        
        self.cmb_pltyp = QComboBox()
        self.cmb_pltyp.addItems(['auto', 'grid', 'county'])
        layout.addRow("Plot Type:", self.cmb_pltyp)
        
        self.cmb_proj = QComboBox()
        self.cmb_proj.addItems(['auto', 'wgs84', 'lcc'])
        layout.addRow("Projection:", self.cmb_proj)
        
        scc_layout = QHBoxLayout()
        scc_layout.setSpacing(4)
        self.cmb_scc = QComboBox()
        self.cmb_scc.setEditable(True)
        self.cmb_scc.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_scc.addItem("All SCC")
        self.cmb_scc.setEnabled(False)
        scc_layout.addWidget(self.cmb_scc, 3)
        
        self.txt_scc_search = QLineEdit()
        self.txt_scc_search.setPlaceholderText("Find...")
        self.txt_scc_search.setMinimumWidth(60)
        self.txt_scc_search.textChanged.connect(self._filter_scc_list)
        scc_layout.addWidget(self.txt_scc_search, 1)
        layout.addRow("SCC Filter:", scc_layout)
        
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_filter_section(self, parent_layout=None):
        group = QGroupBox("3. Filtering")
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setVerticalSpacing(4)
        layout.setLabelAlignment(Qt.AlignRight)
        
        shp_row = QHBoxLayout()
        shp_row.setSpacing(4)
        self.txt_filter_shp = QLineEdit()
        self.txt_filter_shp.setPlaceholderText("Selection Shapefile...")
        self.txt_filter_shp.editingFinished.connect(self._on_filter_path_edit)
        shp_row.addWidget(self.txt_filter_shp, 1)
        btn_shp = QPushButton("Browse")
        btn_shp.clicked.connect(self.browse_filter_shpfile)
        shp_row.addWidget(btn_shp, 0)
        layout.addRow("Shape:", shp_row)
        
        self.cmb_filter_col = QComboBox()
        self.cmb_filter_col.setEditable(True)
        layout.addRow("Column:", self.cmb_filter_col)
        
        self.txt_filter_val = QLineEdit()
        self.txt_filter_val.setPlaceholderText("Value1, Value2...")
        layout.addRow("Values:", self.txt_filter_val)
        
        range_layout = QHBoxLayout()
        self.txt_range_min = QLineEdit()
        self.txt_range_min.setPlaceholderText("Min")
        self.txt_range_max = QLineEdit()
        self.txt_range_max.setPlaceholderText("Max")
        range_layout.addWidget(self.txt_range_min)
        range_layout.addWidget(QLabel("to"))
        range_layout.addWidget(self.txt_range_max)
        layout.addRow("Range:", range_layout)
        
        self.cmb_filter_op = QComboBox()
        self.cmb_filter_op.addItems(['clipped', 'intersect', 'within', 'False'])
        self.cmb_filter_op.setCurrentText('False')
        layout.addRow("Spat Op:", self.cmb_filter_op)
        
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_plot_settings_section(self, parent_layout=None):
        group = QGroupBox("4. View Options")
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setVerticalSpacing(4)
        layout.setLabelAlignment(Qt.AlignRight)
        
        self.txt_bins = QLineEdit()
        self.txt_bins.setPlaceholderText("e.g. 0, 1, 10, 100")
        layout.addRow("Custom Bins:", self.txt_bins)
        
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(sorted([m for m in plt.colormaps() if not m.endswith('_r')]))
        self.cmb_cmap.setCurrentText('viridis')
        layout.addRow("Colormap:", self.cmb_cmap)
        
        ov_row = QHBoxLayout()
        ov_row.setSpacing(4)
        self.txt_overlay_shp = QLineEdit()
        self.txt_overlay_shp.setPlaceholderText("Roads, Cities, etc...")
        self.txt_overlay_shp.editingFinished.connect(self._on_overlay_path_edit)
        ov_row.addWidget(self.txt_overlay_shp, 1)
        btn_ov = QPushButton("Browse")
        btn_ov.clicked.connect(self.browse_overlay_shpfile)
        ov_row.addWidget(btn_ov, 0)
        layout.addRow("Overlay:", ov_row)
        
        # Limits
        lim_layout = QHBoxLayout()
        self.txt_rmin = QLineEdit(); self.txt_rmin.setPlaceholderText("Min")
        self.txt_rmax = QLineEdit(); self.txt_rmax.setPlaceholderText("Max")
        lim_layout.addWidget(self.txt_rmin)
        lim_layout.addWidget(QLabel("to"))
        lim_layout.addWidget(self.txt_rmax)
        layout.addRow("Fixed Scale:", lim_layout)
        
        # Toggles
        check_layout = QGridLayout()
        check_layout.setSpacing(10)
        
        self.chk_log = QCheckBox("Log Scale")
        self.chk_log.setChecked(True)
        self.chk_log.setToolTip("Scale data logarithmically (recommended for emissions)")
        
        self.chk_zoom = QCheckBox("Zoom to Data")
        self.chk_zoom.setChecked(True)
        self.chk_zoom.setToolTip("Auto-zoom to areas with valid emissions")
        
        self.chk_nan0 = QCheckBox("Fill NaN values with 0")
        self.chk_nan0.setToolTip("Fill NaN values with 0.0. This affects both the map visualization and the calculated statistics (Sum, Mean).")
        
        self.chk_graticule = QCheckBox("Graticule")
        self.chk_graticule.setChecked(True)
        self.chk_graticule.setToolTip("Show lat/lon grid lines")
        
        self.chk_rev_cmap = QCheckBox("Reverse CMap")
        self.chk_rev_cmap.setToolTip("Invert the colormap colors")
        
        self.chk_quadmesh = QCheckBox("Fast QuadMesh")
        self.chk_quadmesh.setChecked(True)
        self.chk_quadmesh.setToolTip("Dramatically faster plotting for large grids")
        
        check_layout.addWidget(self.chk_log, 0, 0)
        check_layout.addWidget(self.chk_zoom, 0, 1)
        check_layout.addWidget(self.chk_nan0, 1, 0)
        check_layout.addWidget(self.chk_graticule, 1, 1)
        check_layout.addWidget(self.chk_rev_cmap, 2, 0)
        check_layout.addWidget(self.chk_quadmesh, 2, 1)
        layout.addRow("Controls:", check_layout)
        
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_export_section(self, parent_layout=None):
        group = QGroupBox("5. Export")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setSpacing(6)
        
        btn_save = QPushButton("Save PNG")
        btn_save.clicked.connect(self.save_plot)
        layout.addWidget(btn_save, 1)
        
        btn_export = QPushButton("Export Data")
        btn_export.clicked.connect(self.export_data)
        layout.addWidget(btn_export, 1)
        
        btn_pop = QPushButton("Pop-out")
        btn_pop.setToolTip("Open current plot in a new resizable window")
        btn_pop.clicked.connect(self.pop_out_plot)
        layout.addWidget(btn_pop, 1)
        
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_animation_section(self, parent_layout=None):
        self.anim_group = QGroupBox("6. Animation / Time Nav")
        layout = QVBoxLayout(self.anim_group)
        self.anim_group.setVisible(False)
        
        self.lbl_anim_time = QLabel("Time: -")
        layout.addWidget(self.lbl_anim_time)
        
        nav_layout = QHBoxLayout()
        btn_prev = QPushButton("< Prev")
        btn_prev.clicked.connect(lambda: self._step_time(-1))
        btn_next = QPushButton("Next >")
        btn_next.clicked.connect(lambda: self._step_time(1))
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        layout.addLayout(nav_layout)
        
        agg_layout = QGridLayout()
        btn_tot = QPushButton("Total")
        btn_tot.clicked.connect(lambda: self._show_agg('total'))
        btn_avg = QPushButton("Avg")
        btn_avg.clicked.connect(lambda: self._show_agg('avg'))
        btn_max = QPushButton("Max")
        btn_max.clicked.connect(lambda: self._show_agg('max'))
        btn_min = QPushButton("Min")
        btn_min.clicked.connect(lambda: self._show_agg('min'))
        
        agg_layout.addWidget(btn_tot, 0, 0)
        agg_layout.addWidget(btn_avg, 0, 1)
        agg_layout.addWidget(btn_max, 1, 0)
        agg_layout.addWidget(btn_min, 1, 1)
        layout.addLayout(agg_layout)
        
        if parent_layout: parent_layout.addWidget(self.anim_group)
        else: self.control_layout.addWidget(self.anim_group)

    def _init_summary_section(self, parent_layout=None):
        group = QGroupBox("7. Table Summary")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setSpacing(6)
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("By:"))
        self.cmb_summary_mode = QComboBox()
        self.cmb_summary_mode.addItems(['county', 'state', 'scc', 'grid'])
        h_layout.addWidget(self.cmb_summary_mode, 1)
        layout.addLayout(h_layout)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        btn_sum_pre = QPushButton("Preview")
        btn_sum_pre.clicked.connect(self._on_preview_summary)
        btn_layout.addWidget(btn_sum_pre, 1)
        
        btn_sum_exp = QPushButton("Export")
        btn_sum_exp.clicked.connect(self._on_export_summary)
        btn_layout.addWidget(btn_sum_exp, 1)
        layout.addLayout(btn_layout)
        
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_stats_panel(self, parent_layout=None):
        """Panel to show stats of the current plot."""
        group = QGroupBox("8. Current View Statistics")
        layout = QGridLayout(group)
        
        self.lbl_stats_sum = QLabel("-")
        self.lbl_stats_max = QLabel("-")
        self.lbl_stats_mean = QLabel("-")
        self.lbl_stats_count = QLabel("-")
        
        for w in [self.lbl_stats_sum, self.lbl_stats_max, self.lbl_stats_mean, self.lbl_stats_count]:
            w.setStyleSheet("font-weight: bold; color: #2563eb;")
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            
        layout.addWidget(QLabel("Sum:"), 0, 0)
        layout.addWidget(self.lbl_stats_sum, 0, 1)
        layout.addWidget(QLabel("Max:"), 1, 0)
        layout.addWidget(self.lbl_stats_max, 1, 1)
        layout.addWidget(QLabel("Mean:"), 2, 0)
        layout.addWidget(self.lbl_stats_mean, 2, 1)
        layout.addWidget(QLabel("Count:"), 3, 0)
        layout.addWidget(self.lbl_stats_count, 3, 1)
        
        # New: Metadata and Detailed Stats buttons
        btn_box = QHBoxLayout()
        btn_meta = QPushButton("Meta Info")
        btn_meta.clicked.connect(self.show_metadata)
        btn_box.addWidget(btn_meta)
        
        btn_stats = QPushButton("Detailed Stats")
        btn_stats.clicked.connect(self.show_detailed_stats)
        btn_box.addWidget(btn_stats)
        
        layout.addLayout(btn_box, 4, 0, 1, 2)
        
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    # --- Logic Implementation ---
    
    @Slot(str, str)
    def _handle_notification(self, level, message):
        """Handle log messages from worker threads."""
        is_headless = QApplication.instance().platformName() == 'offscreen'
        
        if level == "ERROR":
            logging.error(message)
            self.progress_bar.setVisible(False)
            if not is_headless:
                QMessageBox.critical(self, "Error", message)
        elif level == "WARNING":
            self.status_label.setText(f"Warning: {message}")
            logging.warning(message)
            self.progress_bar.setVisible(False)
        else:
            self.status_label.setText(message)
            logging.info(message)
            # Pulse progress if busy
            busy_keywords = ["Loading", "Plotting", "Generating", "Computing", "Extracting"]
            stop_keywords = ["Ready", "finished", "Plotted", "available", "loaded", "complete"]
            
            if any(k in message for k in busy_keywords):
                self.progress_bar.setRange(0, 0) # Pulse
                self.progress_bar.setVisible(True)
            elif any(k in message for k in stop_keywords):
                self.progress_bar.setVisible(False)
            
    def _append_log(self, text):
        try:
            self.update_log_signal.emit(text)
        except RuntimeError:
            pass # Signal source deleted

    def _startup_load(self):
        """Called once after UI is shown to load CLI/JSON args."""
        # 1. First apply any generic settings from config (bins, projection, etc)
        self._apply_initial_settings()

        # 2. Trigger loading of paths
        if self.inputfile_path:
            self.load_input_file(self.inputfile_path)
            
        if self.griddesc_path:
            self.load_griddesc(self.griddesc_path)
            
        if self.counties_path:
             self._load_shapes()
        
        # 3. Load extra shapefiles if in config
        cli = getattr(self, '_json_arguments', {})
        
        # Overlay Shapefiles
        ov_shp = cli.get('overlay_shapefile')
        if ov_shp:
            # Handle both list (from YAML/argparse) and string
            if isinstance(ov_shp, list):
                self.txt_overlay_shp.setText("; ".join([str(x) for x in ov_shp]))
            else:
                self.txt_overlay_shp.setText(str(ov_shp))
            self._load_overlay_shpfile(ov_shp)
        
        # Filter Shapefiles
        flt_shp = cli.get('filter_shapefile') or cli.get('filtered_by_shp')
        if flt_shp:
            if isinstance(flt_shp, list):
                self.txt_filter_shp.setText("; ".join([str(x) for x in flt_shp]))
            else:
                self.txt_filter_shp.setText(str(flt_shp))
            self._load_filter_shpfile(flt_shp)
        
        # Spatial Filter Opt
        flt_opt = cli.get('filter_shapefile_opt') or cli.get('filtered_by_overlay')
        if flt_opt:
            self.cmb_filter_op.setCurrentText(str(flt_opt))

    def _apply_initial_settings(self):
        """Map _json_arguments keys to UI widgets."""
        cli = getattr(self, '_json_arguments', {})
        if not cli: return
        
        # 1. Delimiter
        delim = cli.get('delim')
        if delim:
            # Normalize common delimiters
            mapping = {',':'comma', ';':'semicolon', '\t':'tab', '|':'pipe', ' ':'space'}
            if delim in mapping:
                self.cmb_delim.setCurrentText(mapping[delim])
            else:
                self.cmb_delim.setCurrentText('other')
                self.txt_custom_delim.setText(delim)
        
        # 2. Filtering
        if cli.get('filtered_by_column') or cli.get('filter_col'): 
            self.cmb_filter_col.setCurrentText(str(cli.get('filtered_by_column') or cli.get('filter_col')))
        if cli.get('filtered_by_val') or cli.get('filter_val'): 
            self.txt_filter_val.setText(str(cli.get('filtered_by_val') or cli.get('filter_val')))
        if cli.get('filtered_by_op') or cli.get('filter_shapefile_opt'): 
            self.cmb_filter_op.setCurrentText(str(cli.get('filtered_by_op') or cli.get('filter_shapefile_opt')))
        
        # 3. Plotting Options
        # BINS
        bins_val = cli.get('bins')
        if bins_val:
             if isinstance(bins_val, list): 
                 self.txt_bins.setText(", ".join(map(str, bins_val)))
             else: 
                 self.txt_bins.setText(str(bins_val))
             
        # COLORMAP (Support both cmap and colormap)
        cmap_val = cli.get('cmap') or cli.get('colormap')
        if cmap_val:
             cmap = str(cmap_val)
             is_rev = cmap.endswith('_r')
             base_cmap = cmap[:-2] if is_rev else cmap
             # Try exact match first
             idx = self.cmb_cmap.findText(base_cmap)
             if idx >= 0:
                 self.cmb_cmap.setCurrentIndex(idx)
             else:
                 # Try case-insensitive or partial
                 idx = self.cmb_cmap.findText(base_cmap, Qt.MatchContains)
                 if idx >= 0: self.cmb_cmap.setCurrentIndex(idx)
             
             self.chk_rev_cmap.setChecked(is_rev)

        # 4. View State
        use_log = cli.get('log_scale') if cli.get('log_scale') is not None else cli.get('use_log')
        if use_log is not None: self.chk_log.setChecked(bool(use_log))
        
        zoom = cli.get('zoom_to_data') if cli.get('zoom_to_data') is not None else cli.get('zoom')
        if zoom is not None: self.chk_zoom.setChecked(bool(zoom))
        
        nan_val = cli.get('fill_nan')
        if nan_val is not None:
             self.chk_nan0.setChecked(True)
             # Logic in GUI is typically binary (use 0 or not), but we can store the val
             if hasattr(self, 'txt_nan_val'): self.txt_nan_val.setText(str(nan_val))

        low = cli.get('fixed_min') or cli.get('fixed_range_min') or cli.get('vmin')
        high = cli.get('fixed_max') or cli.get('fixed_range_max') or cli.get('vmax')
        if low is not None: self.txt_rmin.setText(str(low))
        if high is not None: self.txt_rmax.setText(str(high))

    def select_input_file(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Emissions File(s)", 
            self._json_arguments.get('initialdir', ''),
            "All Files (*);;CSV Files (*.csv);;NetCDF (*.nc *.ncf);;Text Files (*.txt)"
        )
        if files:
            self.load_input_file(files)

    def load_input_file(self, file_paths):
        """Orchestrate the loading of input files."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        self.input_files_list = file_paths
        
        # Update Input entry
        if len(file_paths) == 1:
            self.txt_input.setText(file_paths[0])
        else:
            self.txt_input.setText("; ".join(file_paths))
            
        # Start Worker Thread
        self._start_progress("Loading data...")
        
        # Use partial to pass args to thread
        threading.Thread(target=self._load_worker, args=(file_paths,), daemon=True).start()

    @Slot()
    def _on_delim_toggle(self, text):
        self.txt_custom_delim.setVisible(text == 'other')

    @Slot()
    def select_county_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Counties Shapefile", "", "Shapefiles (*.shp *.zip *.gpkg)")
        if f:
            self.counties_path = f
            self.txt_counties.setText(f)
            self._start_progress(f"Loading shapefile...")
            # Load in background
            threading.Thread(target=self._load_counties_worker, args=(f,), daemon=True).start()

    @Slot()
    def _on_input_path_edit(self):
        path = self.txt_input.text().strip()
        if not path: return
        if path != getattr(self, 'inputfile_path', None):
            paths = [p.strip() for p in path.split(';') if p.strip()]
            self.load_input_file(paths)

    @Slot()
    def _on_counties_path_edit(self):
        path = self.txt_counties.text().strip()
        if not path: return
        if path != getattr(self, 'counties_path', None):
            self.counties_path = path
            threading.Thread(target=self._load_counties_worker, args=(path,), daemon=True).start()

    @Slot()
    def _on_griddesc_path_edit(self):
        path = self.txt_griddesc.text().strip()
        if not path: return
        if path != getattr(self, 'griddesc_path', None):
            self.load_griddesc(path)

    @Slot()
    def use_online_counties(self):
        year = self.cmb_online_year.currentText()
        url = f"https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_county_500k.zip"
        self.notify_signal.emit("INFO", f"Fetching online counties for {year}...")
        self.counties_path = url
        self.txt_counties.setText(url)
        threading.Thread(target=self._load_counties_worker, args=(url,), daemon=True).start()

    @Slot()
    def _on_overlay_path_edit(self):
        path = self.txt_overlay_shp.text().strip()
        if not path:
             self.overlay_gdf = None
             return
        self._load_overlay_shpfile(path)

    @Slot()
    def _on_filter_path_edit(self):
        path = self.txt_filter_shp.text().strip()
        if not path:
            self.filter_gdf = None
            return
        self._load_filter_shpfile(path)

    def _load_counties_worker(self, path):
        try:
            gdf = read_shpfile(path, True)
            if gdf is not None:
                 self.counties_gdf = gdf
                 self._invalidate_merge_cache()
                 self.notify_signal.emit("INFO", f"Loaded counties: {len(gdf)} features")
        except Exception as e:
            self.notify_signal.emit("WARNING", f"Counties load error: {e}")
        finally:
            QTimer.singleShot(0, self._stop_progress)

    @Slot()
    def select_griddesc_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select GRIDDESC File", "", "All Files (*)")
        if f:
            self.load_griddesc(f)

    def load_griddesc(self, f):
        """Load and parse a GRIDDESC file."""
        if not f: return
        self.griddesc_path = f
        self.txt_griddesc.setText(f)
        try:
            from data_processing import extract_grid
            grid_names = sorted(extract_grid(f, grid_id=None), key=lambda s: s.lower())
            self.cmb_gridname.blockSignals(True)
            self.cmb_gridname.clear()
            self.cmb_gridname.addItem("Select Grid")
            self.cmb_gridname.addItems(grid_names)
            
            # Auto-select if grid_name provided
            auto_select_name = None
            if hasattr(self, 'grid_name') and self.grid_name:
                if self.grid_name in grid_names:
                    self.cmb_gridname.setCurrentText(self.grid_name)
                    auto_select_name = self.grid_name
                    # Sync to json_arguments immediately
                    self._json_arguments['gridname'] = self.grid_name
                    self._json_arguments['griddesc'] = self.griddesc_path
                else:
                    logging.warning(f"Requested grid '{self.grid_name}' not found in {f}")
            
            self.cmb_gridname.blockSignals(False)
            
            # Load grid geometry if a grid was auto-selected (signal won't fire because it was blocked)
            if auto_select_name:
                self._grid_name_changed(auto_select_name)
        except Exception as e:
            self.notify_signal.emit("ERROR", f"Failed to parse GRIDDESC: {e}")

    @Slot(str)
    @Slot(str)
    def _grid_name_changed(self, name):
        """Update grid definition when name selected in combo."""
        logging.info(f"Grid selection changed: {name}")
        self._invalidate_merge_cache()  
        if not name or name == "Select Grid":
            self.grid_gdf = None
            return
            
        try:
            from data_processing import create_domain_gdf
            self._start_progress(f"Building grid: {name}...")
            # We can run this in background if it's slow, but create_domain_gdf is usually fast enough for UI.
            # However, for 12US1 it might take a second. Let's stick to main thread for now as in the code I read,
            # or move to background if it was already backgrounded (it wasn't in the snippet I read).
            self.grid_gdf = create_domain_gdf(self.griddesc_path, name)
            if self.grid_gdf is not None:
                logging.info(f"Created domain GDF: {name} ({len(self.grid_gdf)} cells)")
                
                # Attach grid metadata for QuadMesh optimization (Mirroring ncf_processing logic)
                try:
                    from data_processing import extract_grid
                    coord_params, grid_params = extract_grid(self.griddesc_path, name)
                    _, xorig, yorig, xcell, ycell, ncols, nrows, _ = grid_params
                    # Generate proj_str (assuming spherical Earth for standard SMOKE grids if not specified)
                    # This is a safe assumption for almost all EMP grids.
                    _, p_alpha, p_beta, p_gamma, x_cent, y_cent = coord_params
                    proj_str = (
                        f"+proj=lcc +lat_1={p_alpha} +lat_2={p_beta} +lat_0={y_cent} "
                        f"+lon_0={x_cent} +a=6370000.0 +b=6370000.0 +x_0=0 +y_0=0 +units=m +no_defs"
                    )
                    self.grid_gdf.attrs['_smk_grid_info'] = {
                        'xorig': xorig, 'yorig': yorig, 'xcell': xcell, 'ycell': ycell,
                        'ncols': ncols, 'nrows': nrows, 'proj_str': proj_str
                    }
                except Exception as e:
                    logging.warning(f"Could not attach grid metadata for QuadMesh: {e}")

                # Immediately ensure FF10 mapping if possible
                self._ensure_ff10_grid_mapping()
                self.notify_signal.emit("INFO", f"Grid geometry loaded for '{name}'")
        except Exception as e:
            logging.error(f"Failed to create domain GDF: {e}")
            self.grid_gdf = None
            self.notify_signal.emit("ERROR", f"Failed to create grid shape: {e}")
        finally:
            self._stop_progress()

    def _default_conus_lcc_crs(self):
        """Standard CONUS LCC projection."""
        try:
            proj4 = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
            crs_lcc = pyproj.CRS.from_proj4(proj4)
            tf_fwd = pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs_lcc, always_xy=True)
            return crs_lcc, tf_fwd
        except Exception:
            return None, None

    def _get_plot_crs_info(self, df):
        """Determine projection and transformers for plotting."""
        choice = self.cmb_proj.currentText().lower()
        
        # If auto, use data crs if it exists and looks like it's already projected
        if choice == 'auto':
            if getattr(df, 'crs', None) is not None:
                try:
                    df_epsg = df.crs.to_epsg()
                    # For gridded data (has ROW/COL), prefer projected CRS; for county data, keep as-is
                    if df_epsg == 4326:  # WGS84
                        # Check if data is gridded or county-based
                        is_gridded = 'ROW' in df.columns or 'COL' in df.columns or 'GRID_RC' in df.columns
                        if is_gridded:
                            choice = 'lcc'  # Project gridded data
                        else:
                            return df.crs, None  # Keep WGS84 for ungridded
                    else:
                        # Already projected, use as-is
                        return df.crs, None
                except Exception as e:
                    logging.debug(f"CRS auto-detection failed: {e}")
                    return df.crs, None
            else:
                choice = 'lcc'
        
        if choice == 'wgs84':
            return pyproj.CRS.from_epsg(4326), None
            
        if choice == 'lcc':
            # Try to get from GRIDDESC first
            if self.griddesc_path and self.cmb_gridname.currentText() != "Select Grid":
                try:
                    from data_processing import extract_grid
                    coord_params, _ = extract_grid(self.griddesc_path, self.cmb_gridname.currentText())
                    _, p_alpha, p_beta, _, x_cent, y_cent = coord_params
                    proj4 = f"+proj=lcc +lat_1={p_alpha} +lat_2={p_beta} +lat_0={y_cent} +lon_0={x_cent} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
                    crs = pyproj.CRS.from_proj4(proj4)
                    return crs, pyproj.Transformer.from_crs(pyproj.CRS.from_epsg(4326), crs, always_xy=True)
                except Exception:
                    pass
            
            # Fallback to CONUS LCC
            crs, tf = self._default_conus_lcc_crs()
            return crs, tf
            
        return None, None

    # --- Loading Logic ---
    
    def _get_ncf_params(self):
        """Extract NetCDF time/layer parameters from UI."""
        params = {}
        # Layer
        lay_txt = self.cmb_ncf_layer.currentText().lower()
        if 'sum' in lay_txt: params['layer_op'] = 'sum'; params['layer_idx'] = None
        elif 'avg' in lay_txt or 'mean' in lay_txt: params['layer_op'] = 'mean'; params['layer_idx'] = None
        elif 'layer' in lay_txt:
            try:
                params['layer_idx'] = int(lay_txt.split()[-1]) - 1
                params['layer_op'] = 'select'
            except:
                params['layer_idx'] = 0; params['layer_op'] = 'select'
        else:
            params['layer_idx'] = 0; params['layer_op'] = 'select'

        # Time
        ts_txt = self.cmb_ncf_time.currentText().lower()
        if 'sum' in ts_txt: params['tstep_op'] = 'sum'; params['tstep_idx'] = None
        elif 'avg' in ts_txt or 'mean' in ts_txt: params['tstep_op'] = 'mean'; params['tstep_idx'] = None
        elif 'max' in ts_txt: params['tstep_op'] = 'max'; params['tstep_idx'] = None
        elif 'step' in ts_txt:
            try:
                params['tstep_idx'] = int(ts_txt.split()[-1]) - 1
                params['tstep_op'] = 'select'
            except:
                params['tstep_idx'] = 0; params['tstep_op'] = 'select'
        else:
            params['tstep_idx'] = 0; params['tstep_op'] = 'select'
        
        return params

    def _load_worker(self, paths):
        """Background worker for file loading."""
        try:
            # Determine params
            delim_type = self.cmb_delim.currentText()
            delim = None
            if delim_type == 'comma': delim = ','
            elif delim_type == 'semicolon': delim = ';'
            elif delim_type == 'tab': delim = '\t'
            elif delim_type == 'pipe': delim = '|'
            elif delim_type == 'space': delim = ' '
            elif delim_type == 'other': delim = self.txt_custom_delim.text()
            
            skip = self.spin_skip.value()
            comment = self.txt_comment.text()
            
            # Use read_inputfile from data_processing
            # Combine paths if multiple
            if len(paths) > 1:
                fpath = paths
            else:
                fpath = paths[0]
            
            # --- NetCDF Check ---
            is_nc = False
            first_file = paths[0]
            if is_netcdf_file(first_file):
                is_nc = True
                
            ncf_params = self._get_ncf_params() if is_nc else {}
                
            # Safely call read_inputfile
            # Note: We need a thread-safe notifier
            def worker_notify(lvl, msg):
                try:
                    self.notify_signal.emit(lvl, msg)
                except RuntimeError:
                    pass

            # Define localized read function to ensure consistency during stat re-calc
            # We must define it here to capture the correct delim/skip/comment context
            def _local_read(p, d=delim, s=skip, c=comment, np=ncf_params):
                return read_inputfile(
                    p, delim=d, skiprows=s, comment=c,
                    notify=lambda *_: None, lazy=True, workers=1, ncf_params=np
                )

            # Fix for RuntimeError at data_processing.py:803
            # Bypass library's parallel list-loading by looping manually in GUI thread
            if isinstance(fpath, list) and len(fpath) > 1:
                 worker_notify("INFO", f"Sequential load of {len(fpath)} files initiated in GUI...")
                 from data_processing import _normalize_input_result
                 dfs = []
                 raw_dfs = []
                 for p in fpath:
                     try:
                         # Still pass workers=1 to be safe if a #LIST file is inside the list
                         d, r = read_inputfile(
                             p, delim=delim, skiprows=skip, comment=comment,
                             notify=worker_notify, lazy=True, return_raw=True,
                             workers=1, ncf_params=ncf_params
                         )
                         if d is not None: dfs.append(d)
                         if r is not None: raw_dfs.append(r)
                     except Exception as e:
                         worker_notify("ERROR", f"Failed reading {p}: {e}")
                 
                 if not dfs:
                     df, raw = None, None
                 else:
                     df = pd.concat(dfs, ignore_index=True)
                     raw = pd.concat(raw_dfs, ignore_index=True) if raw_dfs else None
                     # Preserve first file's attributes if possible
                     if hasattr(df, 'attrs') and dfs[0].attrs:
                         df.attrs.update(dfs[0].attrs)
                     df, raw = _normalize_input_result((df, raw))
            else:
                 if len(paths) == 1:
                      worker_notify("INFO", f"Reading 1 file (sequential): {fpath}")

                 df, raw = read_inputfile(
                     fpath,
                     delim=delim,
                     skiprows=skip,
                     comment=comment,
                     notify=worker_notify,
                     lazy=True,
                     return_raw=True,
                     workers=1,
                     ncf_params=ncf_params
                 )
            
            if df is not None:
                # Flag for native rendering optimization (QuadMesh)
                try:
                    is_nc_flag = False
                    if isinstance(fpath, str) and is_netcdf_file(fpath):
                        is_nc_flag = True
                    elif isinstance(fpath, list) and all(is_netcdf_file(f) for f in fpath):
                        is_nc_flag = True
                    
                    source_type = df.attrs.get('source_type')
                    is_gridded_flag = (source_type == 'gridded_netcdf') or ('ROW' in df.columns and 'COL' in df.columns)
                    if is_nc_flag or is_gridded_flag:
                        df._smk_is_native = True
                        df.attrs['_smk_is_native'] = True
                except Exception:
                    pass

                # Re-calculate lost stats for UI (since processor was reverted)
                try:
                    summaries = []
                    file_list = fpath if isinstance(fpath, list) else [fpath]
                    if len(file_list) > 1:
                        for f in file_list:
                            if os.path.exists(f):
                                f_df, _ = _local_read(f)
                                if isinstance(f_df, pd.DataFrame):
                                     p_stats = {}
                                     for p in detect_pollutants(f_df):
                                         try:
                                             vals = pd.to_numeric(f_df[p], errors='coerce').dropna()
                                             if not vals.empty:
                                                 p_stats[p] = {
                                                     'sum': float(vals.sum()),
                                                     'max': float(vals.max()),
                                                     'mean': float(vals.mean()),
                                                     'count': int(len(vals))
                                                 }
                                         except Exception: pass
                                     summaries.append({'source': f, 'pollutants': p_stats})
                    else:
                        p_stats = {}
                        for p in detect_pollutants(df):
                            try:
                                vals = pd.to_numeric(df[p], errors='coerce').dropna()
                                if not vals.empty:
                                    p_stats[p] = {
                                        'sum': float(vals.sum()),
                                        'max': float(vals.max()),
                                        'mean': float(vals.mean()),
                                        'count': int(len(vals))
                                    }
                            except Exception: pass
                        summaries.append({'source': file_list[0], 'pollutants': p_stats})
                    df.attrs['per_file_summaries'] = summaries
                except Exception as e:
                    logging.warning(f"Stat re-calc failed: {e}")

                self.emissions_df = df
                self.raw_df = raw
                
                # Re-verify is_nc from loaded attributes (handle config files resolving to NCF)
                if not is_nc:
                    attrs = getattr(df, 'attrs', {})
                    if attrs.get('is_netcdf') or attrs.get('format') == 'netcdf':
                        is_nc = True
                    elif attrs.get('source_path') and is_netcdf_file(attrs.get('source_path')):
                        is_nc = True
                
                # Identify pollutants
                worker_notify("INFO", "Detecting pollutants...")
                cols = detect_pollutants(df)
                worker_notify("INFO", f"Pollutants detected: {len(cols)}")

                # --- Auto-Grid for NetCDF ---
                if is_nc:
                    try:
                        worker_notify("INFO", "Generating grid from NetCDF attributes...")
                        self.grid_gdf = create_ncf_domain_gdf(first_file if isinstance(first_file, str) else first_file[0])
                        if self.grid_gdf is not None and not self.grid_gdf.empty:
                            worker_notify("INFO", "Grid geometry loaded from NetCDF.")
                    except Exception as ge:
                        worker_notify("WARNING", f"Auto-grid from NetCDF failed: {ge}")
                
                # Update UI in main thread
                worker_notify("INFO", "Triggering UI update...")
                self.data_loaded_signal.emit(cols, is_nc)
                
        except Exception as e:
            self.notify_signal.emit("ERROR", f"Load failed: {e}")
            traceback.print_exc()
        finally:
            worker_notify("INFO", "Worker finished.")
            QTimer.singleShot(0, self._stop_progress)

    def _invalidate_merge_cache(self):
        """Clear the cached merged dataframes."""
        self._merged_cache = {}

    def _start_progress(self, msg="Processing...", pol=None):
        """Show progress bar and status message."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        if pol:
            self.status_label.setText(f"Plotting {pol}...")
        else:
            self.status_label.setText(msg)
        self.btn_main_plot.setEnabled(False)

    def _stop_progress(self):
        """Hide progress bar and restore UI."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready")
        self.btn_main_plot.setEnabled(True)

    def _ensure_ff10_grid_mapping(self, *, notify_success: bool = True) -> None:
        """Populate GRID_RC for FF10 point datasets once grid geometry is available."""
        if not (self._ff10_ready and isinstance(self.emissions_df, pd.DataFrame)):
            return
        if self.grid_gdf is None or not isinstance(self.grid_gdf, gpd.GeoDataFrame):
            return
        if self._ff10_grid_ready:
            return
        logging.info("FF10 point data detected. Mapping to grid cells...")
        try:
            mapped = map_latlon2grd(self.emissions_df, self.grid_gdf)
        except ValueError as exc:
            logging.warning(f"FF10 Grid Mapping: {exc}")
            return
        except Exception as exc:
            logging.error(f"FF10 Grid Mapping Failed: {exc}")
            return

        if not isinstance(mapped, pd.DataFrame) or 'GRID_RC' not in mapped.columns:
            logging.warning('Failed to assign GRID_RC values to FF10 points.')
            return

        self.emissions_df = mapped
        if isinstance(self.raw_df, pd.DataFrame):
            try:
                self.raw_df = map_latlon2grd(self.raw_df, self.grid_gdf)
            except Exception:
                pass
        self._ff10_grid_ready = True
        self._invalidate_merge_cache()
        if notify_success:
            self.status_label.setText("Mapped FF10 point records to grid cells.")

    def _plot_crs(self):
        """Determine projection from UI selection and data configuration.
        Mirroring gui_qt.py: Prefer LCC when available or selected as 'Auto'.
        """
        mode = 'auto'
        try:
            mode = self.cmb_proj.currentText().lower()
        except: pass

        if mode == 'wgs84':
            c = pyproj.CRS.from_epsg(4326)
            tf = pyproj.Transformer.from_crs(c, c, always_xy=True)
            return c, tf, tf

        # Helper to get the absolute default CONUS LCC
        def get_default_lcc():
            try:
                from plotting import _default_conus_lcc_crs
                return _default_conus_lcc_crs()
            except: return None, None, None

        # 1. Try to extract LCC from specific GRIDDESC if provided
        gd = self.griddesc_path
        gn = self.cmb_gridname.currentText()
        if gd and gn and gn != 'Select Grid':
            try:
                from plotting import _plot_crs as get_crs
                res = get_crs(gd, gn)
                # res is (crs, fwd, inv)
                if res and res[0]:
                    # If plotting._plot_crs returned WGS84 fallback, ignore it if we want LCC
                    if (mode == 'lcc' or mode == 'auto') and res[0].is_geographic:
                        pass 
                    else:
                        return res
            except Exception: pass
        
        # 2. Try to extract from currently loaded data attributes (Native LCC)
        if hasattr(self, 'emissions_df') and self.emissions_df is not None:
             try:
                 info = getattr(self.emissions_df, 'attrs', {}).get('_smk_grid_info')
                 if info and info.get('proj_str'):
                      p_str = info['proj_str']
                      crs = pyproj.CRS.from_user_input(p_str)
                      # If it's geographic, we might still want LCC for visualization
                      if (mode == 'lcc' or mode == 'auto') and crs.is_geographic:
                          pass
                      else:
                          tf_inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                          tf_fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                          return crs, tf_fwd, tf_inv
             except Exception: pass

        # 3. Fallback to default CONUS LCC if we want any LCC or Auto
        if mode in ('auto', 'lcc'):
             res = get_default_lcc()
             if res and res[0]:
                  return res

        # 4. Final Absolute Fallback to WGS84
        c = pyproj.CRS.from_epsg(4326)
        tf = pyproj.Transformer.from_crs(c, c, always_xy=True)
        return c, tf, tf
        return pyproj.CRS.from_epsg(4326), None, None

    @Slot(list, bool)
    def _post_load_update(self, pollutants, is_ncf):
        """Update UI after background loading finishes."""
        if self.emissions_df is None: return

        self._invalidate_merge_cache()
        self._ff10_grid_ready = False

        try:
            source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type')
        except Exception:
            source_type = None
        self._ff10_ready = bool(source_type == 'ff10_point')

        if self._ff10_ready and self.grid_gdf is not None:
            self._ensure_ff10_grid_mapping()

        # Find raw SCCs for widgets
        scc_data = []
        if self.raw_df is not None:
            scc_cols = [c for c in self.raw_df.columns if c.lower() in ['scc', 'scc code', 'scc_code']]
            if scc_cols:
                scc_col = scc_cols[0]
                desc_cols = [c for c in self.raw_df.columns if c.lower() in ['scc description', 'scc_description', 'scc_desc']]
                if desc_cols:
                    desc_col = desc_cols[0]
                    # Create display strings: "CODE | DESCRIPTION"
                    scc_df = self.raw_df[[scc_col, desc_col]].drop_duplicates().astype(str)
                    scc_df = scc_df[scc_df[scc_col].str.strip() != '']
                    scc_data = scc_df.apply(lambda x: f"{x[scc_col]} | {x[desc_col]}", axis=1).tolist()
                    self._scc_display_to_code = dict(zip(scc_data, scc_df[scc_col].tolist()))
                else:
                    scc_data = self.raw_df[scc_col].drop_duplicates().astype(str).tolist()
                    scc_data = [s for s in scc_data if s.strip() != '']
                    self._scc_display_to_code = {s: s for s in scc_data}

        self.cmb_scc.blockSignals(True)
        self.cmb_scc.clear()
        self.cmb_scc.addItem("All SCC")
        if scc_data:
            self.cmb_scc.addItems(sorted(scc_data))
            self.cmb_scc.setEnabled(True)
        else:
            self.cmb_scc.setEnabled(False)
        self.cmb_scc.blockSignals(False)

        # Use provided pollutants list or detect
        if not pollutants:
            pollutants = detect_pollutants(self.emissions_df)
            if not pollutants and isinstance(self.emissions_df, pd.DataFrame):
                pollutants = self.emissions_df.attrs.get('available_pollutants', [])
        
        self.pollutants = pollutants
        
        try:
            self.units_map = dict(self.emissions_df.attrs.get('units_map', {}))
        except Exception:
            self.units_map = {}
            
        if not self.pollutants:
            self.notify_signal.emit('WARNING', 'No pollutant columns detected.')
            return

        # Sort pollutants alphabetically
        self.pollutants = sorted(self.pollutants, key=lambda s: s.lower())

        self.cmb_pollutant.blockSignals(True)
        self.cmb_pollutant.clear()
        self.cmb_pollutant.addItems(self.pollutants)
        
        # Restore preselected or first pollutant
        if self.preselected_pollutant in self.pollutants:
            self.cmb_pollutant.setCurrentText(self.preselected_pollutant)
        elif self.pollutants:
            self.cmb_pollutant.setCurrentIndex(0)
        self.cmb_pollutant.blockSignals(False)

        # Update metadata summary
        if hasattr(self.emissions_df, 'attrs'):
            self._last_attrs_summary = "\n".join([f"{k}: {v}" for k, v in self.emissions_df.attrs.items()])
        
        # Success status
        self.status_label.setText(f"Loaded {len(self.emissions_df):,} rows, {len(self.pollutants)} pollutants.")
        
        self.ncf_frame.setVisible(is_ncf)
        self.plot_controls_frame.setVisible(is_ncf)
        if is_ncf:
            # Populate NCF dimensions if available
            fname = self.input_files_list[0]
            try:
                dims = get_ncf_dims(fname)
                # Layers
                n_lay = dims.get('n_layers', 1)
                lay_items = ['Sum All', 'Avg All'] + [f"Layer {i+1}" for i in range(n_lay)]
                self.cmb_ncf_layer.blockSignals(True)
                self.cmb_ncf_layer.clear()
                self.cmb_ncf_layer.addItems(lay_items)
                self.cmb_ncf_layer.blockSignals(False)
                # Time
                n_ts = dims.get('n_tsteps', 1)
                ts_items = ['Sum All', 'Avg All', 'Max'] + [f"Step {i+1}" for i in range(n_ts)]
                self.cmb_ncf_time.blockSignals(True)
                self.cmb_ncf_time.clear()
                self.cmb_ncf_time.addItems(ts_items)
                self.cmb_ncf_time.blockSignals(False)

                # Update GRIDDESC display to show NCF source
                try:
                    _, gp = read_ncf_grid_params(fname)
                    self.txt_griddesc.setText("(NetCDF Attributes)")
                    self.cmb_gridname.blockSignals(True)
                    self.cmb_gridname.clear()
                    self.cmb_gridname.addItem(gp[0])
                    self.cmb_gridname.setCurrentIndex(0)
                    self.cmb_gridname.blockSignals(False)
                except Exception: pass
            except Exception as e:
                logging.error(f"Failed to get NCF dims: {e}")

        # Trigger initial filter column population
        if self.raw_df is not None:
            raw_cols = sorted([str(c) for c in self.raw_df.columns])
            self.cmb_filter_col.clear()
            self.cmb_filter_col.addItems(raw_cols)
            
            # SCC Detection & Mapping
            scc_col = next((c for c in raw_cols if c.lower() in ['scc', 'scc code', 'scc_code']), None)
            desc_cols = [c for c in raw_cols if c.lower() in ['scc description', 'scc_description']]
            desc_col = desc_cols[0] if desc_cols else None
            
            self._scc_display_to_code = {}
            if scc_col:
                try:
                    df_scc = self.raw_df[[scc_col]].copy()
                    df_scc[scc_col] = df_scc[scc_col].astype(str).str.strip()
                    if desc_col:
                        df_scc[desc_col] = self.raw_df[desc_col].astype(str).str.strip()
                        # Create "Code - Description" string
                        df_scc['display'] = df_scc[scc_col] + " - " + df_scc[desc_col]
                    else:
                        df_scc['display'] = df_scc[scc_col]
                        
                    unique_sccs = df_scc.drop_duplicates().sort_values(scc_col)
                    self._scc_display_to_code = { row.display: row[scc_col] for row in unique_sccs.itertuples() }
                    
                    self.cmb_scc.blockSignals(True)
                    self.cmb_scc.clear()
                    self.cmb_scc.addItem("All SCC")
                    self.cmb_scc.addItems(sorted(self._scc_display_to_code.keys()))
                    self.cmb_scc.setEnabled(True)
                    self._scc_full_list = None # Reset search cache
                    self.cmb_scc.blockSignals(False)
                except Exception as e:
                    logging.warning(f"SCC mapping failed: {e}")
                    self.cmb_scc.clear()
                    self.cmb_scc.addItem("All SCC")
                    self.cmb_scc.setEnabled(False)
            else:
                self.cmb_scc.clear()
                self.cmb_scc.addItem("All SCC")
                self.cmb_scc.setEnabled(False)

        # Set plot type from config (validation happens in _merged() during plot)
        # Following gui_qt.py pattern: just read the config, don't try to validate/correct
        requested_pltyp = self._json_arguments.get('pltyp', 'auto')
        if requested_pltyp in ['auto', 'grid', 'county']:
            self.cmb_pltyp.setCurrentText(requested_pltyp)

        # Initialize Projection from config
        requested_proj = self._json_arguments.get('projection', 'lcc')
        if requested_proj in ['auto', 'wgs84', 'lcc']:
            self.cmb_proj.setCurrentText(requested_proj)

        logging.info(f"Loaded {len(self.emissions_df)} rows. Found {len(pollutants)} pollutants.")
        
        # Load counties if configured
        if self.counties_path and self.counties_gdf is None:
             self._load_shapes()

        # Auto-select and plot if configured
        preselected = getattr(self, 'preselected_pollutant', None)
        if not preselected:
            cli = getattr(self, '_json_arguments', {})
            preselected = cli.get('pollutant')

        if preselected:
            if isinstance(preselected, list): 
                preselected = preselected[0]
            elif isinstance(preselected, str) and ',' in preselected:
                preselected = preselected.split(',')[0].strip()
                
            if preselected in pollutants:
                logging.info(f"Auto-selecting pollutant: {preselected}")
                self.cmb_pollutant.setCurrentText(preselected)
                
                # Trigger plot after a brief delay to allow UI to settle
                QTimer.singleShot(500, self.update_plot)

    def _load_shapes(self):
        """Load county shapefiles."""
        threading.Thread(target=self._shape_worker, daemon=True).start()
        
    def _shape_worker(self):
        try:
             gdf = read_shpfile(self.counties_path)
             if gdf is not None:
                 self.counties_gdf = gdf
                 self.notify_signal.emit("INFO", f"Loaded counties: {len(gdf)} features")
        except Exception as e:
            self.notify_signal.emit("WARNING", f"Shapefile load error: {e}")

    def _pollutant_changed(self):
        # Update units label if metadata available
        pol = self.cmb_pollutant.currentText()
        unit = self.units_map.get(pol, "-")
        self.lbl_units.setText(f"Unit: {unit}")
        # Invalidate animation cache if pollutant changed
        self._t_data_cache = None
        self._t_idx = 0
        self._is_showing_agg = False
        # Hide animation controls when switching pollutants
        if hasattr(self, 'anim_group') and self.anim_group:
            self.anim_group.setVisible(False)
        # Keep plot_controls_frame visible if NCF data is loaded
        # (will be reshown if needed by _post_load_update)

    def _ensure_time_data(self):
        if self._t_data_cache is not None: return True
        pol = self.cmb_pollutant.currentText()
        if not pol or self.emissions_df is None: return False
        
        try:
            from ncf_processing import get_ncf_animation_data
            df = self.emissions_df
            # Prepare indices for ALL cells in plot
            r_col = 'ROW'; c_col = 'COL'
            if 'ROW' not in df.columns and 'ROW_x' in df.columns: r_col = 'ROW_x'
            if 'COL' not in df.columns and 'COL_x' in df.columns: c_col = 'COL_x'
             
            # 0-based indices
            try:
                rows = df[r_col].fillna(0).astype(int).values - 1
                cols = df[c_col].fillna(0).astype(int).values - 1
            except Exception as e:
                self.notify_signal.emit("WARNING", f"Index extraction failed: {e}")
                return False
            
            # Layer settings
            l_idx = 0; l_op = 'select'
            lay_txt = self.cmb_ncf_layer.currentText()
            if "Sum" in lay_txt: l_op = 'sum'
            elif "Avg" in lay_txt: l_op = 'mean'
            elif "Layer" in lay_txt:
                try: l_idx = int(lay_txt.split()[-1]) - 1
                except: pass
                
            effective_path = ""
            if hasattr(self, 'input_files_list') and self.input_files_list:
                effective_path = self.input_files_list[0]
                
            stack_groups_path = None
            if self.emissions_df is not None:
                attrs = getattr(self.emissions_df, 'attrs', {})
                if 'proxy_ncf_path' in attrs: effective_path = attrs['proxy_ncf_path']
                elif 'original_ncf_path' in attrs: effective_path = attrs['original_ncf_path']
                stack_groups_path = attrs.get('stack_groups_path')

            if not effective_path or not os.path.exists(effective_path):
                self.notify_signal.emit("ERROR", "Could not locate original NetCDF source for animation.")
                return False

            self.notify_signal.emit("INFO", "Loading full time series for animation...")
            res = get_ncf_animation_data(
                effective_path,
                pol, 
                rows.tolist(), 
                cols.tolist(), 
                layer_idx=l_idx, 
                layer_op=l_op,
                stack_groups_path=stack_groups_path
            )
            
            if res:
                vals = res['values']
                res['tot_val'] = np.sum(vals, axis=0)
                res['avg_val'] = np.mean(vals, axis=0)
                res['mx_val'] = np.nanmax(vals, axis=0)
                res['mn_val'] = np.nanmin(vals, axis=0)
                
                # Global scale
                res['vmin'] = np.nanmin(vals)
                res['vmax'] = np.nanmax(vals)
                
                self._t_data_cache = res
                self._t_idx = 0
                self._is_showing_agg = False
                self.notify_signal.emit("INFO", "Animation data loaded.")
                return True
        except Exception as e:
            self.notify_signal.emit("ERROR", f"Animation prep failed: {e}")
        return False

    def _step_time(self, delta):
        # We need to distinguish if we are fresh or in an aggregate view
        is_fresh = self._t_data_cache is None
        is_agg = self._is_showing_agg
        
        if not self._ensure_time_data(): return
        cache = self._t_data_cache
        n_steps = len(cache['times'])
        
        if n_steps <= 1:
            self.notify_signal.emit("INFO", "Animation: Only 1 time step available.")
            return

        # If we just loaded (fresh) or were showing an aggregate,
        # 'Next' should go to the first step (0), 'Prev' to the last (n-1).
        if is_fresh or is_agg:
            if delta > 0: self._t_idx = 0
            else: self._t_idx = n_steps - 1
        else:
            self._t_idx = (self._t_idx + delta) % n_steps
            
        self._is_showing_agg = False
        self._update_view(cache['values'][self._t_idx], cache['times'][self._t_idx])

    def _show_agg(self, mode):
        if not self._ensure_time_data(): return
        cache = self._t_data_cache
        self._is_showing_agg = True
        if mode == 'total':
            self._update_view(cache['tot_val'], "Total (Sum)")
        elif mode == 'avg':
            self._update_view(cache['avg_val'], "Average (Mean)")
        elif mode == 'max':
            self._update_view(cache['mx_val'], "Max All Time")
        elif mode == 'min':
            self._update_view(cache['mn_val'], "Min All Time")

    def _on_ts_view_click(self, op_mode):
        """Extract time series for all cells in current view."""
        if self._merged_gdf is None or not 'ROW' in self._merged_gdf.columns:
             self.notify_signal.emit("WARNING", "No gridded data available.")
             return
             
        # 1. Get Bounds
        ax = self.figure.axes[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # 2. Filter GDF
        gdf = self._merged_gdf
        try:
            # Use spatial index for speed
            from shapely.geometry import box as shp_box
            bbox = shp_box(xlim[0], ylim[0], xlim[1], ylim[1])
            
            # Fast filter
            cand_idx = []
            sindex = getattr(gdf, 'sindex', None)
            if sindex:
                cand_idx = list(sindex.intersection(bbox.bounds))
                # Refine
                sub = gdf.iloc[cand_idx]
                sub = sub[sub.intersects(bbox)]
            else:
                sub = gdf[gdf.intersects(bbox)]
                
            if sub.empty:
                 self.notify_signal.emit("WARNING", "No cells in current view.")
                 return
                 
            self.notify_signal.emit("INFO", f"Extracting TS for {len(sub)} cells ({op_mode})...")
            
            # 3. Get ROW/COL lists
            rows = sub['ROW'].values.astype(int)
            cols = sub['COL'].values.astype(int)
            
            # 4. Exec (Thread or Timer)
            # Layer/Op
            l_idx = 0; l_op = 'select'
            lay_txt = self.cmb_ncf_layer.currentText()
            if "Sum" in lay_txt: l_op = 'sum'
            elif "Avg" in lay_txt: l_op = 'mean'
            elif "Layer" in lay_txt:
                 try: l_idx = int(lay_txt.split()[-1]) - 1
                 except: pass

            QTimer.singleShot(0, lambda: self._exec_ts_view(rows, cols, l_idx, l_op, op_mode))
            
        except Exception as e:
            self.notify_signal.emit("ERROR", f"View extraction failed: {e}")

    def _exec_ts_view(self, rows, cols, l_idx, l_op, agg_op):
         try:
             # Run extraction
             effective_path = self.input_files_list[0]
             stack_groups_path = None
             if self.emissions_df is not None:
                 attrs = getattr(self.emissions_df, 'attrs', {})
                 if 'proxy_ncf_path' in attrs: effective_path = attrs['proxy_ncf_path']
                 elif 'original_ncf_path' in attrs: effective_path = attrs['original_ncf_path']
                 stack_groups_path = attrs.get('stack_groups_path')

             from ncf_processing import get_ncf_timeseries
             res = get_ncf_timeseries(
                effective_path,
                self.cmb_pollutant.currentText(),
                (rows-1).tolist(), (cols-1).tolist(),
                layer_idx=l_idx, layer_op=l_op, op=agg_op,
                stack_groups_path=stack_groups_path
             )
             
             if res:
                 self._show_ts_window(res, f"View ({len(rows)} cells) - {agg_op.upper()}")
                 self.notify_signal.emit("INFO", "Time Series Plotted.")
             else:
                 self.notify_signal.emit("WARNING", "No data found.")
         except Exception as e:
             self.notify_signal.emit("ERROR", f"TS Extraction failed: {e}")

    def _update_view(self, new_vals, time_lbl):
        """Update the plot for a new time step or aggregate."""
        try:
             if not self.figure.axes: return
             ax = self.figure.axes[0]
             
             # Locate collection
             coll = None
             for c in ax.collections:
                 if getattr(c, 'get_array', lambda: None)() is not None:
                     coll = c
                     break
             
             if coll:
                  # 1. Handle Limits (Especially for LogNorm safety)
                  if self._t_data_cache:
                      c_vmin = self._t_data_cache.get('vmin', 1e-20)
                      c_vmax = self._t_data_cache.get('vmax', 1.0)
                      if isinstance(coll.norm, LogNorm):
                          if c_vmin <= 0: c_vmin = 1e-20
                          if c_vmax <= c_vmin: c_vmax = c_vmin * 10.0
                          coll.norm.vmin = c_vmin
                          coll.norm.vmax = c_vmax
                      coll.set_clim(c_vmin, c_vmax)

                  # 2. Set Data
                  coll.set_array(new_vals)
                  ax._smk_current_vals = new_vals
                  
                  # 3. Update Colorbar
                  if hasattr(coll, 'colorbar') and coll.colorbar:
                      try: coll.colorbar.update_normal(coll)
                      except: pass
                  
                  # 4. Refresh Title Statistics
                  ax._smk_time_lbl = time_lbl
                  self._update_plot_title(ax, immediate=True)
                  
                  self.canvas.draw_idle()
                  
                  # 5. Update Status Labels
                  n_steps = len(self._t_data_cache['times']) if self._t_data_cache else 1
                  if "Sum" in str(time_lbl) or "Mean" in str(time_lbl):
                      self.lbl_anim_status.setText(str(time_lbl))
                  else:
                      self.lbl_anim_status.setText(f"{time_lbl} ({self._t_idx+1}/{n_steps})")
        except Exception as e:
            logging.error(f"Update view failed: {e}")

    def _update_plot_title(self, ax, immediate=False):
        """Update the plot title with dynamic statistics. Debounced by default for responsiveness."""
        self._current_ax_for_title = ax
        if immediate:
            self._exec_title_update()
        else:
            # Start/Restart the timer to debounce rapid view changes (pan/zoom)
            self._title_timer.start(300)

    def _exec_title_update(self):
        """Perform the actual title update and statistics calculation."""
        ax = self._current_ax_for_title
        if ax is None or not hasattr(ax, 'figure'):
            return
            
        try:
            pol = self.cmb_pollutant.currentText()
            unit = self.units_map.get(pol, "")
            time_lbl = getattr(ax, '_smk_time_lbl', None)
            
            # 1. Get current values (Primary: cache, Secondary: collection)
            vals = getattr(ax, '_smk_current_vals', None)
            if vals is None:
                for c in ax.collections:
                    if hasattr(c, 'get_array') and c.get_array() is not None:
                        vals = c.get_array()
                        break
            
            if vals is None or not hasattr(vals, 'size') or vals.size == 0:
                ax.set_title(f"{pol} (No Data)")
                ax.figure.canvas.draw_idle()
                return

            # 2. Check view extent
            try:
                xlim = ax.get_xlim(); ylim = ax.get_ylim()
                x0, x1 = min(xlim), max(xlim)
                y0, y1 = min(ylim), max(ylim)
                
                # Check for Base View (Global Stats optimization)
                is_base = False
                if hasattr(self, '_base_view') and self._base_view:
                    (bx_lim, by_lim) = self._base_view
                    bx0, bx1 = min(bx_lim), max(bx_lim)
                    by0, by1 = min(by_lim), max(by_lim)
                    if abs(x0-bx0) < 1e-4 and abs(x1-bx1) < 1e-4 and \
                       abs(y0-by0) < 1e-4 and abs(y1-by1) < 1e-4:
                        is_base = True
            except:
                is_base = False
                x0, x1, y0, y1 = 0, 0, 0, 0

            # 3. Extract Visible Stats
            try:
                if is_base:
                    filtered = vals
                else:
                    from shapely.geometry import box
                    if x0 == x1 or y0 == y1: # Degenerate view
                        filtered = np.array([])
                    else:
                        view_box = box(x0, y0, x1, y1)
                        filtered = vals
                        if hasattr(self, '_merged_gdf') and self._merged_gdf is not None:
                            gdf = self._merged_gdf
                            # A. Grid Optimization 
                            info = gdf.attrs.get('_smk_grid_info')
                            proj_sel = self.cmb_proj.currentText().lower()
                            is_native = proj_sel in ('auto', 'wgs84', 'native', 'epsg:4326', 'default') 
                            
                            if info and is_native and 'ROW' in gdf.columns:
                                col0 = int(np.floor((x0 - info['xorig']) / info['xcell']))
                                col1 = int(np.ceil((x1 - info['xorig']) / info['xcell']))
                                row0 = int(np.floor((y0 - info['yorig']) / info['ycell']))
                                row1 = int(np.ceil((y1 - info['yorig']) / info['ycell']))
                                
                                mask = (gdf['ROW'] > row0) & (gdf['ROW'] <= row1+1) & \
                                       (gdf['COL'] > col0) & (gdf['COL'] <= col1+1)
                                filtered = vals[mask.values] if mask.size == vals.size else vals
                            else:
                                # B. Precise Spatial Slicing
                                sidx = getattr(gdf, 'sindex', None)
                                if sidx:
                                    idxs = list(sidx.intersection(view_box.bounds))
                                    if idxs:
                                        # Validate indices are within bounds
                                        valid_idxs = [i for i in idxs if 0 <= i < len(gdf)]
                                        if valid_idxs:
                                            sub_gdf = gdf.iloc[valid_idxs]
                                            precise_mask = sub_gdf.geometry.intersects(view_box)
                                            final_ilocs = np.array(valid_idxs)[precise_mask.values]
                                            final_ilocs = final_ilocs[(final_ilocs >= 0) & (final_ilocs < len(vals))]
                                            filtered = vals[final_ilocs] if len(final_ilocs) > 0 else np.array([])
                                    else:
                                        filtered = np.array([])
                                else:
                                    # Fallback: Brute force intersection
                                    mask = gdf.geometry.intersects(view_box)
                                    filtered = vals[mask.values] if mask.size == vals.size else vals

                # Calculate Stats
                filtered_arr = np.asanyarray(filtered)
                clean = filtered_arr[~np.isnan(filtered_arr)] if filtered_arr.size > 0 else np.array([])
                
                if clean.size > 0:
                    mn, mx = np.nanmin(clean), np.nanmax(clean)
                    u, s = np.nanmean(clean), np.nansum(clean)
                    
                    def _f(v): 
                        if abs(v) < 1e-3 or abs(v) > 1e6: return f"{v:.4e}"
                        return f"{v:.4g}"
                        
                    stats_str = f"Min: {_f(mn)}  Mean: {_f(u)}  Max: {_f(mx)}  Sum: {_f(s)}"
                    if unit: stats_str += f" {unit}"
                    
                    # 4. Construct Multi-line Title
                    title_lines = [f"{pol} Emissions"]
                    if unit:
                        title_lines[0] += f" ({unit})"
                    
                    scc = self.cmb_scc.currentText()
                    if scc and scc != "ALL":
                        title_lines.append(f"SCC: {scc}")
                    
                    if self.cmb_ncf_layer.isVisible():
                        title_lines.append(f"Layer: {self.cmb_ncf_layer.currentText()}")
                    
                    if time_lbl:
                        title_lines.append(f"Time: {time_lbl}")
                    elif self.cmb_ncf_time.isVisible():
                         title_lines.append(f"Time: {self.cmb_ncf_time.currentText()}")
                    
                    # Append stats to title (Requested by user)
                    title_lines.append(stats_str)
                    
                    ax.set_title("\n".join(title_lines), fontsize=12, pad=10)

                    # Also update the persistent side panel if it exists
                    try:
                        self.lbl_stats_sum.setText(f"{s:.4g}")
                        self.lbl_stats_max.setText(f"{mx:.4g}")
                        self.lbl_stats_mean.setText(f"{u:.4g}")
                        self.lbl_stats_count.setText(str(len(clean)))
                    except Exception: pass
                    
                    # Remove floating stats text box if it exists
                    if hasattr(ax, '_smk_stats_text'):
                        try:
                            ax._smk_stats_text.remove()
                            del ax._smk_stats_text
                        except: pass
                    
                    self.statusBar().showMessage(f"View Stats | {stats_str}", 3000)
                else:
                    ax.set_title(f"{pol} (No Data in View)", fontsize=12)
                    if hasattr(ax, '_smk_stats_text'):
                        try:
                            ax._smk_stats_text.remove()
                            del ax._smk_stats_text
                        except: pass
                
                ax.figure.canvas.draw_idle()
            except Exception as e:
                logging.debug(f"Stats extraction failed: {e}")
                ax.set_title(f"{pol} Emissions", fontsize=12)
        except Exception as e:
            logging.debug(f"Global title update failed: {e}")

    # --- Updated Plot Logic ---
    def update_plot(self):
        """Main Plotting Logic, mirroring gui_qt.py exactly."""
        if self.emissions_df is None:
            if hasattr(self, 'input_files_list') and self.input_files_list:
                self.on_load_clicked()
                return
            self.notify_signal.emit('WARNING', 'Load smkreport and shapefile first.')
            return

        pol = self.cmb_pollutant.currentText()
        if not pol and hasattr(self, 'pollutants') and self.pollutants:
             pol = self.pollutants[0]
             self.cmb_pollutant.setCurrentText(pol)
        
        if not pol:
            self.notify_signal.emit('WARNING', 'No pollutant selected.')
            return

        # Capture UI state (Mirroring gui_qt.py variables)
        plot_by_mode = self.cmb_pltyp.currentText().lower()
        scc_selection = self.cmb_scc.currentText()
        scc_code_map = self._scc_display_to_code.copy() if self._scc_display_to_code else {}
        
        # Determine plotting CRS
        plot_crs_info = self._plot_crs()

        # NEW: Capture Plot Meta on Main Thread (Thread Safety)
        try:
            is_rev = self.chk_rev_cmap.isChecked()
            cmap_name = self.cmb_cmap.currentText()
            if is_rev: cmap_name += '_r'
            
            meta_fixed = {
                'cmap': cmap_name,
                'use_log': self.chk_log.isChecked(),
                'bins_txt': self.txt_bins.text(),
                'bins': self._parse_bins(),
                'unit': self.units_map.get(pol, ''),
                'vmin': float(self.txt_rmin.text()) if self.txt_rmin.text() else None,
                'vmax': float(self.txt_rmax.text()) if self.txt_rmax.text() else None,
                'show_graticule': self.chk_graticule.isChecked() if hasattr(self, 'chk_graticule') else True,
                'zoom_to_data': self.chk_zoom.isChecked(),
                'fill_nan': self.chk_nan0.isChecked()
            }
        except Exception as e:
            logging.error(f"Failed to capture meta: {e}")
            meta_fixed = {}

        self._start_progress(pol=pol)
        
        threading.Thread(
            target=self._plot_worker, 
            args=(pol, plot_by_mode, scc_selection, scc_code_map, plot_crs_info, meta_fixed),
            daemon=True
        ).start()

    def _merged(self, plot_by_mode=None, scc_selection=None, scc_code_map=None, notify=None, pollutant=None, fill_nan=False) -> Optional[gpd.GeoDataFrame]:
        """Prepare and merge emissions with geometry, mirroring gui_qt.py logic exactly."""
        if self.emissions_df is None:
            return None

        # Determine if we have a native NetCDF/already-gridded dataset
        is_native = getattr(self.emissions_df, 'attrs', {}).get('_smk_is_native', False)

        def _do_notify(level, title, msg, exc=None, **kwargs):
            if notify:
                notify(level, title, msg, exc, **kwargs)
            else:
                self.notify_signal.emit(level, f"{title}: {msg}")

        mode = (plot_by_mode if plot_by_mode is not None else self.cmb_pltyp.currentText().lower())
        
        # Shortcut for Native Grids if in 'auto' or 'grid' mode AND large enough for QuadMesh optimization
        if is_native and mode in ['auto', 'grid'] and len(self.emissions_df) > 10000:
            target_pol = pollutant if pollutant else self.cmb_pollutant.currentText()
            # For native datasets (NetCDF/Inline), we can proceed if the pollutant is available
            if target_pol and target_pol in getattr(self.emissions_df, 'columns', []):
                logging.debug(f"Native grid shortcut for {target_pol}")
                
                # Slicing is disabled to avoid duplicate column header issues
                res_df = self.emissions_df.copy()
                
                # Apply fill_nan in shortcut
                if fill_nan:
                    res_df[target_pol] = res_df[target_pol].fillna(0)

                # Ensure GRID_RC exists (Required for Hover tools and coordination)
                if 'GRID_RC' not in res_df.columns and 'ROW' in res_df.columns and 'COL' in res_df.columns:
                    try: 
                        # Vectorized string conversion is faster
                        res_df['GRID_RC'] = res_df['ROW'].astype(str) + '_' + res_df['COL'].astype(str)
                        # Cache it back to source to avoid re-calculation on next pollutant switch
                        if 'GRID_RC' not in self.emissions_df.columns:
                            self.emissions_df['GRID_RC'] = res_df['GRID_RC']
                    except: pass
                
                # Downstream pipeline (Reprojection/Plotter) expects a GeoDataFrame
                if not isinstance(res_df, gpd.GeoDataFrame):
                    res_gdf = gpd.GeoDataFrame(res_df, geometry=None)
                    # Preserve attributes which contain the grid info
                    res_gdf.attrs = self.emissions_df.attrs.copy()
                    
                    # NEW: Robustly inject grid info from active grid if missing in dataset
                    if '_smk_grid_info' not in res_gdf.attrs and self.grid_gdf is not None:
                         if hasattr(self.grid_gdf, 'attrs') and '_smk_grid_info' in self.grid_gdf.attrs:
                              res_gdf.attrs['_smk_grid_info'] = self.grid_gdf.attrs['_smk_grid_info']
                    
                    # Try to assign CRS from grid info if missing
                    if getattr(res_gdf, 'crs', None) is None:
                        try:
                            info = res_gdf.attrs.get('_smk_grid_info')
                            if info and info.get('proj_str'):
                                res_gdf.set_crs(info['proj_str'], inplace=True)
                        except: pass
                    return res_gdf
                return res_df

        base_gdf: Optional[gpd.GeoDataFrame] = None
        merge_on: Optional[str] = None
        geometry_tag = None
        
        try:
            source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type') if isinstance(self.emissions_df, pd.DataFrame) else None
        except Exception:
            source_type = None

        sel_display = scc_selection if scc_selection is not None else self.cmb_scc.currentText()
        sel_code = ''
        code_map = scc_code_map if scc_code_map is not None else self._scc_display_to_code
        if code_map:
            try:
                sel_code = code_map.get(sel_display, '') or ''
            except Exception:
                sel_code = ''
        
        has_scc_cols = any(c.lower() in ['scc', 'scc code', 'scc_code'] for c in self.emissions_df.columns)
        use_scc_filter = bool(has_scc_cols and sel_code)

        if mode == 'grid':
            if self.grid_gdf is None:
                _do_notify('WARNING', 'Grid not loaded', 'Select a GRIDDESC and Grid Name first to build the grid geometry.')
                raise ValueError("Handled")
            
            # For NetCDF, we don't need to ensure FF10 mapping, but we might need to ensure GRID_RC exists
            if not is_native:
                self._ensure_ff10_grid_mapping(notify_success=False)
            else:
                if 'GRID_RC' not in self.emissions_df.columns and 'ROW' in self.emissions_df.columns:
                    try:
                        self.emissions_df['GRID_RC'] = self.emissions_df['ROW'].astype(str) + '_' + self.emissions_df['COL'].astype(str)
                    except: pass

            base_gdf = self.grid_gdf
            merge_on = 'GRID_RC'
            geometry_tag = 'grid'
        elif mode == 'county':
            if self.counties_gdf is None:
                _do_notify('WARNING', 'Counties not loaded', 'Load a counties shapefile or use the online counties option.')
                raise ValueError("Handled")
            base_gdf = self.counties_gdf
            merge_on = 'FIPS'
            geometry_tag = 'county'
        else: # auto
            if self.grid_gdf is not None:
                if 'GRID_RC' not in getattr(self.emissions_df, 'columns', []):
                    if not is_native:
                        self._ensure_ff10_grid_mapping(notify_success=False)
                    else:
                        if 'ROW' in self.emissions_df.columns:
                             try: self.emissions_df['GRID_RC'] = self.emissions_df['ROW'].astype(str) + '_' + self.emissions_df['COL'].astype(str)
                             except: pass
                             
                if 'GRID_RC' in getattr(self.emissions_df, 'columns', []):
                    base_gdf = self.grid_gdf
                    merge_on = 'GRID_RC'
                    geometry_tag = 'grid'
            if base_gdf is None and self.counties_gdf is not None:
                base_gdf = self.counties_gdf
                merge_on = 'FIPS'
                if not any(c.lower() == 'fips' for c in self.emissions_df.columns):
                    if any(c.lower() == 'region_cd' for c in self.emissions_df.columns):
                        merge_on = 'FIPS' 
                geometry_tag = 'county'
            if base_gdf is None:
                _do_notify('WARNING', 'No suitable geometry', 'Could not find a suitable shapefile (Counties or Grid) for the loaded emissions data.')
                raise ValueError("Handled")

        if merge_on and isinstance(base_gdf, gpd.GeoDataFrame):
            if merge_on not in base_gdf.columns:
                if merge_on == 'FIPS' and 'region_cd' in base_gdf.columns:
                    try:
                        base_gdf = base_gdf.copy()
                        base_gdf['FIPS'] = base_gdf['region_cd']
                    except Exception:
                        pass
                else:
                    _do_notify('WARNING', 'Geometry Missing Column', f"Selected geometry layer lacks '{merge_on}' column.")
                    raise ValueError("Handled")

        pol_tuple = tuple(self.pollutants or [])
        target_pol = pollutant if pollutant else self.cmb_pollutant.currentText()

        # Cache key including NaN fill and spatial filter state
        cache_key = (
            geometry_tag or mode,
            merge_on or '',
            id(base_gdf) if base_gdf is not None else 0,
            id(self.emissions_df) if isinstance(self.emissions_df, pd.DataFrame) else 0,
            id(self.raw_df) if isinstance(self.raw_df, pd.DataFrame) else 0,
            sel_code if use_scc_filter else '',
            pol_tuple,
            target_pol or '',
            fill_nan,
            self.cmb_filter_op.currentText() if hasattr(self, 'cmb_filter_op') else 'False',
            id(self.filter_gdf) if getattr(self, 'filter_gdf', None) is not None else 0
        )
        
        cached = self._merged_cache.get(cache_key)
        if cached is not None:
            return cached[0].copy()

        # Use full copy to avoid issues with duplicate columns in source data
        if isinstance(self.emissions_df, pd.DataFrame):
            emis_for_merge = self.emissions_df.copy()
            # Copy attrs manually
            if hasattr(self.emissions_df, 'attrs'):
                emis_for_merge.attrs = self.emissions_df.attrs.copy()
        else:
            emis_for_merge = None

        if emis_for_merge is None:
            return None

        # ON-DEMAND LAZY NETCDF FETCH
        if target_pol and target_pol not in emis_for_merge.columns:
            ds = emis_for_merge.attrs.get('_smk_xr_ds')
            if ds is not None:
                _do_notify('INFO', 'Fetching Data', f"Lazy-extracting {target_pol} from NetCDF dataset...")
                try:
                    from ncf_processing import read_ncf_emissions
                    ncf_params = emis_for_merge.attrs.get('ncf_params', {})
                    new_data = read_ncf_emissions(self.input_files_list[0], pollutants=[target_pol], xr_ds=ds, **ncf_params)
                    if target_pol in new_data.columns:
                        emis_for_merge[target_pol] = new_data[target_pol].values
                        self.emissions_df[target_pol] = new_data[target_pol].values # Sync back
                        if target_pol not in self.units_map:
                            v_meta = new_data.attrs.get('variable_metadata', {}).get(target_pol, {})
                            self.units_map[target_pol] = v_meta.get('units', '')
                except Exception as e:
                    _do_notify('WARNING', 'Fetch Failed', f"Could not lazy-load {target_pol}: {e}")

        # SCC Filtering & Re-aggregation
        if merge_on not in emis_for_merge.columns or use_scc_filter:
            raw_to_use = self.raw_df
            if use_scc_filter and self.raw_df is not None:
                scc_col = next((c for c in raw_to_use.columns if c.lower() in ['scc', 'scc code', 'scc_code']), None)
                if scc_col:
                    raw_to_use = raw_to_use[raw_to_use[scc_col].astype(str).str.strip() == sel_code].copy()
            
            if isinstance(raw_to_use, pd.DataFrame) and merge_on in raw_to_use.columns:
                pols = list(self.pollutants or detect_pollutants(raw_to_use))
                if pols:
                    try:
                        subset = raw_to_use[[merge_on] + pols]
                        agg = subset.groupby(merge_on, sort=False).sum(numeric_only=True).reset_index()
                        agg.attrs = dict(getattr(emis_for_merge, 'attrs', {}))
                        emis_for_merge = agg
                    except Exception as e:
                        logging.warning(f"Re-aggregation failed: {e}")

        if merge_on not in emis_for_merge.columns:
            _do_notify('WARNING', 'Missing Join Column', f"Data lacks '{merge_on}' required for {geometry_tag} plot.")
            raise ValueError("Handled")

        try:
            from data_processing import merge_emissions_with_geometry
            merged, prepared = merge_emissions_with_geometry(
                emis_for_merge, base_gdf, merge_on, sort=False, copy_geometry=False
            )
            # Propagate attributes from emissions and geometry (CRITICAL for QuadMesh/Hover)
            if hasattr(merged, 'attrs'):
                for src in [emis_for_merge, base_gdf]:
                    if hasattr(src, 'attrs'):
                        merged.attrs.update(src.attrs)

            matched = prepared[merge_on].dropna().unique()
            merged['__has_emissions'] = merged[merge_on].isin(matched)
            
            # fill_nan logic
            if fill_nan:
                pols = list(self.pollutants or [])
                if target_pol and target_pol not in pols:
                    pols.append(target_pol)
                    
                if pols:
                    cols_to_fill = [c for c in pols if c in merged.columns]
                    if cols_to_fill:
                        # Only fill regions where we actually have a geometry but no match
                        # merged['__has_emissions'] tells us if it was a match
                        merged[cols_to_fill] = merged[cols_to_fill].fillna(0.0)

            # Spatial Filtering
            filter_mode = self.cmb_filter_op.currentText()
            if filter_mode != 'False' and self.filter_gdf is not None:
                _do_notify('INFO', 'Filtering', f'Filtering data by shapefile ({filter_mode})...')
                from data_processing import apply_spatial_filter
                try:
                    merged = apply_spatial_filter(merged, self.filter_gdf, filter_mode)
                except Exception as e:
                    _do_notify('WARNING', 'Filter Failed', f"Could not filter by shapefile ({filter_mode}): {e}")

            self._merged_cache[cache_key] = (merged.copy(), prepared, merge_on)
            
            # Ensure the merged GDF has a CRS (fallback to WGS84 for remapping)
            if getattr(merged, 'crs', None) is None:
                 if getattr(base_gdf, 'crs', None) is not None:
                      merged.crs = base_gdf.crs
                 else:
                      merged.crs = "EPSG:4326"
            
            return merged
        except Exception as e:
            _do_notify('ERROR', 'Merge Failed', f"{e}")
            return None

    def _plot_worker(self, pollutant, plot_by_mode, scc_selection, scc_code_map, plot_crs_info, meta_fixed):
        """Worker thread to prepare data and reproject, mirroring gui_qt.py exactly."""
        try:
            logging.info(f"Plotting worker started for: {pollutant}")
            
            def safe_notify(level, title, msg, exc=None, **kwargs):
                try:
                    self.notify_signal.emit(level, f"{title}: {msg}")
                except RuntimeError:
                    pass

            try:
                merged = self._merged(
                    plot_by_mode=plot_by_mode, 
                    scc_selection=scc_selection, 
                    scc_code_map=scc_code_map,
                    notify=safe_notify,
                    pollutant=pollutant,
                    fill_nan=meta_fixed.get('fill_nan', False)
                )
            except ValueError as ve:
                if str(ve) == "Handled":
                     QTimer.singleShot(0, self._stop_progress)
                     return
                raise ve

            if merged is None:
                try:
                    self.notify_signal.emit('WARNING', 'Missing Data. Load smkreport and shapefile first.')
                except RuntimeError:
                    pass
                QTimer.singleShot(0, self._stop_progress)
                return

            # REPROJECTION LOGIC
            plot_crs, tf_fwd, tf_inv = plot_crs_info
            
            try:
                # Ensure merged has a CRS for to_crs() to work (defaulting to 4326 if missing)
                if getattr(merged, 'crs', None) is None:
                     try: merged.set_crs("EPSG:4326", inplace=True)
                     except: pass

                is_native_plot = getattr(merged, 'attrs', {}).get('_smk_is_native', False)
                # Ensure we only skip reprojection if the optimization will actually be used (Size > 10000)
                should_shortcut = is_native_plot and len(merged) > 10000
                
                if plot_crs is not None:
                     # Performance: Skip expensive to_crs if using optimized QuadMesh for native/NetCDF grids
                     if should_shortcut:
                          logging.info("Native grid shortcut: Skipping per-cell vector reprojection.")
                          merged_plot = merged
                     else:
                          logging.info(f"Reprojecting plot data to: {plot_crs}")
                          merged_plot = merged.to_crs(plot_crs)
                else:
                     merged_plot = merged
                     
                # Ensure attributes (grid info) are preserved after reprojection
                if getattr(merged, 'attrs', None) and getattr(merged_plot, 'attrs', None) is not None:
                    for k in ['stack_groups_path', 'proxy_ncf_path', 'original_ncf_path', '_smk_grid_info', '_smk_is_native', 'ncf_params', '_smk_xr_ds', 'per_file_summaries']:
                        if k in merged.attrs:
                                merged_plot.attrs[k] = merged.attrs[k]
            except Exception as e:
                logging.warning(f"Reprojection failed: {e}")
                merged_plot = merged

            # metadata for plotting 
            logging.info(f"DEBUG: Final plot CRS: {getattr(merged_plot, 'crs', 'None')}")
            meta = meta_fixed.copy()
            meta['tf_fwd'] = tf_fwd
            meta['tf_inv'] = tf_inv
            
            # Calculate Zoom Bounds if requested
            if meta.get('zoom_to_data') and pollutant in merged_plot.columns:
                try:
                    # Filter for non-zero emissions to zoom into actual data
                    non_zero = merged_plot[merged_plot[pollutant] > 0]
                    if not non_zero.empty:
                        meta['zoom_gdf'] = non_zero
                    else:
                        meta['zoom_gdf'] = merged_plot
                except Exception:
                    meta['zoom_gdf'] = merged_plot
            else:
                meta['zoom_gdf'] = merged_plot

            try:
                self.plot_ready_signal.emit(merged_plot, pollutant, meta)
            except RuntimeError:
                pass
            
        except Exception as e:
            try:
                self.notify_signal.emit("ERROR", f"Plot Preparation Error: {e}")
            except RuntimeError:
                pass
            logging.exception("Plot worker failed")
        finally:
            QTimer.singleShot(0, self._stop_progress)

    @Slot(object, str, dict)
    def _render_plot_on_main(self, gdf, column, meta):
        """Main thread slot to update the matplotlib figure."""
        try:
            # 1. Clear and setup figure
            self.figure.clear()
            
            # --- Dynamic Controls Update (Mirroring gui_qt.py) ---
            # Clear existing items
            while self.pc_layout.count():
                item = self.pc_layout.takeAt(0)
                widget = item.widget()
                if widget: widget.deleteLater()
            
            # Detect NCF Source
            is_ncf = False
            try:
                attrs = getattr(gdf, 'attrs', {})
                src_type = attrs.get('source_type')
                # FORCE boolean evaluation to avoid NoneType crash in setVisible
                flag = (src_type in ('gridded_netcdf', 'inline_point_lazy')) or \
                       (attrs.get('format') == 'netcdf') or \
                       ('proxy_ncf_path' in attrs) or \
                       (attrs.get('is_netcdf') is True) or \
                       bool(attrs.get('source_path') and is_netcdf_file(attrs.get('source_path')))
                is_ncf = bool(flag)
            except: pass
            
            self.plot_controls_frame.setVisible(is_ncf)
            
            if is_ncf:
                 # Populate Layout
                 self.pc_layout.addWidget(QLabel("Time:"))
                 self.lbl_anim_status = QLabel("-")
                 self.pc_layout.addWidget(self.lbl_anim_status)
                 
                 btn_prev = QPushButton("< Prev")
                 btn_prev.clicked.connect(lambda: self._step_time(-1))
                 self.pc_layout.addWidget(btn_prev)
                 
                 btn_next = QPushButton("Next >")
                 btn_next.clicked.connect(lambda: self._step_time(1))
                 self.pc_layout.addWidget(btn_next)
                 
                 # Aggregation
                 self.pc_layout.addSpacing(10)
                 self.pc_layout.addWidget(QLabel("| View:"))
                 for lbl, mode in [('Total', 'total'), ('Avg', 'avg'), ('Max', 'max'), ('Min', 'min')]:
                     b = QPushButton(lbl)
                     b.clicked.connect(partial(self._show_agg, mode))
                     self.pc_layout.addWidget(b)
                     
                 # TS Extraction
                 self.pc_layout.addSpacing(10)
                 self.pc_layout.addWidget(QLabel("| TS in View:"))
                 for lbl, mode in [('Mean', 'mean'), ('Sum', 'sum')]:
                     b = QPushButton(lbl)
                     b.clicked.connect(partial(self._on_ts_view_click, mode))
                     self.pc_layout.addWidget(b)
                     
                 self.pc_layout.addStretch()
            # -----------------------------------------------------

            ax = self.figure.add_subplot(111)
            pre_axes = set(self.figure.axes)
            
            # 2. Extract meta/settings
            cmap = meta.get('cmap', 'viridis')
            use_log = meta.get('use_log', True)
            bins_txt = meta.get('bins_txt', "")
            bins = meta.get('bins', [])
            zoom_to_data = meta.get('zoom_to_data', False)
            zoom_gdf = meta.get('zoom_gdf', gdf)
            show_graticule = meta.get('show_graticule', True)
            unit = self.units_map.get(column, "")
            
            # 3. Get transformers (needed for hover & graticule)
            # Preference: Use transformers from meta (calculated during projection setup)
            # Fallback: Recalculate if they are missing or CRS changed
            tf_fwd = meta.get('tf_fwd')
            tf_inv = meta.get('tf_inv')
            if tf_fwd is None or tf_inv is None:
                tf_fwd, tf_inv = self._get_transformers(gdf.crs)

            # 4. Initialize Axis & Aspect Ratio
            # Use 'box' adjustment to maintain stability during zoom/resize
            ax.set_aspect('equal', adjustable='box')
            
            # 5. Render main plot
            self._merged_gdf = gdf
            # Store current values for dynamic title stats
            ax._smk_current_vals = gdf[column].values
            ax._smk_time_lbl = None
            
            # Style & Optimization (Mirroring gui_qt.py)
            p_lw = 0.05
            p_ec = 'black'
            if len(gdf) > 20000:
                p_lw = 0.0
                p_ec = 'none'

            unit = self.units_map.get(column, "")
            # Fallback for units
            if not unit:
                vmeta = getattr(self.emissions_df, 'attrs', {}).get('variable_metadata', {})
                if isinstance(vmeta, dict) and column in vmeta:
                     unit = vmeta[column].get('units', '')

            collection = create_map_plot(
                gdf=gdf,
                column=column,
                title="", 
                ax=ax,
                cmap_name=cmap,
                bins=bins,
                log_scale=use_log,
                overlay_counties=self.counties_gdf,
                overlay_shape=self.overlay_gdf,
                unit_label=unit, # Pass units directly for better detection
                crs_proj=getattr(gdf, 'crs', None),
                tf_fwd=tf_fwd if show_graticule else None,
                tf_inv=tf_inv if show_graticule else None,
                zoom_to_data=False, # Limits already set above
                linewidth=p_lw,
                edgecolor=p_ec
            )
            
            # --- Zoom Calculation (Robust for Gridded Shortcut) ---
            if zoom_to_data or self._base_view is None:
                try:
                    # Check if we can use standard total_bounds
                    if hasattr(zoom_gdf, 'total_bounds') and not np.isnan(zoom_gdf.total_bounds).any():
                        minx, miny, maxx, maxy = zoom_gdf.total_bounds
                    elif hasattr(zoom_gdf, 'attrs') and '_smk_grid_info' in zoom_gdf.attrs:
                        # Shortcut path: geometry is None, use grid info
                        info = zoom_gdf.attrs['_smk_grid_info']
                        minx, miny = info['xorig'], info['yorig']
                        maxx = minx + info['ncols'] * info['xcell']
                        maxy = miny + info['nrows'] * info['ycell']
                        
                        # Handle potential coordinate transformation if axes are in different CRS (e.g. 4326)
                        native_crs = info.get('proj_str')
                        target_crs = getattr(ax, '_smk_ax_crs', None) or getattr(gdf, 'crs', None)
                        if native_crs and target_crs and str(native_crs) != str(target_crs):
                            from pyproj import Transformer
                            trans = Transformer.from_crs(native_crs, target_crs, always_xy=True)
                            pts_x, pts_y = trans.transform([minx, maxx, minx, maxx], [miny, miny, maxy, maxy])
                            minx, maxx = min(pts_x), max(pts_x)
                            miny, maxy = min(pts_y), max(pts_y)
                    else:
                        minx, miny, maxx, maxy = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]

                    padx = (maxx - minx) * 0.05
                    pady = (maxy - miny) * 0.05
                    if abs(padx) < 1e-7: padx = 0.5
                    if abs(pady) < 1e-7: pady = 0.5
                    ax.set_xlim(min(minx, maxx) - abs(padx), max(minx, maxx) + abs(padx))
                    ax.set_ylim(min(miny, maxy) - abs(pady), max(miny, maxy) + abs(pady))
                except Exception as ze:
                    logging.warning(f"Zoom calculation failed: {ze}")
            
            # Initial Draw to register axes
            self.canvas.draw_idle()

            # 6. Install Interactions (Hover & Box-Zoom)
            self._setup_hover(gdf, column, ax, tf_inv=tf_inv)
            self._install_box_zoom(ax)
            
            # Step-wise draw to register axes and allow tick generation
            self.canvas.draw()
            
            # 7. Format Colorbar (Mirroring gui_qt.py)
            # Identify the colorbar axis. Prefer the direct colorbar object if attached to collection.
            cbar_ax = None
            if collection is not None and hasattr(collection, 'colorbar') and collection.colorbar is not None:
                cbar_ax = collection.colorbar.ax
            
            if cbar_ax is None:
                # Fallback to axis search
                all_axes = self.figure.axes
                new_candidate = [a for a in all_axes if a not in pre_axes and a is not ax]
                if new_candidate:
                    cbar_ax = new_candidate[0]
                
                if cbar_ax is None:
                    for cand in all_axes:
                        if cand is ax: continue
                        if str(cand.get_label()) == '<colorbar>':
                            cbar_ax = cand
                            break
                
                if cbar_ax is None:
                    for cand in all_axes:
                        if cand is ax: continue
                        try:
                            bbox = cand.get_position()
                            if min(bbox.width, bbox.height) < 0.3 * max(bbox.width, bbox.height):
                                cbar_ax = cand
                                break
                        except: pass

            norm = getattr(collection, 'norm', None) if collection is not None else None

            if cbar_ax is not None and collection is not None:
                # 7a. Robust orientation detection
                try:
                    bbox = cbar_ax.get_position()
                    orient_vertical = (bbox.height >= bbox.width)
                except Exception:
                    orient_vertical = True

                # 7b. Label colorbar with units if available (Use local unit variable)
                try:
                    if unit:
                         if orient_vertical: cbar_ax.set_ylabel(unit, fontweight='bold', fontsize=9)
                         else: cbar_ax.set_xlabel(unit, fontweight='bold', fontsize=9)
                except Exception: pass
                    
                # 7c. Robust Colorbar Formatting (Fix for small values and custom bins)
                try:
                    from matplotlib.ticker import FixedLocator, FuncFormatter, ScalarFormatter, LogFormatter, FixedFormatter, AutoLocator
                    from matplotlib.colors import LogNorm
                    is_log = isinstance(norm, LogNorm) if norm is not None else False
                    bins_ticks = (bins if bins is not None else [])

                    if bins_ticks:
                        ticks = [t for t in bins_ticks if (t > 0) ] if is_log else list(bins_ticks)
                        if ticks:
                            # Trigger scientific if values are extreme
                            txt_bins = self.txt_bins.text().lower() if hasattr(self, 'txt_bins') else ""
                            use_sci = 'e' in txt_bins or any(abs(t) >= 1e6 or (0 < abs(t) < 1e-4) for t in ticks)

                            if not use_sci:
                                # Positional format for human readability if not too small
                                try:
                                    labels = [np.format_float_positional(t, trim='-') for t in ticks]
                                    fmt = FixedFormatter(labels)
                                except Exception:
                                    fmt = FuncFormatter(lambda x, p: f"{x:g}")
                            else:
                                # Scientific format for extreme values
                                fmt = FuncFormatter(lambda x, p: f"{x:.4g}")
                            
                            if orient_vertical:
                                cbar_ax.yaxis.set_major_locator(FixedLocator(ticks))
                                cbar_ax.yaxis.set_major_formatter(fmt)
                                cbar_ax.xaxis.set_ticks([])
                            else:
                                cbar_ax.xaxis.set_major_locator(FixedLocator(ticks))
                                cbar_ax.xaxis.set_major_formatter(fmt)
                                cbar_ax.yaxis.set_ticks([])
                    else:
                        # Default Formatter for standard linear/log scales
                        if is_log:
                            fmt = LogFormatter()
                            if orient_vertical: 
                                cbar_ax.yaxis.set_major_formatter(fmt)
                            else: 
                                cbar_ax.xaxis.set_major_formatter(fmt)
                        else:
                            fmt = ScalarFormatter(useOffset=False)
                            fmt.set_scientific(True)
                            fmt.set_powerlimits((-4, 4))
                            
                            if orient_vertical: 
                                # Force a few ticks if it's empty
                                cbar_ax.yaxis.set_major_locator(AutoLocator())
                                cbar_ax.yaxis.set_major_formatter(fmt)
                            else: 
                                cbar_ax.xaxis.set_major_locator(AutoLocator())
                                cbar_ax.xaxis.set_major_formatter(fmt)
                            
                    # 7d. Style Optimization (Ticks and Fonts)
                    try:
                        # Refresh after setting formatters
                        self.canvas.draw()
                        labels_to_style = cbar_ax.get_yticklabels() if orient_vertical else cbar_ax.get_xticklabels()
                        if labels_to_style:
                            for lbl in labels_to_style:
                                lbl.set_fontsize(8)
                    except Exception: pass
                except Exception as e:
                    logging.debug(f"Colorbar formatting failed: {e}")
            
            # 8. Initial Title Stats
            self._update_plot_title(ax, immediate=True)
            self.canvas.draw_idle()

            # 9. Sync Interaction Logic with gui_qt.py
            try:
                # A. Base View for Home and Clamping
                base_xlim = ax.get_xlim()
                base_ylim = ax.get_ylim()
                self._base_view = (base_xlim, base_ylim)
                
                # B. Home Button Override (Direct Logic)
                def _home_override(*args, **kwargs):
                    try:
                        ax.set_xlim(base_xlim)
                        ax.set_ylim(base_ylim)
                        self.canvas.draw_idle()
                        self._update_plot_title(ax, immediate=True)
                    except: pass
                
                # Override the instance method
                self.toolbar.home = _home_override
                
                # C. Reset Navigation Stack
                if hasattr(self.toolbar, '_nav_stack'):
                    self.toolbar._nav_stack.clear()
                
                # IMPORTANT: In Qt, we must refresh the button states
                if hasattr(self.toolbar, 'set_history_buttons'):
                    self.toolbar.set_history_buttons()
                elif hasattr(self.toolbar, 'update'):
                    self.toolbar.update()

                # D. Axis Callbacks for Live Stats (Robust Update)
                if hasattr(ax, '_smk_stats_cids'):
                    for cid in ax._smk_stats_cids:
                        ax.callbacks.disconnect(cid)
                
                c1 = ax.callbacks.connect('xlim_changed', lambda a: self._update_plot_title(a))
                c2 = ax.callbacks.connect('ylim_changed', lambda a: self._update_plot_title(a))
                ax._smk_stats_cids = [c1, c2]

            except Exception as he:
                logging.debug(f"Interaction sync failed: {he}")
 
            self.notify_signal.emit("INFO", f"Plotted {column}")
            self._update_stats_panel(gdf, column)
            self.canvas.draw_idle()
            
        except Exception as e:
            self.notify_signal.emit("ERROR", f"Render failed: {e}")
            traceback.print_exc()
        finally:
            self._stop_progress()

    def reset_home_view(self):
        """Reset the plot view to the full extent of the loaded data (Full Domain)."""
        try:
            if self._merged_gdf is not None and not self._merged_gdf.empty:
                ax = self.figure.gca()
                bounds = self._merged_gdf.total_bounds
                x_pad = (bounds[2] - bounds[0]) * 0.05
                y_pad = (bounds[3] - bounds[1]) * 0.05
                if x_pad == 0: x_pad = 0.5
                if y_pad == 0: y_pad = 0.5
                ax.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
                ax.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)
                self.canvas.draw_idle()
                self._update_plot_title(ax, immediate=True)
            elif self.toolbar:
                self.toolbar.home()
        except Exception as e:
            logging.warning(f"Reset view failed: {e}")

    def _on_canvas_motion(self, event):
        """Update status bar on hover."""
        if not event.inaxes: return
        try:
             # Use the format_coord string which we already enhanced in _setup_hover
             res = event.inaxes.format_coord(event.xdata, event.ydata)
             self.status_label.setText(res)
        except: pass

    def _handle_cell_click(self, x, y):
        """Handle click for Time Series extraction."""
        if self._merged_gdf is None or not 'ROW' in self._merged_gdf.columns:
            return

        try:
             # Identify cell
             info = self._merged_gdf.attrs.get('_smk_grid_info')
             if not info: return # Can't identify without grid info
             
             col_look = int(np.floor((x - info['xorig']) / info['xcell'])) + 1
             row_look = int(np.floor((y - info['yorig']) / info['ycell'])) + 1
             
             logging.info(f"Click at ({x}, {y}) -> Cell ({row_look}, {col_look})")
             
             # Check if ncf
             is_ncf = self.plot_controls_frame.isVisible()
             if not is_ncf: return
             
             from ncf_processing import get_ncf_timeseries
             self.notify_signal.emit("INFO", f"Extracting Time Series for Cell ({row_look}, {col_look})...")
             
             # Layer/Op
             l_idx = 0; l_op = 'select'
             lay_txt = self.cmb_ncf_layer.currentText()
             if "Sum" in lay_txt: l_op = 'sum'
             elif "Avg" in lay_txt: l_op = 'mean'
             elif "Layer" in lay_txt:
                 try: l_idx = int(lay_txt.split()[-1]) - 1
                 except: pass
             
             QTimer.singleShot(0, lambda: self._exec_ts_plot(row_look, col_look, l_idx, l_op))
             
        except Exception as e:
             logging.error(f"Click handler error: {e}")

    def _exec_ts_plot(self, r, c, l_idx, l_op):
         try:
             # Run extraction (blocking is okay for short TS, or move to thread if slow)
             from ncf_processing import get_ncf_timeseries
             res = get_ncf_timeseries(
                self.input_files_list[0],
                self.cmb_pollutant.currentText(),
                [r-1], [c-1],
                layer_idx=l_idx, layer_op=l_op, op='mean'
             )
             
             if res:
                 self._show_ts_window(res, f"Cell ({r}, {c})")
                 self.notify_signal.emit("INFO", f"Plotted Cell ({r}, {c})")
             else:
                 self.notify_signal.emit("WARNING", "No data found for cell.")
         except Exception as e:
             self.notify_signal.emit("ERROR", f"TS Extraction failed: {e}")

    def _show_ts_window(self, data, title):
        try:
             win = TimeSeriesPlotWindow(data, title, self.cmb_pollutant.currentText(), self.units_map.get(self.cmb_pollutant.currentText(), ""), self)
             win.show()
        except Exception as e:
            logging.error(f"TS Window error: {e}")


    def _parse_bins(self) -> List[float]:
        """Parse custom bins from the GUI field (comma or space separated)."""
        raw = self.txt_bins.text().strip()
        if not raw:
            return []
        try:
            # Handle both commas and spaces
            parts = [p for p in raw.replace(',', ' ').split() if p]
            # Use sorted set to ensure strictly monotonic increasing values
            vals = sorted(list(set(float(p) for p in parts)))
            return vals
        except Exception:
            self.notify_signal.emit('WARNING', f"Bins Parse Error: Could not parse '{raw}'; expected numbers.")
            return []

    def _setup_hover(self, merged: gpd.GeoDataFrame, pollutant: str, ax=None, tf_inv: Optional[pyproj.Transformer] = None) -> None:
        """Enhance status bar to show pollutant at cursor and WGS84 lon/lat (Optimized for speed and accuracy)."""
        from shapely.geometry import Point
        
        gdf = merged
        sindex = getattr(gdf, 'sindex', None)
        target_ax = ax if ax is not None else getattr(self, 'ax', None)
        if target_ax is None: return

        # 1. Pre-calculate lookup data (avoid repeated Series/DataFrame lookups on every mouse move)
        # Using numpy arrays directly is significantly faster than Pandas row lookups (iloc)
        gdf_cols = gdf.columns.tolist()
        
        # Robust column finding (handle suffixes like _x/_y from merges)
        r_col = next((c for c in gdf_cols if c.upper().startswith('ROW')), None)
        c_col = next((c for c in gdf_cols if c.upper().startswith('COL')), None)
        
        r_arr = gdf[r_col].values if r_col else None
        c_arr = gdf[c_col].values if c_col else None
        f_arr = gdf['FIPS'].values if 'FIPS' in gdf_cols else None
        re_arr = gdf['region_cd'].values if 'region_cd' in gdf_cols else (gdf['REGION_CD'].values if 'REGION_CD' in gdf_cols else None)
        g_arr = gdf['GRID_RC'].values if 'GRID_RC' in gdf_cols else None
        p_arr = gdf[pollutant].values if pollutant in gdf_cols else None
        geom_arr = gdf.geometry.values if hasattr(gdf, 'geometry') else []

        # 2. Pre-build Grid lookup transformer to avoid CRS/Projection overhead
        info = getattr(gdf, 'attrs', {}).get('_smk_grid_info')
        lookup_tf = None
        if info:
            try:
                native_crs_str = info.get('proj_str')
                if native_crs_str:
                    native_crs = pyproj.CRS.from_user_input(native_crs_str)
                    # Use the actual CRS from the axes if available
                    axes_crs = getattr(target_ax, '_smk_ax_crs', None) or getattr(gdf, 'crs', None)
                    if axes_crs is not None:
                         # Always create transformer if they might be different; pyproj optimizes identity.
                         lookup_tf = pyproj.Transformer.from_crs(axes_crs, native_crs, always_xy=True)
            except Exception: pass

        def _fmt(x: float, y: float) -> str:
            # Lon/Lat mapping for status bar display
            base = None
            try:
                if tf_inv is not None:
                    lon_val, lat_val = tf_inv.transform(x, y)
                    if abs(lon_val) < 1000 and abs(lat_val) < 1000: # Sanity check for degrees
                        base = f"lon={lon_val:.4f}\u00b0, lat={lat_val:.4f}\u00b0"
                elif getattr(gdf, 'crs', None) is not None:
                    try:
                        is_geog = gdf.crs.is_geographic if hasattr(gdf.crs, 'is_geographic') else False
                        if is_geog:
                            base = f"lon={x:.4f}\u00b0, lat={y:.4f}\u00b0"
                    except Exception: pass
            except Exception: pass
                
            if base is None:
                base = f"x={x:.4f}, y={y:.4f}"

            try:
                # Optimized multi-stage lookup
                cand_idx = []
                
                # Fast Strategy 1: Native Grid Math (O(1) search candidate range)
                if info is not None and r_arr is not None:
                    try:
                        lx, ly = x, y
                        if lookup_tf:
                            lx, ly = lookup_tf.transform(lx, ly)
                        
                        c_look = int(np.floor((lx - info['xorig']) / info['xcell'])) + 1
                        r_look = int(np.floor((ly - info['yorig']) / info['ycell'])) + 1
                        
                        # Find indices in GDF that match this row/col
                        mask = (r_arr == r_look) & (c_arr == c_look)
                        cand_idx = np.where(mask)[0]
                    except Exception: 
                        cand_idx = []
                
                # Strategy 2: Spatial Index Fallback (for non-gridded or if math missed)
                if (len(cand_idx) == 0) and sindex is not None:
                    try:
                        cand_idx = list(sindex.intersection((x, y, x, y)))
                    except Exception: 
                        cand_idx = []

                if len(cand_idx) == 0:
                    return base

                pt = Point(x, y)
                best_parts = None
                best_val = -1.0
                
                for idx in cand_idx:
                    geom = geom_arr[idx]
                    # Robust containment check required for 100% accuracy near edges
                    if geom is not None and geom.contains(pt):
                        parts = [base]
                        
                        # Region/FIPS lookup
                        fips = f_arr[idx] if f_arr is not None else None
                        region_cd = re_arr[idx] if re_arr is not None else None
                        gridrc = g_arr[idx] if g_arr is not None else None
                        
                        if fips is not None and str(fips) != 'nan':
                            f_str = str(fips).split('.')[0]
                            parts.append(f"FIPS={f_str.zfill(6) if f_str.isdigit() else f_str}")
                        elif region_cd is not None and str(region_cd) != 'nan':
                            r_str = str(region_cd).split('.')[0]
                            parts.append(f"region_cd={r_str.zfill(6) if r_str.isdigit() else r_str}")
                        elif gridrc is not None:
                            parts.append(f"GRID_RC={gridrc}")
                        
                        # Value lookup (Sync with current animation view if available)
                        ax_vals = getattr(target_ax, '_smk_current_vals', None)
                        val = ax_vals[idx] if (ax_vals is not None and idx < len(ax_vals)) else (p_arr[idx] if p_arr is not None else None)
                        
                        try:
                            f_val = float(val) if not pd.isna(val) else 0.0
                            parts.append(f"{pollutant}={f_val:.4g}")
                        except Exception:
                            if val is not None: parts.append(f"{pollutant}={val}")
                            f_val = 0.0
                            
                        if f_val > best_val or best_parts is None:
                            best_val = f_val
                            best_parts = parts
                            if best_val > 0: break # Match found

                if best_parts:
                    return " | ".join(best_parts)
                return base
            except Exception as e:
                return base

        # 4. Apply to axis
        target_ax.format_coord = _fmt

    def _install_box_zoom(self, ax: plt.Axes):
        """Install interactive box zoom handles with proper connection management."""
        # 1. Cleanup old connections to prevent accumulation/interference
        if hasattr(self, '_zoom_cids'):
            for cid in self._zoom_cids:
                self.canvas.mpl_disconnect(cid)
        self._zoom_cids = []
        
        self._zoom_press = None
        self._zoom_rect = mpatches.Rectangle((0, 0), 0, 0, fill=False, ec='red', lw=1.2, zorder=9999)
        ax.add_patch(self._zoom_rect)
        self._zoom_rect.set_visible(False)
        
        def on_press(event):
            if not event.inaxes or event.inaxes != ax: return
            if self.toolbar and self.toolbar.mode: return
            if event.button not in (1, 3): return 
            self._zoom_press = (event.xdata, event.ydata, event.button)

        def on_motion(event):
            if self._zoom_press is None or event.inaxes != ax: return
            x0, y0, btn = self._zoom_press
            x1, y1 = event.xdata, event.ydata
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            self._zoom_rect.set_xy((xmin, ymin))
            self._zoom_rect.set_width(xmax - xmin)
            self._zoom_rect.set_height(ymax - ymin)
            self._zoom_rect.set_visible(True)
            self.canvas.draw_idle()

        def on_release(event):
            if self._zoom_press is None: return
            # IMPORTANT: Verify this event belongs to THIS axes to prevent stealing events from active plot
            if event.inaxes != ax and event.inaxes is not None: return
            
            x0, y0, btn = self._zoom_press
            x1, y1 = event.xdata, event.ydata
            if x1 is None: x1, y1 = x0, y0 
            
            self._zoom_press = None
            self._zoom_rect.set_visible(False)
            
            dist = np.hypot(x1 - x0, y1 - y0)
            if dist < (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01:
                 if btn == 1:
                      self._handle_cell_click(x0, y0)
                 self.canvas.draw_idle()
                 return

            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            if abs(xmax - xmin) < 1e-6 or abs(ymax - ymin) < 1e-6:
                self.canvas.draw_idle()
                return

            if btn == 1: # Zoom In
                try:
                    self.toolbar.push_current()
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                except: pass
            else: # Zoom Out (Clamped logic from gui_qt.py)
                try:
                    cur_w = ax.get_xlim()[1] - ax.get_xlim()[0]
                    cur_h = ax.get_ylim()[1] - ax.get_ylim()[0]
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    
                    s = min(box_w / cur_w, box_h / cur_h)
                    if s < 1e-6: s = 0.1 
                    
                    new_w = cur_w / s
                    new_h = cur_h / s
                    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                    
                    if hasattr(self, '_base_view') and self._base_view:
                        (bx0, bx1), (by0, by1) = self._base_view
                        bw, bh = bx1 - bx0, by1 - by0
                        if new_w >= bw or new_h >= bh:
                            self.toolbar.push_current()
                            ax.set_xlim(bx0, bx1); ax.set_ylim(by0, by1)
                            self.canvas.draw_idle()
                            return
                    
                    self.toolbar.push_current()
                    ax.set_xlim(cx - new_w/2, cx + new_w/2)
                    ax.set_ylim(cy - new_h/2, cy + new_h/2)
                except: pass
            
            self.canvas.draw_idle()
            if self.toolbar:
                if hasattr(self.toolbar, 'set_history_buttons'):
                    self.toolbar.set_history_buttons()
                elif hasattr(self.toolbar, 'update'):
                    self.toolbar.update()

        # Connect and store CIDs for future cleanup
        self._zoom_cids.append(self.canvas.mpl_connect('button_press_event', on_press))
        self._zoom_cids.append(self.canvas.mpl_connect('motion_notify_event', on_motion))
        self._zoom_cids.append(self.canvas.mpl_connect('button_release_event', on_release))
        self._zoom_cids.append(self.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion))

    def _update_stats_panel(self, gdf, col):
        """Update the side-panel stats after plotting."""
        try:
            vals = pd.to_numeric(gdf[col], errors='coerce').dropna()
            if not vals.empty:
                self.lbl_stats_sum.setText(f"{vals.sum():.4g}")
                self.lbl_stats_max.setText(f"{vals.max():.4g}")
                self.lbl_stats_mean.setText(f"{vals.mean():.4g}")
                self.lbl_stats_count.setText(str(len(vals)))
            else:
                for w in [self.lbl_stats_sum, self.lbl_stats_max, self.lbl_stats_mean, self.lbl_stats_count]:
                    w.setText("-")
        except:
            for w in [self.lbl_stats_sum, self.lbl_stats_max, self.lbl_stats_mean, self.lbl_stats_count]:
                w.setText("Error")

    def show_metadata(self):
        """Show raw metadata popup."""
        if self.emissions_df is not None:
             MetadataWindow(self.emissions_df, self).exec()

    def show_detailed_stats(self):
        """Show detailed pollutant statistics popup."""
        pol = self.cmb_pollutant.currentText()
        if not pol:
            QMessageBox.warning(self, "No Pollutant", "Please select a pollutant first.")
            return
        if self.emissions_df is not None:
             DetailedStatsWindow(self.emissions_df, pol, self).exec()

    def _filter_scc_list(self, text):
        """Filter the SCC ComboBox list based on search text."""
        if not hasattr(self, '_scc_full_list') or not self._scc_full_list:
            items = [self.cmb_scc.itemText(i) for i in range(self.cmb_scc.count())]
            self._scc_full_list = items
        
        self.cmb_scc.blockSignals(True)
        self.cmb_scc.clear()
        query = text.lower()
        filtered = [it for it in self._scc_full_list if query in it.lower()]
        if not filtered:
             filtered = ["All SCC"]
        self.cmb_scc.addItems(filtered)
        self.cmb_scc.blockSignals(False)

    def _filter_pollutant_list(self, text):
        """Filter the Pollutant ComboBox based on search text."""
        if not hasattr(self, '_full_pollutant_list'):
            self._full_pollutant_list = [self.cmb_pollutant.itemText(i) for i in range(self.cmb_pollutant.count())]
        
        current = self.cmb_pollutant.currentText()
        self.cmb_pollutant.blockSignals(True)
        self.cmb_pollutant.clear()
        
        query = text.lower()
        matches = [p for p in self._full_pollutant_list if query in p.lower()]
        if not matches:
            matches = self._full_pollutant_list
            
        self.cmb_pollutant.addItems(matches)
        
        idx = self.cmb_pollutant.findText(current)
        if idx >= 0:
            self.cmb_pollutant.setCurrentIndex(idx)
        
        self.cmb_pollutant.blockSignals(False)

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Image (*.png);;PDF (*.pdf)")
        if path:
            self.figure.savefig(path, dpi=300)
            self.status_label.setText(f"Saved to {path}")

    def pop_out_plot(self):
        """Open current plot in a separate window."""
        if self._merged_gdf is None:
            self.notify_signal.emit("WARNING", "No data to pop out.")
            return
            
        p = self.cmb_pollutant.currentText()
        m = {
            'cmap': self.cmb_cmap.currentText() + ('_r' if self.chk_rev_cmap.isChecked() else ''),
            'use_log': self.chk_log.isChecked(),
            'unit': self.units_map.get(p, ''),
            'vmin': self.txt_rmin.text() if self.txt_rmin.text() else None,
            'vmax': self.txt_rmax.text() if self.txt_rmax.text() else None
        }
        
        # Parse bins
        bt = self.txt_bins.text()
        m['bins'] = []
        if bt:
            try: m['bins'] = [float(x.strip()) for x in bt.split(',')]
            except: pass
            
        win = PlotWindow(self._merged_gdf, p, m, parent=self)
        if not hasattr(self, '_pop_wins'): self._pop_wins = []
        self._pop_wins.append(win)
        win.show()

    def export_data(self):
        """Export current data to file."""
        df = self.raw_df if self.raw_df is not None else self.emissions_df
        if df is None:
             QMessageBox.warning(self, "No Data", "No data to export.")
             return

        path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "CSV Files (*.csv);;GeoPackage (*.gpkg)")
        if path:
            try:
                self.status_label.setText(f"Exporting to {path}...")
                if path.endswith('.gpkg'):
                    # Need geometry
                    if hasattr(df, 'geometry'):
                        gpd.GeoDataFrame(df).to_file(path, driver="GPKG")
                    elif self.counties_gdf is not None:
                         # Try to join? Complex. Just warn.
                         QMessageBox.warning(self, "Geometry Missing", "Data has no geometry for GPKG. Saving as CSV instead.")
                         path = path.replace('.gpkg', '.csv')
                         df.to_csv(path, index=False)
                    else:
                        QMessageBox.warning(self, "Error", "Cannot export GPKG without geometry.")
                        return
                else:
                    df.to_csv(path, index=False)
                self.status_label.setText(f"Exported to {path}")
                self.notify_signal.emit("INFO", f"Data exported to {path}")
            except Exception as e:
                self.notify_signal.emit("ERROR", f"Export failed: {e}")

    def _get_transformers(self, crs):
        """Get transformers for graticule."""
        try:
            if not crs: return None, None
            # Project to LatLon (4326)
            tf_inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            tf_fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            return tf_fwd, tf_inv
        except:
            return None, None

    def preview_data(self):
        """Show a table preview of the current data."""
        if self.emissions_df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
            
        # Choice: Raw or Plotted
        # The original code had 'if self._merged_gdf is not None:' here.
        # The instruction provided an 'elif plot_kwargs.get('legend'):' which is syntactically incorrect
        # in this context and refers to a variable not available in this method.
        # Assuming the intent was to modify the condition for showing the choice dialog,
        # but without 'plot_kwargs' or a clear replacement, the original logic is kept
        # for _merged_gdf, as it's the most sensible interpretation given the context.
        # If the intent was to remove the choice dialog entirely or change its condition
        # based on an external 'plot_kwargs', more context would be needed.
        if self._merged_gdf is not None:
            msg = QMessageBox(self)
            msg.setWindowTitle("Preview Data")
            msg.setText("Which data would you like to preview?")
            btn_plot = msg.addButton("Plotted (Filtered/Joined)", QMessageBox.ActionRole)
            btn_raw = msg.addButton("Raw (Full File)", QMessageBox.ActionRole)
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
            
            if msg.clickedButton() == btn_plot:
                df, title = self._merged_gdf, "Plotted Data Preview"
            elif msg.clickedButton() == btn_raw:
                df, title = self.emissions_df, "Raw Data Preview"
            else:
                return
        else:
            df, title = self.emissions_df, "Raw Data Preview"
            
        self.preview_win = TableWindow(df, title=title, parent=self)
        self.preview_win.show()

    def browse_filter_shpfile(self):
        """Select filter shapefile."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Shapefile", DEFAULT_SHPFILE_INITIALDIR, "Shapefiles (*.shp *.gpkg *.geojson)")
        if path:
            if hasattr(self, 'txt_filter_shp'):
                 self.txt_filter_shp.setText(path)
            self._load_filter_shpfile(path)

    def _load_filter_shpfile(self, path):
         try:
             # Handle multiple paths
             file_list = []
             if isinstance(path, str):
                 file_list = [p.strip() for p in path.split(';') if p.strip()]
             elif isinstance(path, (list, tuple)):
                 for item in path:
                     if isinstance(item, str):
                         file_list.extend([p.strip() for p in item.split(';') if p.strip()])
                     else: file_list.append(str(item))
             else: file_list = [str(path)]
             
             if not file_list: return
             
             parts = []
             for f in file_list:
                 self.status_label.setText(f"Loading filter shape: {os.path.basename(f)}")
                 gdf = read_shpfile(f)
                 if gdf is not None and not gdf.empty:
                     if gdf.crs and gdf.crs.to_epsg() != 4326:
                         gdf = gdf.to_crs(epsg=4326)
                     parts.append(gdf)
             
             if not parts:
                 self.filter_gdf = None
                 return
                 
             if len(parts) == 1:
                 self.filter_gdf = parts[0]
             else:
                 # Combine multiple filter shapes
                 self.filter_gdf = pd.concat(parts, ignore_index=True)
                 
             self.notify_signal.emit("INFO", f"Loaded filter shape: {len(self.filter_gdf)} features from {len(parts)} files")
             self.status_label.setText("Ready")
         except Exception as e:
             self.notify_signal.emit("ERROR", f"Failed to load filter shape: {e}")

    def browse_overlay_shpfile(self):
        """Select overlay shapefile."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Overlay", DEFAULT_SHPFILE_INITIALDIR, "Shapefiles (*.shp *.gpkg *.geojson)")
        if path:
            if hasattr(self, 'txt_overlay_shp'):
                 self.txt_overlay_shp.setText(path)
            self._load_overlay_shpfile(path)

    def _load_overlay_shpfile(self, path):
         try:
             # Handle multiple paths
             file_list = []
             if isinstance(path, str):
                 file_list = [p.strip() for p in path.split(';') if p.strip()]
             elif isinstance(path, (list, tuple)):
                 for item in path:
                     if isinstance(item, str):
                         file_list.extend([p.strip() for p in item.split(';') if p.strip()])
                     else: file_list.append(str(item))
             else: file_list = [str(path)]
             
             if not file_list: return
             
             parts = []
             for f in file_list:
                 self.status_label.setText(f"Loading overlay: {os.path.basename(f)}")
                 gdf = read_shpfile(f)
                 if gdf is not None and not gdf.empty:
                     if gdf.crs and gdf.crs.to_epsg() != 4326:
                         gdf = gdf.to_crs(epsg=4326)
                     parts.append(gdf)
             
             if not parts:
                 self.overlay_gdf = None
                 return
                 
             # Store as list if multiple, plotter handles it
             self.overlay_gdf = parts if len(parts) > 1 else parts[0]
             self.notify_signal.emit("INFO", f"Loaded overlay: {len(parts)} files")
             self.status_label.setText("Ready")
         except Exception as e:
             self.notify_signal.emit("ERROR", f"Failed to load overlay: {e}")

    def _on_preview_summary(self):
        try:
            df = self._build_summary()
            self.sum_win = TableWindow(df, title=f"Summary Preview ({self.cmb_summary_mode.currentText()})", parent=self)
            self.sum_win.show()
        except Exception as e:
            self._handle_notification("ERROR", f"Summary failed: {e}")

    def _on_export_summary(self):
        try:
            df = self._build_summary()
            path, _ = QFileDialog.getSaveFileName(self, "Export Summary", "", "CSV Files (*.csv)")
            if path:
                df.to_csv(path, index=False)
                self.status_label.setText(f"Summary saved to {path}")
        except Exception as e:
            self._handle_notification("ERROR", f"Export failed: {e}")

    def _build_summary(self):
        if self.emissions_df is None:
            raise ValueError("No data loaded.")
        
        mode = self.cmb_summary_mode.currentText().lower()
        raw = self.raw_df if self.raw_df is not None else self.emissions_df
        
        # Detect pollutants
        pols = detect_pollutants(raw)
        if not pols:
            pols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
            
        group_cols = []
        if mode == 'county':
            fips_col = next((c for c in raw.columns if c.lower() in ['fips', 'region_cd']), None)
            if not fips_col: raise ValueError("No FIPS column found.")
            group_cols = [fips_col]
        elif mode == 'state':
            fips_col = next((c for c in raw.columns if c.lower() in ['fips', 'region_cd']), None)
            if not fips_col: 
                # Fallback to checking any column that might look like FIPS
                fips_col = next((c for c in raw.columns if 'fips' in c.lower()), None)
            
            if not fips_col: raise ValueError("No FIPS column found for state summary.")
            # Local copy to add state
            raw = raw.copy()
            raw['STATEFP'] = raw[fips_col].astype(str).str.zfill(6).str[1:3]
            group_cols = ['STATEFP']
        elif mode == 'scc':
            scc_col = next((c for c in raw.columns if c.lower() in ['scc', 'scc code']), None)
            if not scc_col: raise ValueError("No SCC column found.")
            desc_cols = [c for c in raw.columns if c.lower() in ['scc description', 'scc_description']]
            group_cols = [scc_col] + ([desc_cols[0]] if desc_cols else [])
        elif mode == 'grid':
            if 'ROW' in raw.columns and 'COL' in raw.columns:
                group_cols = ['ROW', 'COL']
            elif 'GRID_RC' in raw.columns:
                group_cols = ['GRID_RC']
            else:
                raise ValueError("No grid indices found.")
                
        # Aggregate
        summary = raw.groupby(group_cols, as_index=False)[pols].sum()
        
        # Enrich state names if needed
        if 'STATEFP' in summary.columns:
             summary.insert(1, 'STATE_NAME', summary['STATEFP'].map(US_STATE_FIPS_TO_NAME))
             
        return summary

class TimeSeriesPlotWindow(QDialog):
    def __init__(self, data, title, pollutant, unit, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Time Series - {title}")
        self.resize(800, 500)
        self.pollutant = pollutant
        self.unit = unit
        
        layout = QVBoxLayout(self)
        
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        self.plot_data(data)
        
    def plot_data(self, data):
        ax = self.figure.add_subplot(111)
        times = data.get('times', [])
        vals = data.get('values', [])
        
        # Convert times to datetime if string
        try:
            if times and isinstance(times[0], str):
                times = [pd.to_datetime(t) for t in times]
        except: pass

        if isinstance(vals, dict):
            # Check length of first series to validate times
            v_len = 0
            if vals: v_len = len(list(vals.values())[0])
            
            if len(times) != v_len:
                times = range(v_len)

            # Multi-line plot
            for label, series in vals.items():
                lw = 2.5 if label == 'Total' else 1.0
                alpha = 1.0 if label == 'Total' else 0.6
                ms = 4 if label == 'Total' else 2
                ax.plot(times, series, marker='o', markersize=ms, label=label, linewidth=lw, alpha=alpha)
            
            if len(vals) < 25:
                ax.legend(fontsize='small', loc='upper right')
            else:
                try: ax.legend(['Total'], loc='upper right')
                except: pass
        else:
            # Single series (list or array)
            if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], (list, np.ndarray)):
                 # If list of lists (single cell raw), take first
                 vals = vals[0]
            
            # Fallback for times
            if len(times) != len(vals):
                times = range(len(vals))

            ax.plot(times, vals, marker='o', linestyle='-', markersize=4)

        u_str = str(self.unit or '').strip()
        y_lbl = f"{self.pollutant} ({u_str})" if u_str else self.pollutant
        ax.set_ylabel(y_lbl)
        ax.set_xlabel("Time Step")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Date formatting
        import matplotlib.dates as mdates
        if len(times) > 0:
             try:
                 ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                 self.figure.autofmt_xdate()
             except: pass

        self.canvas.draw()

def main():
    # Use 'spawn' start method to avoid deadlocks when using ProcessPoolExecutor from threads
    if sys.platform != 'win32':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except Exception:
            pass

    app = QApplication(sys.argv)
    
    # Modern styling (Dark Theme approximation)
    app.setStyle("Fusion")
    palette = QColor(50, 50, 50)
    # ... (Full palette setup could go here)
    
    # Parse args manually or via smkplot.py passing
    # Here we assume class handles it or we parse sys.argv
    
    gui = NativeEmissionGUI()
    gui.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
