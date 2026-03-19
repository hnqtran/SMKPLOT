#!/usr/bin/env python3
# Author: tranhuy@email.unc.edu
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

# --- Qt GUI Framework Initialization ---
try:
    from PySide6 import QtWidgets, QtCore, QtGui
    from PySide6.QtCore import Qt, Signal, Slot, QObject, QThread, QTimer, QSize, QEvent, QSettings, QRunnable, QThreadPool
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, 
        QFileDialog, QMessageBox, QProgressBar, QTabWidget, 
        QSplitter, QFrame, QSizePolicy, QScrollArea, QGridLayout,
        QMenu, QMenuBar, QStatusBar, QListWidget, QTextEdit,
        QLayout, QTreeWidget, QTreeWidgetItem, QStyle, QListView,
        QDockWidget, QToolBar, QDialog, QDialogButtonBox, QFormLayout,
        QSpinBox, QDoubleSpinBox, QGroupBox, QTableWidget, QTableWidgetItem,
        QAbstractItemView
    )
    from PySide6.QtGui import QAction, QIcon, QFont, QIntValidator, QDoubleValidator, QTextCursor, QColor
    QT_BINDING = "PySide6"
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot, QObject, QThread, QTimer, QSize, QEvent, QSettings
    from PyQt5.QtCore import QRunnable, QThreadPool
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, 
        QFileDialog, QMessageBox, QProgressBar, QTabWidget, 
        QSplitter, QFrame, QSizePolicy, QScrollArea, QGridLayout,
        QMenu, QMenuBar, QStatusBar, QListWidget, QTextEdit,
        QLayout, QTreeWidget, QTreeWidgetItem, QStyle, QListView,
        QDockWidget, QToolBar, QDialog, QDialogButtonBox, QFormLayout,
        QSpinBox, QDoubleSpinBox, QGroupBox, QTableWidget, QTableWidgetItem,
        QAbstractItemView
    )
    from PyQt5.QtGui import QIcon, QFont, QIntValidator, QDoubleValidator, QTextCursor, QColor
    try: from PyQt5.QtWidgets import QAction
    except ImportError: from PyQt5.QtGui import QAction
    QT_BINDING = "PyQt5"

import shapely
# --- Matplotlib Backend Setup ---
import matplotlib
if QT_BINDING == "PySide6":
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    except (ImportError, TypeError):
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
else:
    # Use Qt5Agg backends for PyQt5
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    except (ImportError, TypeError):
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
        DEFAULT_ONLINE_COUNTIES_URL, SCC_COLS, DESC_COLS
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

    # --- Graticule Logic with Zoom Support ---
    import plotting
    
    def _fixed_draw_graticule(ax, tf_fwd, tf_inv, lon_step=None, lat_step=None, with_labels=True):
        """Implementation of graticule drawing optimized for map projection zoom."""
        artists = {'lines': [], 'texts': []}
        if ax is None or tf_fwd is None or tf_inv is None:
            return artists

        # Use internal flags for separate operation phases if needed, 
        # but don't block entry if caller managed the guard.
        try:
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            if not (np.isfinite([x0, x1, y0, y1]).all()): return artists
            
            # Detect Lat/Lon span to determine nice steps
            sx = np.array([x0, x1, x1, x0, 0.5*(x0+x1)])
            sy = np.array([y0, y0, y1, y1, 0.5*(y0+y1)])
            sample_lons, sample_lats = tf_inv.transform(sx, sy)
            valid = (np.abs(sample_lons) < 181.01) & (np.abs(sample_lats) < 91.01)
            if not valid.any(): return artists
            
            lons, lats = sample_lons[valid], sample_lats[valid]
            lon_span, lat_span = lons.max() - lons.min(), lats.max() - lats.min()
            
            def _nice_step(span):
                if span <= 0: return 1.0
                target = span / 5.0
                power = 10 ** np.floor(np.log10(target))
                base = target / power
                if base < 1.5: step = 1.0 * power
                elif base < 3.5: step = 2.0 * power
                else: step = 5.0 * power
                return max(step, 1e-6)

            actual_lon_step = lon_step or _nice_step(lon_span)
            actual_lat_step = lat_step or _nice_step(lat_span)
            
            # Use relative buffer: extend half-step beyond view to ensure crossing
            buf_lon = actual_lon_step * 0.5
            buf_lat = actual_lat_step * 0.5
            
            lon_min, lon_max = lons.min() - buf_lon, lons.max() + buf_lon
            lat_min, lat_max = lats.min() - buf_lat, lats.max() + buf_lat
            
            # Use high density sampling for line continuity
            lats_samp = np.linspace(max(-90, lat_min), min(90, lat_max), 505)
            lons_samp = np.linspace(max(-180, lon_min), min(180, lon_max), 505)
            
            old_autoscale = ax.get_autoscale_on()
            ax.set_autoscale_on(False)
            
            # Helper for degree labels
            def _fmt(v, step):
                p = 0 if step >= 1 else (1 if step >= 0.1 else (2 if step >= 0.01 else 3))
                val = abs(v)
                s = f"{val:.{p}f}".rstrip('0').rstrip('.') if p > 0 else f"{int(round(val))}"
                return f"{s}\u00b0"

            # 1. Longitude Lines
            l_start = np.floor(lon_min / actual_lon_step) * actual_lon_step
            for lon in np.arange(l_start, lon_max + actual_lon_step, actual_lon_step):
                if lon < -180.001 or lon > 180.001: continue
                xs, ys = tf_fwd.transform(np.full_like(lats_samp, lon), lats_samp)
                
                # High zorder (20) to stay above everything
                lines = ax.plot(xs, ys, color='#cccccc', linewidth=0.5, alpha=0.5, zorder=20)
                artists['lines'].extend(lines)
                
                if with_labels:
                    mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
                    if mask.any():
                        idx = np.argmin(np.abs(ys[mask] - y0))
                        lbl = _fmt(lon, actual_lon_step) + ('W' if lon < 0 else 'E' if lon > 0 else '')
                        t = ax.text(xs[mask][idx], y0 + 0.005*(y1-y0), lbl, 
                                    fontsize=7, color='#666666', ha='center', va='bottom', zorder=21)
                        t.set_clip_on(True)
                        artists['texts'].append(t)

            # 2. Latitude Lines
            l_start = np.floor(lat_min / actual_lat_step) * actual_lat_step
            for lat in np.arange(l_start, lat_max + actual_lat_step, actual_lat_step):
                if lat < -90.001 or lat > 90.001: continue
                xs, ys = tf_fwd.transform(lons_samp, np.full_like(lons_samp, lat))
                lines = ax.plot(xs, ys, color='#cccccc', linewidth=0.5, alpha=0.5, zorder=20)
                artists['lines'].extend(lines)
                
                if with_labels:
                    mask = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
                    if mask.any():
                        idx = np.argmin(np.abs(xs[mask] - x0))
                        lbl = _fmt(lat, actual_lat_step) + ('S' if lat < 0 else 'N' if lat > 0 else '')
                        t = ax.text(x0 + 0.005*(x1-x0), ys[mask][idx], lbl, 
                                    fontsize=7, color='#666666', ha='left', va='center', zorder=21)
                        t.set_clip_on(True)
                        artists['texts'].append(t)
            
            ax.set_autoscale_on(old_autoscale)
        except Exception as e:
            logging.debug(f"Graticule Draw Error: {e}")
        return artists

    plotting._draw_graticule = _fixed_draw_graticule
    _draw_graticule_fn = _fixed_draw_graticule

    _orig_create_map_plot = plotting.create_map_plot
    def _guarded_create_map_plot(*args, **kwargs):
        # Allow disabling quadmesh via kwargs even if plotting.py doesn't natively support the toggle
        disable_qm = kwargs.pop('disable_quadmesh', False)
        force_qm = kwargs.pop('force_quadmesh', False)
        gdf = args[0] if args else kwargs.get('gdf')
        
        if disable_qm and gdf is not None:
            # Temporarily hide _smk_grid_info to trick plotting.py into using standard plotting
            orig_info = gdf.attrs.get('_smk_grid_info')
            if orig_info:
                gdf.attrs['_smk_grid_info'] = None
                try:
                    res = _orig_create_map_plot(*args, **kwargs)
                finally:
                    gdf.attrs['_smk_grid_info'] = orig_info
                return res
        
        # Guard internal _draw_graticule calls inside create_map_plot
        ax = kwargs.get('ax')
        if not ax and len(args) > 3: ax = args[3]
        if ax:
             if getattr(ax, '_smk_drawing_graticule', False):
                  return _orig_create_map_plot(*args, **kwargs)
             ax._smk_drawing_graticule = True
             try:
                  return _orig_create_map_plot(*args, **kwargs)
             finally:
                  ax._smk_drawing_graticule = False
        
        return _orig_create_map_plot(*args, **kwargs)
    
    plotting.create_map_plot = _guarded_create_map_plot
    create_map_plot = _guarded_create_map_plot
    # --------------------------------------------------------
except ImportError:
    # Fallback for when running directly
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import *
    from utils import *
    from data_processing import *
    from plotting import *
    from ncf_processing import *

def _get_adaptive_window_size(preferred_w, preferred_h, min_w=800, min_h=600, scale_factor=0.85):
    """Scale window dimensions based on available screen space.
    
    Args:
        preferred_w, preferred_h: Desired window dimensions (for large screens)
        min_w, min_h: Minimum acceptable dimensions
        scale_factor: Scale to available space (0.85 = 85% of available screen)
    
    Returns:
        Tuple (width, height) adapted to available screen space.
    """
    try:
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            avail_w = geom.width()
            avail_h = geom.height()
            
            # Scale to percentage of available space
            scaled_w = int(avail_w * scale_factor)
            scaled_h = int(avail_h * scale_factor)
            
            # Respect minimums
            final_w = max(min_w, min(scaled_w, preferred_w))
            final_h = max(min_h, min(scaled_h, preferred_h))
            return final_w, final_h
    except Exception:
        pass
    
    return preferred_w, preferred_h

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

class WorkerSignals(QObject):
    """Signals for background workers."""
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)

class Worker(QRunnable):
    """Generic worker for running functions in background threads."""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            # Pass the progress signal emitter as the progress_callback
            result = self.fn(*self.args, **self.kwargs, progress_callback=self.signals.progress.emit)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

class TableWindow(QMainWindow):
    """Modern window to display a DataFrame in a searchable, sortable table."""
    def __init__(self, df=None, title="Data Preview", parent=None, modes=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        w, h = _get_adaptive_window_size(1100, 700, min_w=750, min_h=450)
        self.resize(w, h)
        self.modes = modes or {}
        self.current_df = df
        
        # UI Setup
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header Area
        header_layout = QHBoxLayout()
        
        # View Selector (if modes available)
        if self.modes:
            header_layout.addWidget(QLabel("View:"))
            self.cmb_view = QComboBox()
            self.cmb_view.addItems(list(self.modes.keys()))
            self.cmb_view.currentTextChanged.connect(self._on_view_changed)
            header_layout.addWidget(self.cmb_view)
            header_layout.addSpacing(20)
            
        self.info_lbl = QLabel("")
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
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
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
        btn_copy.setShortcut("Ctrl+C") # Keyboard shortcut
        btn_copy.clicked.connect(self._copy_selection)
        footer_layout.addWidget(btn_copy)
        
        footer_layout.addStretch()
        
        btn_export = QPushButton("Export to CSV")
        btn_export.setObjectName("primaryBtn")
        btn_export.clicked.connect(self.export_current_csv)
        footer_layout.addWidget(btn_export)
        
        layout.addLayout(footer_layout)
        
        # Initial Populate
        if self.modes:
            # Trigger first view
            self.cmb_view.setCurrentIndex(0)
            self._on_view_changed(self.cmb_view.currentText())
        elif self.current_df is not None:
            self._populate_table(self.current_df)

    def _populate_table(self, df):
        self.current_df = df
        self.table.setSortingEnabled(False) # Disable during populate
        self.table.clear()
        
        if df is None: 
            self.table.setRowCount(0); self.table.setColumnCount(0)
            self.info_lbl.setText("No data available")
            return

        df_show = df.head(100) 
        self.table.setRowCount(len(df_show))
        self.table.setColumnCount(len(df_show.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df_show.columns])
        
        self.info_lbl.setText(f"Showing {len(df_show):,} of {len(df):,} rows | {len(df.columns)} Columns")
        
        self._all_items = []
        for i, row in enumerate(df_show.itertuples(index=False)):
            row_items = []
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                self.table.setItem(i, j, item)
                row_items.append(item)
            self._all_items.append(row_items)
            
        self.table.setSortingEnabled(True)

    def _on_view_changed(self, view_name):
        if view_name not in self.modes: return
        data = self.modes[view_name]
        
        # Resolve lazy loading (callable)
        if callable(data):
            try:
                df = data()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load view: {e}")
                return
        else:
            df = data
            
        self._populate_table(df)

    def export_current_csv(self):
        if self.current_df is not None:
             self.export_csv(self.current_df)

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
        w, h = _get_adaptive_window_size(600, 500, min_w=450, min_h=350)
        self.resize(w, h)
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
        w, h = _get_adaptive_window_size(800, 400, min_w=650, min_h=350)
        self.resize(w, h)
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

class MultiSelectionDialog(QDialog):
    """Searchable dialog for selecting multiple items with checkboxes."""
    def __init__(self, title, items, selected=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        w, h = _get_adaptive_window_size(500, 600, min_w=400, min_h=450)
        self.resize(w, h)
        self.selected = list(selected or [])
        
        layout = QVBoxLayout(self)
        
        # Search Box
        search_layout = QHBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Find items...")
        self.txt_search.textChanged.connect(self._filter_items)
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.txt_search)
        layout.addLayout(search_layout)
        
        # List with checkboxes
        self.list_widget = QListWidget()
        self.items_data = items # Full list
        self._populate_list(self.items_data)
        layout.addWidget(self.list_widget)
        
        # Action Buttons
        row_btns = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton("Clear All")
        btn_none.clicked.connect(self._clear_all)
        row_btns.addWidget(btn_all)
        row_btns.addWidget(btn_none)
        row_btns.addStretch()
        layout.addLayout(row_btns)
        
        # Standard Dialog Buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _populate_list(self, items):
        self.list_widget.clear()
        for it in items:
            list_item = QtWidgets.QListWidgetItem(it)
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            if it in self.selected:
                list_item.setCheckState(Qt.Checked)
            else:
                list_item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(list_item)

    def _filter_items(self, text):
        query = text.lower().strip()
        # Update current selected state from list BEFORE clearing
        self._sync_selected()
        
        filtered = [it for it in self.items_data if query in it.lower()]
        self._populate_list(filtered)

    def _sync_selected(self):
        """Update self.selected from current list widget states."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            text = item.text()
            if item.checkState() == Qt.Checked:
                if text not in self.selected: self.selected.append(text)
            else:
                if text in self.selected: self.selected.remove(text)

    def _select_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)
        self._sync_selected()

    def _clear_all(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)
        self.selected = []

    def get_selected(self):
        self._sync_selected()
        return self.selected

class PlotWindow(QMainWindow):
    """Pop-out window for a specific plot."""
    def __init__(self, gdf, column, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Plot: {column}")
        w, h = _get_adaptive_window_size(1000, 800, min_w=800, min_h=600)
        self.resize(w, h)
        
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
        # Aspect ratio is handled by Figure size and create_map_plot management
        
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
    stop_progress_signal = Signal() # New signal to re-enable UI safely
    
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

        # --------------------------------------------------

        self.setWindowTitle(f"SMKPLOT v{app_version} (Native Qt) (Author: tranhuy@email.unc.edu)")
        w, h = _get_adaptive_window_size(1600, 900, min_w=1100, min_h=650)
        self.resize(w, h)
        
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
        self._loading_active = False
        
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
        self._ncf_refresh_active = False
        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(lambda: self._step_time(1))
        
        # --- Thread Pool ---
        self.threadpool = QThreadPool()
        
        # Respect --workers or cap at 8 to prevent overwhelming HPC nodes
        req_workers = self._json_arguments.get('workers', 0)
        if req_workers and req_workers > 0:
            self.threadpool.setMaxThreadCount(int(req_workers))
        else:
            self.threadpool.setMaxThreadCount(min(8, os.cpu_count() or 1))

        logging.info(f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads")
        
        # Add a debouncer for plot title/stats updates to maintain responsiveness
        self._title_timer = QTimer()
        self._title_timer.setSingleShot(True)
        self._title_timer.timeout.connect(self._exec_title_update)
        self._current_ax_for_title = None

        # UI State
        self.class_bins_var = "" 
        self._last_loaded_delim_state = None
        self._scc_display_to_code = {} # Mapping for display-to-code filtering
        self.selected_sccs = [] # For multiple SCC selection
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
        self.stop_progress_signal.connect(self._stop_progress)
        
        # --- Auto-Load if arguments present ---
        QTimer.singleShot(100, self._startup_load)

        # Debug: Log initialization
        if hasattr(self.cli_args, 'debug') and getattr(self.cli_args, 'debug', False):
            logging.debug("=" * 70)
            logging.debug("NATIVE EMISSION GUI INITIALIZATION")
            logging.debug("=" * 70)
            logging.debug(f"Input file: {self.inputfile_path}")
            logging.debug(f"Counties shapefile: {self.counties_path}")
            logging.debug(f"Grid name: {self.grid_name}")
            logging.debug(f"GRIDDESC: {self.griddesc_path}")
            logging.debug(f"Preselected pollutant: {self.preselected_pollutant}")
            logging.debug(f"Delimiter: {self.emissions_delim}")

    def _apply_styles(self):
        """Apply a high-contrast, premium 'Slate & Indigo' theme."""
        style = """
            /* Global Base */
            QWidget { 
                font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif; 
                font-size: 12px; 
                color: #0F172A; 
                background: #F8FAFC;
            }
            
            QMainWindow, QDialog, QMessageBox {
                background-color: #F8FAFC;
            }

            /* Group Box - Modern Panel Style */
            QGroupBox {
                font-weight: 700; 
                border: 1px solid #E2E8F0;
                border-radius: 8px; 
                margin-top: 1.2em; 
                padding-top: 15px;
                background-color: #FFFFFF;
                color: #1E293B;
            }
            QGroupBox::title {
                subcontrol-origin: margin; 
                left: 12px; 
                padding: 0 8px; 
                color: #4F46E5;
            }
            
            /* Inputs - High Contrast */
            QLineEdit, QComboBox, QSpinBox, QTextEdit, QPlainTextEdit {
                background-color: #FFFFFF;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 6px 10px;
                min-height: 28px;
            }
            QLineEdit:focus, QComboBox:focus { 
                border: 2px solid #6366F1; 
                background-color: #F8FAFC;
            }
            
            QComboBox::drop-down { 
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left: 1px solid #E2E8F0;
            }
            
            /* Buttons - Premium Feel */
            QPushButton {
                background-color: #FFFFFF;
                color: #334155;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover { 
                background-color: #F1F5F9; 
                border-color: #94A3B8; 
                color: #0F172A;
            }
            
            QPushButton#primaryBtn {
                background-color: #4F46E5;
                color: #FFFFFF !important;
                border: 1px solid #4338CA;
            }
            QPushButton#primaryBtn:hover { background-color: #4338CA; }
            QPushButton#primaryBtn:pressed { background-color: #3730A3; }
            QPushButton#primaryBtn:disabled { background-color: #E2E8F0; color: #94A3B8; }

            /* Tabs - Clean & Minimal */
            QTabWidget::pane { 
                border: 1px solid #E2E8F0; 
                top: -1px; 
                background: #FFFFFF; 
                border-radius: 0 0 8px 8px; 
            }
            QTabBar::tab {
                background: #F1F5F9; 
                border: 1px solid #E2E8F0;
                padding: 10px 20px; 
                margin-right: 2px;
                border-top-left-radius: 6px; 
                border-top-right-radius: 6px;
                color: #64748B;
            }
            QTabBar::tab:selected { 
                background: #FFFFFF; 
                border-bottom-color: #FFFFFF; 
                color: #4F46E5;
                font-weight: 700;
            }
            
            /* Scrollbars - Subtle */
            QScrollBar:vertical {
                border: none;
                background: #F1F5F9;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                min-height: 30px;
                border-radius: 6px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover { background: #94A3B8; }

            /* Logs & Progress */
            QTextEdit#log_text { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; }

            /* Status Bar */
            QStatusBar { 
                background-color: #FFFFFF; 
                border-top: 1px solid #E2E8F0; 
                color: #64748B;
                min-height: 32px;
            }
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
        
        # Page 1: Data Source & View Settings
        page_source = QWidget(); l_source = QVBoxLayout(page_source)
        l_source.setContentsMargins(6, 6, 6, 6)
        l_source.setSpacing(2)
        
        # Source Sections
        self._init_inputs_section(l_source)
        self._init_variable_section(l_source)
        
        # View / Plot Settings (Moved from separate tab)
        self._init_plot_settings_section(l_source)
        


        
        l_source.addStretch()
        self.tabs.addTab(page_source, "Source & View")
        
        # Page 2: Filtering
        page_filter = QWidget(); l_filter = QVBoxLayout(page_filter)
        l_filter.setContentsMargins(6, 6, 6, 6)
        self._init_filter_section(l_filter)
        l_filter.addStretch()
        self.tabs.addTab(page_filter, "Filter")
        
        # Page 3: Analysis (Stats)
        page_stats = QWidget(); l_stats = QVBoxLayout(page_stats)
        l_stats.setContentsMargins(6, 6, 6, 6)
        self._init_summary_section(l_stats)
        self._init_stats_panel(l_stats)
        l_stats.addStretch()
        self.tabs.addTab(page_stats, "Stats")

        left_layout.addWidget(self.tabs)
        
        # Always-visible footer buttons
        self.footer_frame = QFrame()
        self.footer_frame.setFrameShape(QFrame.NoFrame)
        self.footer_frame.setMinimumHeight(60)
        footer_layout = QHBoxLayout(self.footer_frame)
        footer_layout.setContentsMargins(5, 5, 5, 5)
        footer_layout.setSpacing(10)
        
        self.btn_main_plot = QPushButton("GENERATE PLOT")
        self.btn_main_plot.setObjectName("primaryBtn")
        self.btn_main_plot.setMinimumHeight(45)
        self.btn_main_plot.clicked.connect(self._on_main_plot_clicked)
        
        footer_layout.addWidget(self.btn_main_plot, 2)

        self.btn_export = QPushButton("Export Configuration")
        self.btn_export.setToolTip("Save current settings to a YAML configuration file")
        self.btn_export.setMinimumHeight(45)
        self.btn_export.clicked.connect(self.export_configuration)
        footer_layout.addWidget(self.btn_export, 1)
        
        left_layout.addWidget(self.footer_frame)
        
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
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #3b82f6; border-radius: 2px; }")
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(self.progress_bar)
        
        right_splitter.addWidget(log_group)
        right_splitter.setStretchFactor(0, 5)
        right_splitter.setStretchFactor(1, 1)
        
        self.main_splitter.addWidget(right_splitter)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 5)
        self.main_splitter.setSizes([550, 1050])
        
        main_layout.addWidget(self.main_splitter)

        # --- Modern Status Bar Setup ---
        self._setup_modern_status_bar()

    def _setup_modern_status_bar(self):
        """Create a segmented, modern status bar with telemetry."""
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        
        # Left Side: Icon + Message
        self.status_icon = QLabel("●")
        self.status_icon.setObjectName("statusIcon")
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusMsg")
        
        sb.addWidget(self.status_icon)
        sb.addWidget(self.status_label)
        
        # Permanent Widgets (Right Alignment)
        self.lbl_grid_info = QLabel("Grid: --")
        self.lbl_grid_info.setObjectName("statusStat")
        
        self.lbl_threads = QLabel("Threads: 1")
        self.lbl_threads.setObjectName("statusStat")
        
        self.lbl_mem = QLabel("RAM: --")
        self.lbl_mem.setObjectName("statusStat")
        
        self.lbl_version = QLabel(f"SMKPLOT v2.0")
        self.lbl_version.setObjectName("statusStat")
        
        sb.addPermanentWidget(self.lbl_grid_info)
        sb.addPermanentWidget(self.lbl_threads)
        sb.addPermanentWidget(self.lbl_mem)
        sb.addPermanentWidget(self.lbl_version)
        
        # Background Timer for Stats
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_sb_telemetry)
        self._stats_timer.start(3000) # Every 3s

    def _update_sb_telemetry(self):
        """Refresh system stats in the status bar."""
        try:
            # Active Threads
            tc = threading.active_count()
            self.lbl_threads.setText(f"Threads: {tc}")
            
            # Memory (Estimate if psutil missing)
            import os
            try:
                import psutil
                mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                self.lbl_mem.setText(f"RAM: {mem:.0f} MB")
            except ImportError:
                # Fallback: very basic memory estimation for Linux
                try:
                    with open('/proc/self/status') as f:
                        for line in f:
                            if line.startswith('VmRSS:'):
                                rss = int(line.split()[1]) / 1024
                                self.lbl_mem.setText(f"RAM: {rss:.0f} MB")
                                break
                except:
                    self.lbl_mem.setText("RAM: N/A")
        except Exception:
            pass

    def _set_status_busy(self, is_busy=True, msg=None):
        """Update status bar appearance based on system state."""
        if is_busy:
            self.status_icon.setText("○")
            self.status_icon.setStyleSheet("color: #3b82f6;") # Blue
            if msg: self.status_label.setText(msg)
        else:
            self.status_icon.setText("●")
            self.status_icon.setStyleSheet("color: #10b981;") # Green
            if msg: self.status_label.setText(msg)
        QApplication.processEvents()

    def _init_inputs_section(self, parent_layout=None):
        group = QGroupBox("1. Input Data")
        
        # Main vertical layout for the group
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 15, 10, 10)
        layout.setSpacing(8)

        # --- A. Primary File Input ---
        file_box = QWidget()
        l_file = QVBoxLayout(file_box) 
        l_file.setContentsMargins(0, 0, 0, 0); l_file.setSpacing(4)
        
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Path to Emissions File (CSV, NetCDF, Excel)...")
        self.txt_input.setToolTip("Drag & Drop supported")
        self.txt_input.editingFinished.connect(self._on_input_path_edit)
        
        btn_browse_input = QPushButton("Browse...")
        btn_browse_input.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        btn_browse_input.clicked.connect(self.select_input_file)
        
        row_input = QHBoxLayout()
        row_input.addWidget(self.txt_input)
        row_input.addWidget(btn_browse_input)
        
        l_file.addLayout(row_input)
        
        # Tools Row (Load, Preview, Meta)
        row_tools = QHBoxLayout()
        self.btn_load_data = QPushButton("Load Data")
        self.btn_load_data.setObjectName("primaryBtn")
        self.btn_load_data.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.btn_load_data.clicked.connect(lambda: self.load_input_file(self.txt_input.text()))
        
        self.btn_preview = QPushButton("View Table")
        self.btn_preview.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.btn_preview.clicked.connect(self.preview_data)
        
        self.btn_metadata = QPushButton("Metadata")
        self.btn_metadata.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        self.btn_metadata.clicked.connect(self.show_metadata)
        
        row_tools.addWidget(self.btn_load_data, 2)
        row_tools.addWidget(self.btn_preview, 1)
        row_tools.addWidget(self.btn_metadata, 1)
        l_file.addLayout(row_tools)
        
        layout.addWidget(file_box)
        
        # --- B. configuration Form (Grid, Counties, Parsing) ---
        form_frame = QFrame()
        form_frame.setStyleSheet("QFrame { background: #f8fafc; border-radius: 4px; border: 1px solid #e2e8f0; }")
        l_form = QGridLayout(form_frame)
        l_form.setContentsMargins(8, 8, 8, 8)
        l_form.setVerticalSpacing(6)
        l_form.setHorizontalSpacing(10)
        
        # Row 1: Delimiter & Skip & Comment
        l_form.addWidget(QLabel("Delimiter:"), 0, 0)
        self.cmb_delim = QComboBox(); self.cmb_delim.setFixedWidth(100)
        self.cmb_delim.addItems(['auto', 'comma', 'semicolon', 'tab', 'pipe', 'space', 'other'])
        self.cmb_delim.currentTextChanged.connect(self._on_delim_toggle)
        self.txt_custom_delim = QLineEdit(); self.txt_custom_delim.setPlaceholderText("Char"); self.txt_custom_delim.setVisible(False); self.txt_custom_delim.setFixedWidth(30)
        d_box = QHBoxLayout(); d_box.setContentsMargins(0,0,0,0); d_box.setSpacing(2)
        d_box.addWidget(self.cmb_delim); d_box.addWidget(self.txt_custom_delim)
        l_form.addLayout(d_box, 0, 1)
        
        l_form.addWidget(QLabel("# Row Skip:"), 0, 2)
        self.spin_skip = QSpinBox(); self.spin_skip.setRange(0, 100); self.spin_skip.setFixedWidth(60)
        l_form.addWidget(self.spin_skip, 0, 3)

        l_form.addWidget(QLabel("Comment:"), 0, 4)
        self.txt_comment = QLineEdit("#"); self.txt_comment.setFixedWidth(40); self.txt_comment.setAlignment(Qt.AlignCenter)
        l_form.addWidget(self.txt_comment, 0, 5)
        
        # Row 2: GRIDDESC & Selection
        l_form.addWidget(QLabel("GRIDDESC:"), 1, 0)
        self.txt_griddesc = QLineEdit()
        self.txt_griddesc.setPlaceholderText("Path to GRIDDESC")
        self.txt_griddesc.editingFinished.connect(self._on_griddesc_path_edit)
        self.btn_browse_gd = QPushButton("..."); self.btn_browse_gd.setFixedWidth(30); self.btn_browse_gd.clicked.connect(self.select_griddesc_file)
        g_box = QHBoxLayout(); g_box.setContentsMargins(0,0,0,0); g_box.setSpacing(2)
        g_box.addWidget(self.txt_griddesc); g_box.addWidget(self.btn_browse_gd)
        l_form.addLayout(g_box, 1, 1, 1, 5) # Spans all 5 input columns

        # Row 3: Grid Name Selection
        l_form.addWidget(QLabel("Grid Name:"), 2, 0)
        self.cmb_gridname = QComboBox()
        self.cmb_gridname.addItem("Select Grid")
        self.cmb_gridname.currentTextChanged.connect(self._grid_name_changed)
        l_form.addWidget(self.cmb_gridname, 2, 1, 1, 5)

        # Row 4: County Shapefile
        l_form.addWidget(QLabel("Counties:"), 3, 0)
        self.txt_counties = QLineEdit()
        self.txt_counties.setPlaceholderText("Shp Path / Online Year")
        self.txt_counties.editingFinished.connect(self._on_counties_path_edit)
        self.btn_browse_counties = QPushButton("..."); self.btn_browse_counties.setFixedWidth(30); self.btn_browse_counties.clicked.connect(self.select_county_file)
        c_box = QHBoxLayout(); c_box.setContentsMargins(0,0,0,0); c_box.setSpacing(2)
        c_box.addWidget(self.txt_counties); c_box.addWidget(self.btn_browse_counties)
        l_form.addLayout(c_box, 3, 1, 1, 5)
        
        # Row 5: Online Counties (Fetch)
        l_form.addWidget(QLabel("Online Map:"), 4, 0)
        self.cmb_online_year = QComboBox()
        self.cmb_online_year.addItems(['2020', '2023'])
        self.btn_fetch_online = QPushButton("Fetch")
        self.btn_fetch_online.setToolTip("Download US Counties shapefile from Census.gov")
        self.btn_fetch_online.clicked.connect(self.use_online_counties)
        
        o_box = QHBoxLayout(); o_box.setContentsMargins(0,0,0,0)
        o_box.addWidget(self.cmb_online_year)
        o_box.addWidget(self.btn_fetch_online)
        o_box.addStretch()
        l_form.addLayout(o_box, 4, 1, 1, 5)
        
        layout.addWidget(form_frame)
        
        # --- C. NetCDF Options (Hidden by default) ---
        self.ncf_frame = QFrame()
        self.ncf_frame.setVisible(False)
        self.ncf_frame.setStyleSheet("QFrame { background: #eff6ff; border: 1px dashed #bfdbfe; border-radius: 4px; }")
        l_ncf = QHBoxLayout(self.ncf_frame)
        l_ncf.setContentsMargins(8, 4, 8, 4)
        
        self.cmb_ncf_layer = QComboBox()
        self.cmb_ncf_time = QComboBox()
        # Connect signals: Use helper to ensure plot update after load
        self.cmb_ncf_layer.currentIndexChanged.connect(self._trigger_ncf_refresh)
        self.cmb_ncf_time.currentIndexChanged.connect(self._trigger_ncf_refresh)
        
        l_ncf.addWidget(QLabel("<b>NetCDF:</b>"))
        l_ncf.addWidget(QLabel("Layer:"))
        l_ncf.addWidget(self.cmb_ncf_layer, 1)
        l_ncf.addWidget(QLabel("TimeStep:"))
        l_ncf.addWidget(self.cmb_ncf_time, 1)
        
        layout.addWidget(self.ncf_frame)

        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_variable_section(self, parent_layout=None):
        group = QGroupBox("2. Variable Selection")
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setVerticalSpacing(4)
        
        self.cmb_pollutant = QComboBox()
        self.cmb_pollutant.setMaxVisibleItems(8)
        self.cmb_pollutant.view().setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.cmb_pollutant.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.cmb_pollutant.view().setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cmb_pollutant.currentIndexChanged.connect(self._pollutant_changed)
        
        pol_layout = QHBoxLayout()
        pol_layout.addWidget(self.cmb_pollutant, 3)
        layout.addRow("Pollutant:", pol_layout)
        
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)
        
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
        self.cmb_scc.setMinimumWidth(100)
        self.cmb_scc.setMaximumWidth(320)
        self.cmb_scc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cmb_scc.setMaxVisibleItems(8)
        # Enable smooth pixel-based scrolling for long lists
        self.cmb_scc.view().setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.cmb_scc.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.cmb_scc.view().setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cmb_scc.view().setMinimumWidth(450)
        self.cmb_scc.setEditable(True)
        self.cmb_scc.setInsertPolicy(QComboBox.NoInsert)
        self.cmb_scc.addItem("All SCC")
        self.cmb_scc.setEnabled(False)
        scc_layout.addWidget(self.cmb_scc, 1)
        
        self.btn_scc_multi = QPushButton("Search")
        self.btn_scc_multi.setFixedWidth(90)
        self.btn_scc_multi.setToolTip("Select multiple SCCs from a searchable list")
        self.btn_scc_multi.setEnabled(False)
        self.btn_scc_multi.clicked.connect(self._open_scc_multi)
        scc_layout.addWidget(self.btn_scc_multi)

        layout.addRow("SCC:", scc_layout)
        
        if parent_layout: parent_layout.addWidget(group)
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_filter_section(self, parent_layout=None):
        group = QGroupBox("Advanced Filtering")
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
        layout.addRow("Select Shapefile:", shp_row)
        
        self.cmb_filter_op = QComboBox()
        self.cmb_filter_op.addItems(['clipped', 'intersect', 'within', 'False'])
        self.cmb_filter_op.setCurrentText('False')
        layout.addRow("Spatial Operation:", self.cmb_filter_op)
        
        self.cmb_filter_col = QComboBox()
        self.cmb_filter_col.setMaxVisibleItems(8)
        self.cmb_filter_col.view().setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.cmb_filter_col.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.cmb_filter_col.setEditable(True)
        layout.addRow("Filter Column:", self.cmb_filter_col)
        
        self.txt_filter_val = QLineEdit()
        self.txt_filter_val.setPlaceholderText("Value1, Value2...")
        layout.addRow("Filter Values:", self.txt_filter_val)
        
        range_layout = QHBoxLayout()
        self.txt_range_min = QLineEdit()
        self.txt_range_min.setPlaceholderText("Min")
        self.txt_range_max = QLineEdit()
        self.txt_range_max.setPlaceholderText("Max")
        range_layout.addWidget(self.txt_range_min)
        range_layout.addWidget(QLabel("to"))
        range_layout.addWidget(self.txt_range_max)
        layout.addRow("Value Range:", range_layout)
        
        if parent_layout: parent_layout.addWidget(group)
        else: self.control_layout.addWidget(group)

    def _init_plot_settings_section(self, parent_layout=None):
        group = QGroupBox("3. View Options")
        layout = QFormLayout(group)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setVerticalSpacing(4)
        layout.setLabelAlignment(Qt.AlignRight)
        
        # Consolidate Bins & CMap into one row
        bins_cmap_layout = QHBoxLayout()
        bins_cmap_layout.setSpacing(6)
        
        self.txt_bins = QLineEdit()
        self.txt_bins.setPlaceholderText("e.g. 0, 1, 10, 100")
        bins_cmap_layout.addWidget(self.txt_bins, 1)
        
        bins_cmap_layout.addWidget(QLabel("CMap:"))
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.setFixedWidth(120)
        self.cmb_cmap.setMaxVisibleItems(8)
        self.cmb_cmap.view().setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.cmb_cmap.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.cmb_cmap.view().setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.cmb_cmap.addItems(sorted([m for m in plt.colormaps() if not m.endswith('_r')]))
        self.cmb_cmap.setCurrentText('jet')
        bins_cmap_layout.addWidget(self.cmb_cmap)
        
        layout.addRow("Bins:", bins_cmap_layout)
        
        ov_row = QHBoxLayout()
        ov_row.setSpacing(4)
        self.txt_overlay_shp = QLineEdit()
        self.txt_overlay_shp.setPlaceholderText("Roads, Cities, etc...")
        self.txt_overlay_shp.editingFinished.connect(self._on_overlay_path_edit)
        ov_row.addWidget(self.txt_overlay_shp, 1)
        self.btn_browse_overlay = QPushButton("Browse")
        self.btn_browse_overlay.clicked.connect(self.browse_overlay_shpfile)
        ov_row.addWidget(self.btn_browse_overlay, 0)
        layout.addRow("Overlay:", ov_row)
        
        # Limits
        lim_layout = QHBoxLayout()
        tip = "Force a specific color-bar range for the map. For NetCDF files, this overrides the automatic global scaling (e.g. for animations)."
        
        self.txt_rmin = QLineEdit(); self.txt_rmin.setPlaceholderText("Min")
        self.txt_rmin.setToolTip(tip)
        self.txt_rmax = QLineEdit(); self.txt_rmax.setPlaceholderText("Max")
        self.txt_rmax.setToolTip(tip)
        
        lim_layout.addWidget(self.txt_rmin)
        l_to = QLabel("to"); l_to.setToolTip(tip)
        lim_layout.addWidget(l_to)
        lim_layout.addWidget(self.txt_rmax)
        
        l_fixed = QLabel("Fixed Scale:")
        l_fixed.setToolTip(tip)
        layout.addRow(l_fixed, lim_layout)
        
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
        
        self.chk_rev_cmap = QCheckBox("Reverse CMap")
        self.chk_rev_cmap.setToolTip("Invert the colormap colors")
        
        check_layout.addWidget(self.chk_log, 0, 0)
        check_layout.addWidget(self.chk_zoom, 0, 1)
        check_layout.addWidget(self.chk_nan0, 1, 0)
        check_layout.addWidget(self.chk_rev_cmap, 1, 1)
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
            self._stop_progress()
            if not is_headless:
                QMessageBox.critical(self, "Error", message)
        elif level == "WARNING":
            logging.warning(message)
            self.status_label.setText(f"Warning: {message}")
            if "failed" in message.lower() or "error" in message.lower():
                self._stop_progress()
            
            # --- Smart Error Interception ---
            if not is_headless:
                if "Counties not loaded" in message or "No suitable geometry" in message:
                    reply = QMessageBox.question(
                        self, "Missing Geometry", 
                        "Plotting within counties requires a county shapefile.\n\nWould you like to fetch the default US Counties map (approx 5MB) now?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.use_online_counties()
                elif "Grid not loaded" in message:
                    reply = QMessageBox.question(
                        self, "Missing Grid Definition", 
                        "Plotting grid data requires a GRIDDESC file and Grid Name.\n\nWould you like to select a GRIDDESC file now?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.select_griddesc_file()
                elif "Missing Join Column" in message:
                    QMessageBox.warning(self, "Warning", message)
            # --------------------------------

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
                # Fallback re-enable if worker finished
                if "finished" in message:
                    sys.stdout.write(">>> GUI_FLOW: _handle_notification re-enabling button <<<\n")
                    sys.stdout.flush()
                    self.btn_main_plot.setEnabled(True)
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
            if hasattr(self.cli_args, 'debug') and getattr(self.cli_args, 'debug', False):
                logging.debug(f"Loading emissions from: {self.inputfile_path}")
            self.load_input_file(self.inputfile_path)
            
        if self.griddesc_path:
            self.load_griddesc(self.griddesc_path)
            
        if self.counties_path:
             self._load_shapes()
        else:
             # Auto-load online counties if none specified (ensures we always have a base map)
             # We use a short delay to ensure UI components (like cmb_online_year) are fully ready
             QTimer.singleShot(500, self.use_online_counties)
        
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
        # Data-loading filters (applied during read_inputfile)
        f_col = cli.get('filter_col')
        if f_col:
            if hasattr(self, 'cmb_filter_col'):
                self.cmb_filter_col.setCurrentText(str(f_col))
        
        f_val = cli.get('filter_val') or cli.get('filter_values')
        if f_val:
            if hasattr(self, 'txt_filter_val'):
                if isinstance(f_val, list):
                    self.txt_filter_val.setText(", ".join(map(str, f_val)))
                else:
                    self.txt_filter_val.setText(str(f_val))
        
        f_start = cli.get('filter_start')
        if f_start and hasattr(self, 'txt_range_min'):
            self.txt_range_min.setText(str(f_start))
        
        f_end = cli.get('filter_end')
        if f_end and hasattr(self, 'txt_range_max'):
            self.txt_range_max.setText(str(f_end))

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
        cmap_val = cli.get('cmap') or cli.get('colormap') or 'jet'
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
             default_checked = False
             if isinstance(nan_val, bool): default_checked = nan_val
             elif str(nan_val).lower() in ('true', 'yes'): default_checked = True
             elif str(nan_val).lower() in ('false', 'no'): default_checked = False
             else:
                 try:
                     if abs(float(nan_val)) < 1e-9: default_checked = True
                 except: pass
             self.chk_nan0.setChecked(default_checked)
             
             if hasattr(self, 'txt_nan_val'): self.txt_nan_val.setText(str(nan_val))
        # Scale overrides
        low = cli.get('fixed_min') or cli.get('fixed_range_min') or cli.get('vmin')
        high = cli.get('fixed_max') or cli.get('fixed_range_max') or cli.get('vmax')
        if low is not None: self.txt_rmin.setText(str(low))
        if high is not None: self.txt_rmax.setText(str(high))

        # 5. NetCDF Settings
        zdim = cli.get('ncf-zdim') if 'ncf-zdim' in cli else cli.get('ncf_zdim')
        if zdim is not None: self._last_cfg_zdim = str(zdim)
        tdim = cli.get('ncf-tdim') if 'ncf-tdim' in cli else cli.get('ncf_tdim')
        if tdim is not None: self._last_cfg_tdim = str(tdim)

    def _batch_dim(self, ui_text: str) -> str:
        """Convert UI labels like 'Avg All' to batch keywords like 'avg', and 'LAY 0' to index '0'."""
        if not ui_text: return "0"
        t = ui_text.lower()
        if "avg" in t: return "avg"
        if "sum" in t: return "sum"
        if "max" in t: return "max"
        if "min" in t: return "min"
        # Consistency: Convert "LAY X" or "TSTEP X" to 0-based integer index
        if "lay" in t or "step" in t:
            try:
                return str(int(t.split()[-1]))
            except: pass
        return ui_text 

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
        if hasattr(self, '_loading_active') and self._loading_active:
             logging.warning("Load already in progress. Ignoring redundant call.")
             return
        self._loading_active = True
        
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        self.input_files_list = file_paths
        
        # Update Input entry
        if len(file_paths) == 1:
            self.txt_input.setText(file_paths[0])
        else:
            self.txt_input.setText("; ".join(file_paths))
            
        # Capture UI State (MUST be done on Main Thread)
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
        
        # NetCDF Check
        is_nc = is_netcdf_file(file_paths[0])
        ncf_params = self._get_ncf_params() if is_nc else {}

        # Capture Filter Args (Priority: 1. Current UI State, 2. Initial Config)
        f_col = self.cmb_filter_col.currentText() if hasattr(self, 'cmb_filter_col') else ""
        if f_col in ("", "Select Column"):
             f_col = getattr(self, '_json_arguments', {}).get('filter_col')
        
        f_val_txt = self.txt_filter_val.text() if hasattr(self, 'txt_filter_val') else ""
        if f_val_txt:
            # Parse possible comma-separated values from the UI text box
            f_val = [x.strip() for x in f_val_txt.split(",") if x.strip()]
        else:
            f_val = getattr(self, '_json_arguments', {}).get('filter_val') or getattr(self, '_json_arguments', {}).get('filter_values')

        f_start = self.txt_range_min.text() if hasattr(self, 'txt_range_min') else ""
        if not f_start: f_start = getattr(self, '_json_arguments', {}).get('filter_start')
        
        f_end = self.txt_range_max.text() if hasattr(self, 'txt_range_max') else ""
        if not f_end: f_end = getattr(self, '_json_arguments', {}).get('filter_end')
        
        # Start Worker Thread
        self._start_progress("Loading data...")
        
        # Prepare background task
        worker = Worker(self._task_load_input, file_paths, delim, skip, comment, ncf_params, f_col, f_start, f_end, f_val)
        worker.signals.result.connect(self._on_input_load_finished)
        worker.signals.error.connect(self._on_worker_error)
        self.threadpool.start(worker)

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
            
            # Load in background using QThreadPool
            worker = Worker(self._task_load_counties, f)
            worker.signals.result.connect(self._on_counties_loaded)
            worker.signals.error.connect(self._on_worker_error)
            self.threadpool.start(worker)

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
            # Trigger reload via worker
            self._start_progress(f"Loading shapefile...")
            worker = Worker(self._task_load_counties, path)
            worker.signals.result.connect(self._on_counties_loaded)
            worker.signals.error.connect(self._on_worker_error)
            self.threadpool.start(worker)

    @Slot()
    def _on_griddesc_path_edit(self):
        path = self.txt_griddesc.text().strip()
        if not path: return
        if path != getattr(self, 'griddesc_path', None):
            self.load_griddesc(path)

    @Slot()
    def use_online_counties(self):
        year = self.cmb_online_year.currentText()
        if not year: year = "2020"
        url = f"https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_county_500k.zip"
        self.notify_signal.emit("INFO", f"Fetching online counties for {year}...")
        self.counties_path = url
        self.txt_counties.setText(url)
        
        self._start_progress(f"Downloading shapefile...")
        worker = Worker(self._task_load_counties, url)
        worker.signals.result.connect(self._on_counties_loaded)
        worker.signals.error.connect(self._on_worker_error)
        self.threadpool.start(worker)

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
            self.stop_progress_signal.emit()

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
            self.stop_progress_signal.emit()

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
        elif 'lay' in lay_txt:
            try:
                params['layer_idx'] = int(lay_txt.split()[-1])
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
                params['tstep_idx'] = int(ts_txt.split()[-1])
                params['tstep_op'] = 'select'
            except:
                params['tstep_idx'] = 0; params['tstep_op'] = 'select'
        else:
            params['tstep_idx'] = 0; params['tstep_op'] = 'select'
        
        return params

    def _task_load_input(self, paths, delim, skip, comment, ncf_params, f_col, f_start, f_end, f_val, progress_callback):
        """Worker task: Load input file(s) and calculate stats."""
        # This runs in a background thread. NO UI ACCESS ALLOWED.
        try:
            fpath = paths if len(paths) > 1 else paths[0]
            
            # --- NetCDF Check ---
            is_nc = False
            first_file = paths[0]
            if is_netcdf_file(first_file):
                is_nc = True
                
            # Helper for notifications (Signal emission is thread-safe in PySide6)
            def worker_notify(lvl, msg):
                self.notify_signal.emit(lvl, msg)

            grid_gdf = None
            df = None
            raw = None
            
            if isinstance(fpath, list) and len(fpath) > 1:
                 worker_notify("INFO", f"Sequential load of {len(fpath)} files initiated...")
                 from data_processing import _normalize_input_result
                 dfs = []
                 raw_dfs = []
                 for p in fpath:
                     try:
                         d, r = read_inputfile(
                             p, delim=delim, skiprows=skip, comment=comment,
                             notify=worker_notify, lazy=True, return_raw=True,
                             workers=1, ncf_params=ncf_params,
                             flter_col=f_col, flter_start=f_start, flter_end=f_end, flter_val=f_val
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
                     ncf_params=ncf_params,
                     flter_col=f_col, flter_start=f_start, flter_end=f_end, flter_val=f_val
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
                except Exception: pass

                # Re-calculate lost stats for UI
                try:
                    summaries = []
                    file_list = fpath if isinstance(fpath, list) else [fpath]
                    
                    if isinstance(df, pd.DataFrame):
                        pols = detect_pollutants(df)
                        # Filter for columns that actually exist in df (NetCDF lazy loading has attributes for cols that aren't loaded yet)
                        pols = [p for p in pols if p in df.columns]
                        if pols:
                            sums = df[pols].sum()
                            maxs = df[pols].max()
                            means = df[pols].mean()
                            counts = df[pols].count()
                            
                            p_stats = {}
                            for p in pols:
                                try:
                                    p_stats[p] = {
                                        'sum': float(sums[p]), 'max': float(maxs[p]),
                                        'mean': float(means[p]), 'count': int(counts[p])
                                    }
                                except: pass
                            summaries.append({'source': file_list[0], 'pollutants': p_stats})
                        else:
                            summaries.append({'source': file_list[0], 'pollutants': {}})
                    
                    df.attrs['per_file_summaries'] = summaries
                except Exception as e:
                    logging.warning(f"Stat re-calc failed: {e}")

                # Pollutants
                cols = detect_pollutants(df)
                
                # Auto-Grid for NetCDF
                if is_nc:
                    try:
                        worker_notify("INFO", "Generating grid from NetCDF attributes...")
                        grid_gdf = create_ncf_domain_gdf(first_file if isinstance(first_file, str) else first_file[0])
                    except Exception as ge:
                        worker_notify("WARNING", f"Auto-grid from NetCDF failed: {ge}")

                return {
                    'df': df,
                    'raw': raw,
                    'is_nc': is_nc,
                    'pols': cols,
                    'grid_gdf': grid_gdf
                }
            return None

        except Exception as e:
            traceback.print_exc()
            raise e

    @Slot(object)
    def _on_input_load_finished(self, res):
        """Main thread: Update dataframes and UI."""
        self._loading_active = False
        if res is None: 
             self.stop_progress_signal.emit()
             return
        
        self.emissions_df = res['df']
        self.raw_df = res['raw']
        
        # Merge-in NetCDF Grid if generated
        if res.get('grid_gdf') is not None:
             self.grid_gdf = res['grid_gdf']
             self.notify_signal.emit("INFO", "Grid geometry loaded from NetCDF.")

        # Re-verify is_nc from loaded attributes if needed
        is_nc = res['is_nc']
        if not is_nc and self.emissions_df is not None:
             attrs = getattr(self.emissions_df, 'attrs', {})
             if attrs.get('is_netcdf') or attrs.get('format') == 'netcdf':
                 is_nc = True

        self.data_loaded_signal.emit(res['pols'], is_nc)
        self.notify_signal.emit("INFO", "Data load complete.")
        self.stop_progress_signal.emit()

    @Slot(tuple)
    def _on_worker_error(self, err):
        """Generic error handler for workers."""
        self._loading_active = False
        self.stop_progress_signal.emit()
        self.notify_signal.emit("ERROR", f"Task failed: {err[1]}")

    def _task_load_counties(self, path, progress_callback):
        """Worker task: Read shapefile."""
        return read_shpfile(path, True)

    @Slot(object)
    def _on_counties_loaded(self, gdf):
        """Main thread: Update counties GDF."""
        if gdf is not None:
             self.counties_gdf = gdf
             self._invalidate_merge_cache()
             self.notify_signal.emit("INFO", f"Loaded counties: {len(gdf)} features")
        else:
             self.notify_signal.emit("WARNING", "Counties file load returned no data.")
        
        self.stop_progress_signal.emit()

    def _invalidate_merge_cache(self):
        """Clear the cached merged dataframes."""
        self._merged_cache = {}

    def _start_progress(self, msg="Processing...", pol=None):
        """Show progress bar and status message."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate
        
        if pol:
            self.status_label.setText(f"Plotting {pol}...")
            self.btn_main_plot.setText("PLOTTING...")
        else:
            self.status_label.setText(msg)
            if "Loading data" in msg:
                self.btn_load_data.setText("LOADING...")
        
        # Disable all major action buttons to prevent conflicting tasks
        for btn in [self.btn_main_plot, self.btn_load_data, self.btn_preview, 
                    self.btn_metadata, self.btn_browse_gd, self.btn_browse_counties, 
                    self.btn_fetch_online, self.btn_export, self.btn_browse_overlay,
                    self.btn_scc_multi]:
            if hasattr(self, btn.objectName()) or btn: # Basic check
                 try: btn.setEnabled(False)
                 except: pass

    @Slot()
    def _stop_progress(self):
        """Helper to re-enable UI after plot or load."""
        QApplication.restoreOverrideCursor()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        
        # Re-enable buttons and reset text
        self.btn_main_plot.setEnabled(True)
        self.btn_main_plot.setText("GENERATE PLOT")
        self.btn_load_data.setEnabled(True)
        self.btn_load_data.setText("Load Data")
        
        for btn in [self.btn_preview, self.btn_metadata, self.btn_browse_gd, 
                    self.btn_browse_counties, self.btn_fetch_online, 
                    self.btn_export, self.btn_browse_overlay, self.btn_scc_multi]:
             if btn: btn.setEnabled(True)
             
        self._set_status_busy(False)

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

    @Slot(list, bool)
    def _post_load_update(self, pollutants, is_ncf):
        """Update UI after background loading finishes."""
        if self.emissions_df is None: return

        # Capture current selections to prevent reset during UI refresh
        cur_pol = self.cmb_pollutant.currentText()
        cur_scc = self.cmb_scc.currentText()

        self._invalidate_merge_cache()
        self._ff10_grid_ready = False
        self._base_view = None # Reset zoom for new data

        try:
            source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type')
        except Exception:
            source_type = None
        self._ff10_ready = bool(source_type == 'ff10_point')

        if self._ff10_ready and self.grid_gdf is not None:
            self._ensure_ff10_grid_mapping()

        # (SCC population moved further down to avoid redundant execution and overwriting)


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
        
        # Selection Priority: 1. Current user session, 2. CLI/Config preference, 3. First available
        if cur_pol in self.pollutants:
            self.cmb_pollutant.setCurrentText(cur_pol)
        elif self.preselected_pollutant in self.pollutants:
            self.cmb_pollutant.setCurrentText(self.preselected_pollutant)
        elif self.pollutants:
            self.cmb_pollutant.setCurrentIndex(0)
        self.cmb_pollutant.blockSignals(False)
        self._full_pollutant_list = None 
        
        # Trigger manual update to sync units label
        self._pollutant_changed()

        # Update Grid Info in status bar
        grid_desc = "--"
        if hasattr(self, 'emissions_df') and hasattr(self.emissions_df, 'attrs'):
            info = self.emissions_df.attrs.get('_smk_grid_info', {})
            name = info.get('grid_name')
            if name:
                grid_desc = f"{name} ({info.get('ncols')}x{info.get('nrows')})"
            elif self.grid_gdf is not None:
                grid_desc = f"Custom ({len(self.grid_gdf)} cells)"
        if hasattr(self, 'lbl_grid_info'):
            self.lbl_grid_info.setText(f"Grid: {grid_desc}")

        # Success status
        self._set_status_busy(False, f"Loaded {len(self.emissions_df):,} rows, {len(self.pollutants)} pollutants.")
        
        self.ncf_frame.setVisible(is_ncf)
        self.plot_controls_frame.setVisible(is_ncf)
        if is_ncf:
            # Populate NCF dimensions if available
            fname = self.input_files_list[0]
            try:
                dims = get_ncf_dims(fname)
                
                # Capture current selections to prevent reset
                cur_lay = self.cmb_ncf_layer.currentText()
                cur_ts = self.cmb_ncf_time.currentText()

                # Layers
                n_lay = dims.get('n_layers', 1)
                lay_items = ['Sum All', 'Avg All'] + [f"LAY {i}" for i in range(n_lay)]
                self.cmb_ncf_layer.blockSignals(True)
                self.cmb_ncf_layer.clear()
                self.cmb_ncf_layer.addItems(lay_items)
                
                # Selection priority: 1. Session persistence, 2. Config/CLI settings, 3. Default (LAY 0)
                if cur_lay in lay_items:
                    self.cmb_ncf_layer.setCurrentText(cur_lay)
                elif hasattr(self, '_last_cfg_zdim'):
                    cfg_z = str(self._last_cfg_zdim).lower()
                    for item in lay_items:
                        # Prioritize strict integer mapping for 0-based consistency
                        if cfg_z.isdigit() and item == f"LAY {cfg_z}":
                            self.cmb_ncf_layer.setCurrentText(item)
                            break
                        elif not cfg_z.isdigit() and cfg_z in item.lower():
                            self.cmb_ncf_layer.setCurrentText(item)
                            break
                elif 'LAY 0' in lay_items:
                     self.cmb_ncf_layer.setCurrentText('LAY 0')
                self.cmb_ncf_layer.blockSignals(False)
                
                # Time
                n_ts = dims.get('n_tsteps', 1)
                ts_items = ['Avg All', 'Sum All', 'Max'] + [f"TSTEP {i}" for i in range(n_ts)]
                self.cmb_ncf_time.blockSignals(True)
                self.cmb_ncf_time.clear()
                self.cmb_ncf_time.addItems(ts_items)
                
                # Selection priority: 1. Session persistence, 2. Config/CLI settings, 3. Default (TSTEP 0)
                if cur_ts in ts_items:
                    self.cmb_ncf_time.setCurrentText(cur_ts)
                elif hasattr(self, '_last_cfg_tdim'):
                    cfg_t = str(self._last_cfg_tdim).lower()
                    for item in ts_items:
                        # Prioritize strict integer mapping
                        if cfg_t.isdigit() and item == f"TSTEP {cfg_t}":
                            self.cmb_ncf_time.setCurrentText(item)
                            break
                        elif not cfg_t.isdigit() and cfg_t in item.lower():
                            self.cmb_ncf_time.setCurrentText(item)
                            break
                elif 'TSTEP 0' in ts_items:
                    self.cmb_ncf_time.setCurrentText('TSTEP 0')
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

        # Trigger initial filter column population - exclude pollutants
        if self.raw_df is not None:
            # Exclude known pollutants from the generic filter list to reduce confusion
            p_list = self.pollutants or []
            raw_cols = sorted([str(c) for c in self.raw_df.columns if c not in p_list])
            self.cmb_filter_col.clear()
            self.cmb_filter_col.addItems(raw_cols)

            # Restore filter column from configuration if requested (persistence)
            f_col_cfg = self._json_arguments.get('filter_col')
            if f_col_cfg:
                # Case-insensitive match against what we just loaded
                target = str(f_col_cfg).strip().lower()
                match = next((c for c in raw_cols if str(c).lower() == target), None)
                if match:
                    self.cmb_filter_col.setCurrentText(match)
            
            # SCC Detection & Mapping
            self._scc_display_to_code = {}
            try:
                raw_cols_list = list(self.raw_df.columns)
                raw_cols_lower = {c.lower(): c for c in raw_cols_list}
                scc_col = next((raw_cols_lower[c] for c in SCC_COLS if c in raw_cols_lower), None)
                desc_col = next((raw_cols_lower[c] for c in DESC_COLS if c in raw_cols_lower), None)
                
                print(f">>> SCC_DEBUG: Columns found: {raw_cols_list}")
                print(f">>> SCC_DEBUG: scc_col={scc_col}, desc_col={desc_col}")
                
                if scc_col:
                    if desc_col:
                        df_scc = self.raw_df[[scc_col, desc_col]].drop_duplicates().astype(str)
                        df_scc[scc_col] = df_scc[scc_col].str.strip()
                        df_scc[desc_col] = df_scc[desc_col].str.strip()
                        df_scc = df_scc[df_scc[scc_col].str.len() > 0]
                        df_scc['display'] = df_scc[scc_col] + " | " + df_scc[desc_col]
                    else:
                        unique_sccs = sorted(self.raw_df[scc_col].astype(str).str.strip().unique())
                        unique_sccs = [s for s in unique_sccs if s]
                        df_scc = pd.DataFrame({'display': unique_sccs, scc_col: unique_sccs})
                        
                    self._scc_display_to_code = dict(zip(df_scc['display'], df_scc[scc_col]))
                    print(f">>> SCC_DEBUG: Success. Found {len(self._scc_display_to_code)} SCC items.")
                elif desc_col:
                    unique_descs = sorted(self.raw_df[desc_col].astype(str).str.strip().unique())
                    unique_descs = [s for s in unique_descs if s]
                    self._scc_display_to_code = {s: s for s in unique_descs}
                    print(f">>> SCC_DEBUG: Success (Desc only). Found {len(self._scc_display_to_code)} items.")
                else:
                    print(f">>> SCC_DEBUG: No SCC or Description column found.")

                self.cmb_scc.blockSignals(True)
                self.cmb_scc.clear()
                self.cmb_scc.addItem("All SCC")
                if self._scc_display_to_code:
                    scc_items = sorted(self._scc_display_to_code.keys())
                    self.cmb_scc.addItems(scc_items)
                    
                    # Restore previous SCC selection or Multi state
                    if cur_scc in scc_items or cur_scc == "All SCC":
                        self.cmb_scc.setCurrentText(cur_scc)
                    elif self.selected_sccs:
                        self.cmb_scc.setCurrentText("(Multi)")
                        
                    self.cmb_scc.setEnabled(True)
                    self.btn_scc_multi.setEnabled(True)
                    print(f">>> SCC_DEBUG: SCC dropdown ENABLED.")
                else:
                    self.cmb_scc.setEnabled(False)
                    self.btn_scc_multi.setEnabled(False)
                    print(f">>> SCC_DEBUG: SCC dropdown DISABLED (no items).")
                self.cmb_scc.blockSignals(False)
                self._scc_full_list = None 
            except Exception as e:
                print(f">>> SCC_DEBUG: FAILED with error: {e}")
                logging.warning(f"SCC mapping failed: {e}")
                self.cmb_scc.setEnabled(False)
                self.btn_scc_multi.setEnabled(False)

        # Set plot type from config (validation happens in _merged() during plot)
        # Following gui_qt.py pattern: just read the config, don't try to validate/correct
        requested_pltyp = self._json_arguments.get('pltyp') or 'auto'
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

        # Auto-select and plot if configured - ONLY if no user session exists
        if not cur_pol or cur_pol == "":
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
        else:
            # If a refresh was triggered by NCF dim change, update the current plot
            if self._ncf_refresh_active:
                logging.info("NCF dimension change detected. Updating plot...")
                QTimer.singleShot(300, self.update_plot)
        
        self._ncf_refresh_active = False

    def _load_shapes(self):
        """Load county shapefiles."""
        threading.Thread(target=self._shape_worker, daemon=True).start()
        
    def _shape_worker(self):
        try:
             gdf = read_shpfile(self.counties_path, True)
             if gdf is not None:
                 self.counties_gdf = gdf
                 self._invalidate_merge_cache()
                 self.notify_signal.emit("INFO", f"Loaded counties: {len(gdf)} features")
                 # Trigger redraw if county mode is active
                 if self.cmb_pltyp.currentText().lower() == 'county' and self.emissions_df is not None:
                      QTimer.singleShot(800, self.update_plot)
        except Exception as e:
            self.notify_signal.emit("WARNING", f"Shapefile load error: {e}")

    def _clear_anim_cache(self):
        """Reset the animation data cache and index."""
        self._t_data_cache = None
        self._t_idx = 0
        self._is_showing_agg = False

    def _trigger_ncf_refresh(self):
        """Trigger a background reload and visual refresh when NCF dimensions change."""
        self._clear_anim_cache()
        self._invalidate_merge_cache()  # Crucial to force re-fetch of pollutants
        self._ncf_refresh_active = True
        self.load_input_file(self.input_files_list)

    def _pollutant_changed(self):
        # Update units label if metadata available
        pol = self.cmb_pollutant.currentText()
        unit = self.units_map.get(pol, "-")
        
        # Fallback to variable_metadata if units_map doesn't have it (Common for NetCDF)
        if (unit == "-" or not unit) and hasattr(self, 'emissions_df') and self.emissions_df is not None:
             vmeta = getattr(self.emissions_df, 'attrs', {}).get('variable_metadata', {})
             if isinstance(vmeta, dict) and pol in vmeta:
                  unit = vmeta[pol].get('units', "-")
        
        self.lbl_units.setText(f"Unit: {unit}")
        # Invalidate animation cache if pollutant changed
        self._clear_anim_cache()
        
        # Keep plot_controls_frame visible if NCF data is loaded
        # Trigger background load of time series to get global min/max for scaling
        if self.ncf_frame.isVisible():
             QTimer.singleShot(100, self._ensure_time_data)

    def _ensure_time_data(self):
        if self._t_data_cache is not None: return True
        pol = self.cmb_pollutant.currentText()
        if not pol or self.emissions_df is None: return False
        
        try:
            from ncf_processing import get_ncf_animation_data
            # Use _merged_gdf (the plotted GDF) to ensure row/col order matches _update_view
            df = self._merged_gdf if self._merged_gdf is not None and 'ROW' in getattr(self._merged_gdf, 'columns', []) else self.emissions_df
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
        # Capturing these BEFORE _ensure_time_data possibly resets them
        is_fresh = self._t_data_cache is None
        is_agg = self._is_showing_agg or (self._t_idx < 0)
        
        if not self._ensure_time_data(): return
        cache = self._t_data_cache
        n_steps = len(cache['times'])
        
        if n_steps <= 1:
            self.notify_signal.emit("INFO", "Animation: Only 1 time step available.")
            return

        # If we were showing an aggregate, 'Next' starts at 0, 'Prev' at end.
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
                
            # Fallback for Native Shortcut (No Geometry)
            if sub.empty and '_smk_grid_info' in gdf.attrs:
                try:
                    info = gdf.attrs['_smk_grid_info']
                    # Get view bounds in Grid CRS
                    x0, x1 = xlim
                    y0, y1 = ylim
                    
                    # Resolve CRS transform
                    plot_crs = getattr(ax, '_smk_ax_crs', None)
                    native_crs_str = info.get('proj_str')
                    
                    if plot_crs and native_crs_str:
                        native_crs = pyproj.CRS.from_user_input(native_crs_str)
                        # Check if transform needed (comparing CRS objects)
                        if plot_crs != native_crs:
                            tf = pyproj.Transformer.from_crs(plot_crs, native_crs, always_xy=True)
                            # Transform corners to cover extent
                            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                            tx, ty = [], []
                            for cx, cy in corners:
                                tx_pt, ty_pt = tf.transform(cx, cy)
                                tx.append(tx_pt); ty.append(ty_pt)
                            x0, x1 = min(tx), max(tx)
                            y0, y1 = min(ty), max(ty)

                    # Map to Col/Row indices (1-based assumption for SMOKE standard)
                    # COL = floor((X - XORIG) / XCELL) + 1
                    c0 = int(np.floor((x0 - info['xorig']) / info['xcell'])) + 1
                    c1 = int(np.floor((x1 - info['xorig']) / info['xcell'])) + 1
                    r0 = int(np.floor((y0 - info['yorig']) / info['ycell'])) + 1
                    r1 = int(np.floor((y1 - info['yorig']) / info['ycell'])) + 1
                    
                    # Filter by Index
                    sub = gdf[
                        (gdf['COL'] >= c0) & (gdf['COL'] <= c1) &
                        (gdf['ROW'] >= r0) & (gdf['ROW'] <= r1)
                    ]
                except Exception as ex:
                    logging.warning(f"Fallback grid filtering failed: {ex}")

            if sub.empty:
                 self.notify_signal.emit("WARNING", "No cells in current view or grid definition missing.")
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
                      
                      # Set vmin/vmax on norm for ALL normalization types
                      if hasattr(coll.norm, 'vmin'):
                          coll.norm.vmin = c_vmin
                      if hasattr(coll.norm, 'vmax'):
                          coll.norm.vmax = c_vmax
                      coll.set_clim(c_vmin, c_vmax)

                  # 2. Set Data
                  final_vals = new_vals
                  
                  # --- County Aggregation for Animation ---
                  is_county_plot = False
                  try:
                      is_county_plot = self.cmb_pltyp.currentText().lower() == 'county'
                  except: pass
                  
                  if is_county_plot and hasattr(self, '_grid_to_county_map') and self._grid_to_county_map is not None:
                      try:
                          # We have grid-level values (new_vals) corresponding to self.emissions_df rows(? not guaranteed)
                          # Actually, get_ncf_animation_data returns values for rows/cols passed to it.
                          # _ensure_time_data passed rows/cols from self.emissions_df.
                          # So new_vals is aligned with self.emissions_df rows.
                          
                          # We need to map self.emissions_df indices to FIPS, then sum by FIPS.
                          # self._grid_to_county_map links (ROW, COL) -> FIPS.
                          # Efficient approach:
                          # merge new_vals with map.
                          
                          # But new_vals is a numpy array.
                          # Create a temp dataframe.
                          # This is slightly slow per frame but robust.
                          
                          # Check if emissions_df has ROW/COL
                          if 'ROW' in self.emissions_df.columns and 'COL' in self.emissions_df.columns:
                              tmp = pd.DataFrame({
                                  'ROW': self.emissions_df['ROW'],
                                  'COL': self.emissions_df['COL'],
                                  'val': new_vals
                              })
                              # Merge with map
                              merged_tmp = tmp.merge(self._grid_to_county_map, on=['ROW', 'COL'], how='inner')
                              k_col = self._grid_to_county_key
                              
                              # Sum by FIPS
                              agg = merged_tmp.groupby(k_col)['val'].sum()
                              
                              # Use current plot order (self._merged_gdf)
                              if self._merged_gdf is not None:
                                   # The plot collection aligns with _merged_gdf
                                   # We need to map 'agg' to _merged_gdf[k_col]
                                   final_vals = self._merged_gdf[k_col].map(agg).fillna(0.0).values
                      except Exception as e:
                           logging.debug(f"County anim aggregation failed: {e}")
                           # fallback to raw, though it will likely throw size error
                  # ----------------------------------------

                  coll.set_array(final_vals)
                  ax._smk_current_vals = final_vals
                  
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
                  status_txt = str(time_lbl)
                  
                  if hasattr(self, 'lbl_anim_status') and self.lbl_anim_status:
                      self.lbl_anim_status.setText(status_txt)
                  
                  if hasattr(self, 'lbl_anim_time') and self.lbl_anim_time:
                      self.lbl_anim_time.setText(f"Time: {status_txt}")
                      
                  if "Sum" in status_txt or "Mean" in status_txt:
                      self.lbl_anim_status.setStyleSheet("color: #059669; font-weight: bold;")
                  else:
                      self.lbl_anim_status.setStyleSheet("color: #2563eb; font-weight: bold;")
                      if self._t_data_cache and not self._is_showing_agg:
                          self.lbl_anim_status.setText(f"{status_txt} ({self._t_idx+1}/{n_steps})")
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
            pol = getattr(ax, '_smk_pollutant', self.cmb_pollutant.currentText())
            if not pol:
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
                            
                            # Native implies the plot axes match the grid's native coordinates (usually LCC or WGS84)
                            is_native = proj_sel in ('auto', 'wgs84', 'native', 'epsg:4326', 'default', 'lcc') 
                            
                            if info and is_native and 'ROW' in gdf.columns:
                                try:
                                    col0 = int(np.floor((x0 - info['xorig']) / info['xcell']))
                                    col1 = int(np.ceil((x1 - info['xorig']) / info['xcell']))
                                    row0 = int(np.floor((y0 - info['yorig']) / info['ycell']))
                                    row1 = int(np.ceil((y1 - info['yorig']) / info['ycell']))
                                    
                                    mask = (gdf['ROW'] > row0) & (gdf['ROW'] <= row1+1) & \
                                           (gdf['COL'] > col0) & (gdf['COL'] <= col1+1)
                                    # Use .values for speed and to avoid index alignment issues
                                    m_vals = mask.values if hasattr(mask, 'values') else np.array(mask)
                                    filtered = vals[m_vals] if m_vals.size == vals.size else vals
                                except Exception:
                                    filtered = vals
                            else:
                                # B. Precise Spatial Slicing
                                try:
                                    sidx = getattr(gdf, 'sindex', None)
                                    if sidx and hasattr(gdf, 'geometry') and gdf.geometry is not None:
                                        # Query spatial index using view bounds
                                        valid_idxs = list(sidx.intersection(view_box.bounds))
                                        if valid_idxs:
                                            # PERFORMANCE OPTIMIZATION:
                                            # If we have many candidates (> 5000), skip the precise geometry check
                                            # and assume the index (bbox) intersection is "good enough" for title stats.
                                            if len(valid_idxs) > 5000:
                                                final_ilocs = np.array(valid_idxs)
                                            else:
                                                # For smaller sets, do the precise check
                                                sub_gdf = gdf.iloc[valid_idxs]
                                                precise_mask = sub_gdf.geometry.intersects(view_box)
                                                if hasattr(precise_mask, 'values'):
                                                    precise_mask = precise_mask.values
                                                final_ilocs = np.array(valid_idxs)[precise_mask]
                                            
                                            # Safety filter for values alignment
                                            final_ilocs = final_ilocs[(final_ilocs >= 0) & (final_ilocs < vals.size)]
                                            if final_ilocs.size > 0:
                                                filtered = vals[final_ilocs]
                                            else:
                                                filtered = np.array([])
                                        else:
                                            filtered = np.array([])
                                    elif hasattr(gdf, 'geometry') and vals.size < 2000:
                                        # Brute force only for small datasets to avoid freezing
                                        mask = gdf.geometry.intersects(view_box)
                                        m_vals = mask.values if hasattr(mask, 'values') else np.array(mask)
                                        filtered = vals[m_vals]
                                    else:
                                        # Fallback/Safety
                                        filtered = vals
                                except Exception as e:
                                    logging.debug(f"Title stats slice failed: {e}")
                                    filtered = vals
                # Calculate Stats
                filtered_arr = np.asanyarray(filtered)
                clean = filtered_arr[~np.isnan(filtered_arr)] if filtered_arr.size > 0 else np.array([])
                
                # 4. Construct Multi-line Title (Mirroring gui_qt.py style)
                title_lines = []
                
                # Line 1: Pollutant (Unit)
                # gui_qt.py: f"{pollutant}{u_str}"
                l1 = f"{pol}"
                if unit: l1 += f" ({unit})"
                title_lines.append(l1)
                
                # Line 2: Time (if applicable)
                if time_lbl:
                    title_lines.append(f"Time: {time_lbl}")
                elif self.cmb_ncf_time.isVisible():
                     title_lines.append(f"Time: {self.cmb_ncf_time.currentText()}")
                
                # Line 3: Stats & Side Panel Update
                if clean.size > 0:
                    mn, mx = np.nanmin(clean), np.nanmax(clean)
                    u, s = np.nanmean(clean), np.nansum(clean)
                    
                    def _f(v): return f"{v:.4g}"
                        
                    stats_str = f"Min: {_f(mn)}  Mean: {_f(u)}  Max: {_f(mx)}  Sum: {_f(s)}"
                    if unit: stats_str += f" {unit}"
                    title_lines.append(stats_str)
                    
                    # Update side panel
                    try:
                        if hasattr(self, 'lbl_stats_sum'):
                            self.lbl_stats_sum.setText(f"{s:.4g}")
                            self.lbl_stats_max.setText(f"{mx:.4g}")
                            self.lbl_stats_mean.setText(f"{u:.4g}")
                            self.lbl_stats_count.setText(str(len(clean)))
                    except Exception: pass
                    
                    self.statusBar().showMessage(f"View Stats | {stats_str}", 3000)
                else:
                    title_lines.append("No Data in View")
                    # Clear side panel if empty
                    try:
                        if hasattr(self, 'lbl_stats_sum'):
                            self.lbl_stats_sum.setText("-")
                            self.lbl_stats_max.setText("-")
                            self.lbl_stats_mean.setText("-")
                            self.lbl_stats_count.setText("0")
                    except Exception: pass
                
                ax.set_title("\n".join(title_lines), fontsize=12, pad=10)
                
                # Remove legacy floating stats text box if it exists
                if hasattr(ax, '_smk_stats_text'):
                    try:
                        ax._smk_stats_text.remove()
                        del ax._smk_stats_text
                    except: pass
                
                # Force redraw to show updated title
                ax.figure.canvas.draw_idle()
                
            except Exception as e:
                logging.debug(f"Title update warning: {e}")
                
                ax.figure.canvas.draw_idle()
            except Exception as e:
                logging.debug(f"Stats extraction failed: {e}")
                ax.set_title(f"{pol} Emissions", fontsize=12)
        except Exception as e:
            logging.debug(f"Global title update failed: {e}")

    @Slot()
    def _on_main_plot_clicked(self):
        """Internal handler for the main plot button click."""
        sys.stdout.write("\n>>> GUI_EVENT: [GENERATE PLOT] Button Clicked! <<<\n")
        sys.stdout.flush()
        logging.warning("GUI_DEBUG: [_on_main_plot_clicked] Entering...")
        self.on_generate_clicked()

    def on_generate_clicked(self):
        """Dedicated slot for the Generate Plot button."""
        sys.stdout.write(">>> GUI_FLOW: Entering on_generate_clicked <<<\n")
        sys.stdout.flush()
        logging.warning("GUI_DEBUG: [on_generate_clicked] Entering method...")
        self.update_plot()

    # --- Updated Plot Logic ---
    def update_plot(self):
        """Main Plotting Logic, mirroring gui_qt.py exactly."""
        try:
            sys.stdout.write(f">>> GUI_FLOW: update_plot for '{self.cmb_pollutant.currentText()}' <<<\n")
            sys.stdout.flush()
            logging.warning(f"GUI_DEBUG: [update_plot] Started for '{self.cmb_pollutant.currentText()}'")
            
            if self.emissions_df is None:
                print("DEBUG: [update_plot] emissions_df is None. Checking for auto-load paths...")
                logging.info("DEBUG: [update_plot] emissions_df is None.")
                if hasattr(self, 'input_files_list') and self.input_files_list:
                    logging.info("DEBUG: [update_plot] Attempting auto-load...")
                    self.load_input_file(self.input_files_list)
                    return
                self.notify_signal.emit('WARNING', 'Load smkreport and shapefile first.')
                return

            pol = self.cmb_pollutant.currentText()
            if not pol and hasattr(self, 'pollutants') and self.pollutants:
                 pol = self.pollutants[0]
                 print(f"DEBUG: [update_plot] Auto-selecting first pollutant: {pol}")
                 self.cmb_pollutant.setCurrentText(pol)
                 # Re-fetch pol after UI update
                 pol = self.cmb_pollutant.currentText()
            
            if not pol:
                print("DEBUG: [update_plot] ABORT: No pollutant selected.")
                logging.info("DEBUG: [update_plot] No pollutant selected.")
                self.notify_signal.emit('WARNING', 'No pollutant selected.')
                return

            print(f"DEBUG: [update_plot] Proceeding with pollutant: {pol}")
            logging.info(f"DEBUG: [update_plot] Processing pollutant: {pol}")

            # Capture UI state (Mirroring gui_qt.py variables)
            plot_by_mode = self.cmb_pltyp.currentText().lower()
            scc_selection = self.cmb_scc.currentText()
            scc_code_map = self._scc_display_to_code.copy() if self._scc_display_to_code else {}
            
            # Determine plotting CRS
            plot_crs_info = self._plot_crs()

            # NEW: Capture Plot Meta on Main Thread (Thread Safety)
            def safe_float(txt):
                if not txt: return None
                try: return float(txt)
                except: return None

            is_rev = self.chk_rev_cmap.isChecked()
            cmap_name = self.cmb_cmap.currentText()
            if is_rev: cmap_name += '_r'
            
            # Robust Unit resolution for meta
            unit = self.units_map.get(pol, "")
            if not unit and hasattr(self, 'emissions_df') and self.emissions_df is not None:
                 vmeta = getattr(self.emissions_df, 'attrs', {}).get('variable_metadata', {})
                 if isinstance(vmeta, dict) and pol in vmeta:
                      unit = vmeta[pol].get('units', "")

            meta_fixed = {
                'cmap': cmap_name,
                'use_log': self.chk_log.isChecked(),
                'bins_txt': self.txt_bins.text(),
                'bins': self._parse_bins(),
                'unit': unit,
                'vmin': safe_float(self.txt_rmin.text()),
                'vmax': safe_float(self.txt_rmax.text()),
                'show_graticule': True,
                'zoom_to_data': self.chk_zoom.isChecked(),
                'fill_nan': self.chk_nan0.isChecked()
            }

            # Thread-Safety: Capture filter UI state on the main thread
            filter_state = {
                'filter_col': self.cmb_filter_col.currentText() if hasattr(self, 'cmb_filter_col') else '',
                'filter_val': self.txt_filter_val.text() if hasattr(self, 'txt_filter_val') else '',
                'range_min': self.txt_range_min.text() if hasattr(self, 'txt_range_min') else '',
                'range_max': self.txt_range_max.text() if hasattr(self, 'txt_range_max') else '',
                'filter_op': self.cmb_filter_op.currentText() if hasattr(self, 'cmb_filter_op') else 'False',
            }
            
            logging.warning(f"GUI_DEBUG: [update_plot] Starting background thread for {pol}...")
            print(f"GUI_DEBUG: [update_plot] Starting background thread for {pol}...", flush=True)
            self._start_progress(pol=pol)
            
            threading.Thread(
                target=self._plot_worker, 
                args=(pol, plot_by_mode, scc_selection, scc_code_map, plot_crs_info, meta_fixed, filter_state),
                daemon=True
            ).start()
            
        except Exception as e:
            sys.stdout.write(f">>> GUI_ERROR: update_plot failed: {e} <<<\n")
            sys.stdout.flush()
            logging.error(f"DEBUG: [update_plot] FAILED on main thread: {e}")
            logging.error(traceback.format_exc())
            self._stop_progress()
            self.notify_signal.emit("ERROR", f"Plot Update: {e}")

    def _merged(self, plot_by_mode=None, scc_selection=None, scc_code_map=None, notify=None, pollutant=None, fill_nan=False, filter_state=None) -> Optional[gpd.GeoDataFrame]:
        """Prepare and merge emissions with geometry, mirroring gui_qt.py logic exactly."""
        def _do_notify(level, title, msg, exc=None, **kwargs):
            if notify: notify(level, title, msg, exc, **kwargs)
            else: self.notify_signal.emit(level, f"{title}: {msg}")

        # Thread-Safety: Use pre-captured filter_state if provided (from main thread)
        if filter_state is None:
            filter_state = {}

        # Determine target pollutant early
        target_pol = pollutant if pollutant else ''
        logging.warning(f"GUI_DEBUG: [_merged] Start. Pollutant: {target_pol}, FillNaN: {fill_nan}")
        
        if self.emissions_df is None:
            logging.warning("GUI_DEBUG: [_merged] emissions_df is NONE. Aborting.")
            return None

        # Determine if we already have geometry (e.g. Inline Point Source processed or FF10 mapped)
        has_geometry = isinstance(self.emissions_df, gpd.GeoDataFrame) and self.emissions_df.geometry is not None
        
        # Robust Native/Gridded Check: Attributes are fragile, so check columns too
        is_native = getattr(self.emissions_df, 'attrs', {}).get('_smk_is_native', False)
        if not is_native and hasattr(self.emissions_df, 'columns'):
             is_native = ('ROW' in self.emissions_df.columns and 'COL' in self.emissions_df.columns)
             
        mode = (plot_by_mode if plot_by_mode is not None else self.cmb_pltyp.currentText().lower())
        
        logging.warning(f"GUI_DEBUG: [_merged] State: has_geometry={has_geometry}, is_native={is_native}, mode={mode}, len={len(self.emissions_df)}")

        # Local storage for lazy-fetched data in this specific merge call
        # Using a local variable instead of self._lazy_fetched_col for better thread safety
        current_lazy_col = None

        # LAZY FETCH: Check if we need to fetch the pollutant into the main DF before proceeding
        if target_pol and hasattr(self.emissions_df, 'columns') and target_pol not in self.emissions_df.columns:
            ds = getattr(self.emissions_df, 'attrs', {}).get('_smk_xr_ds')
            if ds is not None:
                try:
                    _do_notify('INFO', 'Fetching Data', f"Lazy-extracting {target_pol} from NetCDF dataset...")
                    from ncf_processing import read_ncf_emissions
                    ncf_params = self.emissions_df.attrs.get('ncf_params', {})
                    path = self.input_files_list[0] if getattr(self, 'input_files_list', None) else None
                    if path:
                        new_data = read_ncf_emissions(path, pollutants=[target_pol], xr_ds=ds, **ncf_params)
                        if target_pol in new_data.columns:
                            # Thread-Safety: Do NOT mutate self.emissions_df from background thread.
                            # Store column locally for application on the copy below.
                            current_lazy_col = (target_pol, new_data[target_pol].values)
                            if target_pol not in self.units_map:
                                v_meta = new_data.attrs.get('variable_metadata', {}).get(target_pol, {})
                                self.units_map[target_pol] = v_meta.get('units', '')
                except Exception as e:
                    _do_notify('WARNING', 'Fetch Failed', f"Could not lazy-load {target_pol}: {e}")

        # Shortcut for Native Grids or Already-Geometrized Data
        # We always take the shortcut for native gridded data to avoid losing lazy-loaded pollutants
        # during the complex aggregation/merge logic intended for vector data.
        can_shortcut = has_geometry or is_native
        
        if can_shortcut and mode in ['auto', 'grid']:
            # Determine which columns to keep (IDs + active pollutant)
            id_cols = {'ROW', 'COL', 'FIPS', 'region_cd', 'GRID_RC', 'SCC', 'geometry'}
            # Case-insensitive and whitespace-stripped match
            t_clean = target_pol.strip().upper() if target_pol else None
            cols_to_keep = [c for c in self.emissions_df.columns if c in id_cols or c.upper() in id_cols or (t_clean and c.strip().upper() == t_clean)]
            
            # Create subset copy
            res_df = self.emissions_df[cols_to_keep].copy()

            # Apply lazy column if it was fetched
            if current_lazy_col and current_lazy_col[0] == target_pol and target_pol not in res_df.columns:
                res_df[target_pol] = current_lazy_col[1]

            # Ensure ROW and COL are integers for QuadMesh optimization
            for c_idx in ['ROW', 'COL']:
                if c_idx in res_df.columns:
                    # Fill NaNs with -1 (invalid index) and explicitly cast to int64
                    res_df[c_idx] = res_df[c_idx].fillna(-1).astype('int64')

            if fill_nan and target_pol and target_pol in res_df.columns:
                res_df[target_pol] = res_df[target_pol].fillna(0).astype('float32')
            elif target_pol and target_pol in res_df.columns:
                # Ensure float32 for memory efficiency and matplotlib compatibility
                if pd.api.types.is_numeric_dtype(res_df[target_pol]):
                    res_df[target_pol] = res_df[target_pol].astype('float32')
            
            # Ensure GRID_RC exists
            if 'GRID_RC' not in res_df.columns and 'ROW' in res_df.columns and 'COL' in res_df.columns:
                try: res_df['GRID_RC'] = res_df['ROW'].astype(str) + '_' + res_df['COL'].astype(str)
                except: pass
            
            # Return if already a GeoDataFrame
            if isinstance(res_df, gpd.GeoDataFrame):
                return res_df
            
            # Convert to GDF with robust geometry init
            if 'geometry' not in res_df.columns:
                res_df['geometry'] = [None] * len(res_df)
            
            res_gdf = gpd.GeoDataFrame(res_df, geometry='geometry')
            res_gdf.attrs = self.emissions_df.attrs.copy()
            res_gdf.attrs['_smk_is_native'] = True
           
            # Inject grid info if missing
            if '_smk_grid_info' not in res_gdf.attrs and self.grid_gdf is not None:
                 if hasattr(self.grid_gdf, 'attrs') and '_smk_grid_info' in self.grid_gdf.attrs:
                      res_gdf.attrs['_smk_grid_info'] = self.grid_gdf.attrs['_smk_grid_info']
            
            if getattr(res_gdf, 'crs', None) is None:
                try:
                    info = res_gdf.attrs.get('_smk_grid_info')
                    if info and info.get('proj_str'):
                        res_gdf.set_crs(info['proj_str'], inplace=True)
                except: pass
            
            return res_gdf

        base_gdf: Optional[gpd.GeoDataFrame] = None
        merge_on: Optional[str] = None
        geometry_tag = None
        
        try:
            source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type') if isinstance(self.emissions_df, pd.DataFrame) else None
        except Exception:
            source_type = None

        sel_display = scc_selection if scc_selection is not None else ''
        sel_code = ''  # Empty string for consistent falsy behavior
        code_map = scc_code_map if scc_code_map is not None else self._scc_display_to_code
        
        if self.selected_sccs and scc_selection is None:
            # Multi-mode active
            codes = []
            for it in self.selected_sccs:
                 c = code_map.get(it)
                 if c: codes.append(c)
            sel_code = codes if codes else None
        else:
            # Single-mode or override
            if code_map:
                try:
                    sel_code = code_map.get(sel_display, '') or ''
                except Exception:
                    sel_code = ''
        
        has_scc_cols = any(c.lower() in SCC_COLS for c in self.emissions_df.columns)
        use_scc_filter = bool(has_scc_cols and sel_code)

        if mode == 'grid':
            if self.grid_gdf is None:
                _do_notify('WARNING', 'Grid not loaded', 'Select a GRIDDESC and Grid Name first to build the grid geometry.')
                raise ValueError("Handled")
            
            # For NetCDF, we don't need to ensure FF10 mapping, but we might need to ensure GRID_RC exists
            if not is_native:
                self._ensure_ff10_grid_mapping(notify_success=False)
            else:
                # GRID_RC will be created on the local copy (emis_for_merge) below
                pass

            base_gdf = self.grid_gdf
            merge_on = 'GRID_RC'
            geometry_tag = 'grid'
        elif mode == 'county':
            if self.counties_gdf is None:
                if self.grid_gdf is not None:
                    _do_notify('INFO', 'Geo Fallback', 'Counties shapefile still loading. Falling back to Grid view for now.')
                    base_gdf = self.grid_gdf
                    merge_on = 'GRID_RC'
                    geometry_tag = 'grid'
                    if not is_native:
                        self._ensure_ff10_grid_mapping(notify_success=False)
                else:
                    _do_notify('WARNING', 'Counties not loaded', 'Load a counties shapefile or use the online counties option.')
                    raise ValueError("Handled")
            else:
                base_gdf = self.counties_gdf
                merge_on = 'FIPS'
                geometry_tag = 'county'
        else: # auto
            if self.grid_gdf is not None:
                if 'GRID_RC' not in getattr(self.emissions_df, 'columns', []):
                    if not is_native:
                        self._ensure_ff10_grid_mapping(notify_success=False)
                    else:
                        # GRID_RC will be created on the local copy (emis_for_merge) below
                        pass
                             
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
                    elif 'ROW' in self.emissions_df.columns and 'COL' in self.emissions_df.columns:
                        # Fallback for grid-to-county mapping
                        merge_on = 'FIPS'
                geometry_tag = 'county'
            if base_gdf is None:
                _do_notify('WARNING', 'No suitable geometry', 'Could not find a suitable shapefile (Counties or Grid) for the loaded emissions data.')
                raise ValueError("Handled")

        if merge_on and isinstance(base_gdf, gpd.GeoDataFrame):
            if merge_on not in base_gdf.columns:
                # Try standard aliases for FIPS
                if merge_on == 'FIPS':
                    if 'GEOID' in base_gdf.columns:
                        base_gdf = base_gdf.copy()
                        base_gdf['FIPS'] = base_gdf['GEOID']
                    elif 'region_cd' in base_gdf.columns:
                        base_gdf = base_gdf.copy()
                        base_gdf['FIPS'] = base_gdf['region_cd']
                    else:
                         _do_notify('WARNING', 'Geometry Missing FIPS', f"Selected geometry layer lacks 'FIPS', 'GEOID', or 'region_cd' column.")
                         raise ValueError("Handled")
                else:
                    _do_notify('WARNING', 'Geometry Missing Column', f"Selected geometry layer lacks '{merge_on}' column.")
                    raise ValueError("Handled")

        pol_tuple = tuple(self.pollutants or [])
        # target_pol determined above

        # Thread-Safety: Read filter state from pre-captured dict (not from widgets)
        f_col = filter_state.get('filter_col', '')
        f_vals_str = filter_state.get('filter_val', '')
        f_min = filter_state.get('range_min', '')
        f_max = filter_state.get('range_max', '')
        f_op = filter_state.get('filter_op', 'False')

        # Cache key including NaN fill and spatial filter state
        cache_key = (
            geometry_tag or mode,
            merge_on or '',
            id(base_gdf) if base_gdf is not None else 0,
            id(self.emissions_df) if isinstance(self.emissions_df, pd.DataFrame) else 0,
            tuple(self.emissions_df.columns) if hasattr(self.emissions_df, 'columns') else (),
            tuple(self.raw_df.columns) if hasattr(self.raw_df, 'columns') else (),
            tuple(self.selected_sccs) if self.selected_sccs else '',
            sel_code if use_scc_filter else '',
            pol_tuple,
            target_pol,
            fill_nan,
            f_op,
            id(self.filter_gdf) if getattr(self, 'filter_gdf', None) is not None else 0,
            f_col,
            f_vals_str,
            f_min,
            f_max
        )
        
        cached = self._merged_cache.get(cache_key)
        if cached is not None:
            return cached[0].copy()

        # Memory Optimization: Subset columns before heavy operations
        id_cols = {'ROW', 'COL', 'FIPS', 'region_cd', 'GRID_RC', 'SCC', 'geometry', f_col}
        t_clean = target_pol.strip().upper() if target_pol else None
        cols_to_keep = [c for c in self.emissions_df.columns if c in id_cols or c.upper() in id_cols or (t_clean and c.strip().upper() == t_clean)]
        
        if isinstance(self.emissions_df, pd.DataFrame):
            # Subset before copying to save time/memory
            emis_for_merge = self.emissions_df[cols_to_keep].copy()
            # Copy attrs manually
            if hasattr(self.emissions_df, 'attrs'):
                emis_for_merge.attrs = self.emissions_df.attrs.copy()
        else:
            emis_for_merge = None

        if emis_for_merge is None:
            return None

        # Apply lazy-fetched column to local copy if available
        if current_lazy_col and current_lazy_col[0] == target_pol and target_pol not in emis_for_merge.columns:
            emis_for_merge[target_pol] = current_lazy_col[1]

        # Thread-Safety: Ensure GRID_RC on local copy if needed
        if merge_on == 'GRID_RC' and 'GRID_RC' not in emis_for_merge.columns:
            if 'ROW' in emis_for_merge.columns and 'COL' in emis_for_merge.columns:
                try:
                    emis_for_merge['GRID_RC'] = emis_for_merge['ROW'].astype(str) + '_' + emis_for_merge['COL'].astype(str)
                except Exception:
                    pass

        # --- Advanced Attribute Filtering ---
        # Filter settings already read from filter_state above
        
        def _apply_attr_filters(df):
            if not isinstance(df, pd.DataFrame) or not f_col or f_col not in df.columns:
                return df
            df_res = df
            if f_vals_str:
                # Support comma or space separated list of discrete values
                f_vals = [v.strip() for v in f_vals_str.replace(',', ' ').split() if v.strip()]
                if f_vals:
                    df_res = filter_dataframe_by_values(df_res, f_col, f_vals)
            if f_min or f_max:
                # Apply numerical range filter
                df_res = filter_dataframe_by_range(df_res, f_col, f_min or None, f_max or None)
            return df_res

        emis_for_merge = _apply_attr_filters(emis_for_merge)

        # Lazy Fetch already handled at top of function
        
        # --- Pre-Merge: Grid-to-County Mapping (NetCDF) ---
        if geometry_tag == 'county' and merge_on == 'FIPS':
            # Check if we have FIPS. If not, but we have GRID (ROW/COL), try to map it.
            has_fips = 'FIPS' in emis_for_merge.columns or 'region_cd' in emis_for_merge.columns
            has_grid = 'ROW' in emis_for_merge.columns and 'COL' in emis_for_merge.columns
            
            if not has_fips and has_grid:
                _do_notify("INFO", "Mapping Grid to County", "Calculating county aggregate from grid cells...")
                try:
                    emis_for_merge = self._augment_with_county_mapping(emis_for_merge)
                    # Check success
                    if 'FIPS' not in emis_for_merge.columns and 'region_cd' not in emis_for_merge.columns:
                        _do_notify("WARNING", "Mapping Failed", "Could not map grid cells to counties. Ensure county shapefile covers the grid domain.")
                except Exception as e:
                    _do_notify("ERROR", "Mapping Error", f"Failed to map grid to county: {e}")

        # --- FF10/County Join Helper ---
        if geometry_tag == 'county' and merge_on == 'FIPS':
            # Robust FIPS normalization: US County shapefiles expect 5-digit GEOID (state+county).
            # Emissions data often carries a 6-digit prefix (0 for US, 1 for CA, etc.) from FF10/Reports.
            m_col = 'FIPS' if 'FIPS' in emis_for_merge.columns else next((c for c in emis_for_merge.columns if c.lower() == 'region_cd'), None)
            
            if m_col:
                s_reg = emis_for_merge[m_col].astype(str).str.strip().str.zfill(5)
                # Handle SMKREPORT's 12-digit FIPS format (extract last 6 digits)
                # Also ensure 6-digit format for consistent merging
                def normalize_fips(x):
                    x_str = str(x).strip()
                    # If 12+ digits, extract last 6 (SMKREPORT format)
                    if len(x_str) > 6:
                        return x_str[-6:]
                    # Otherwise, pad to 6 digits
                    else:
                        return x_str.zfill(6)
                
                emis_for_merge['FIPS'] = s_reg.apply(normalize_fips)
            
            # Normalize geometry FIPS to 6-digit format to match emissions data
            if base_gdf is not None and 'FIPS' in base_gdf.columns:
                 base_gdf = base_gdf.copy()
                 s_geom = base_gdf['FIPS'].astype(str).str.strip()
                 # Convert to 6-digit format: handle 12-digit (SMKREPORT), extract last 6,
                 # or pad 5-digit standard US FIPS with leading 0
                 def normalize_geom_fips(x):
                     x_str = str(x).strip()
                     # If 12+ digits, extract last 6 (rare, but handle edge case)
                     if len(x_str) > 6:
                         return x_str[-6:]
                     # If 5 digits, prepend '0' for 6-digit format
                     elif len(x_str) == 5:
                         return '0' + x_str
                     # Otherwise use as-is or pad to 6
                     else:
                         return x_str.zfill(6)
                 base_gdf['FIPS'] = s_geom.apply(normalize_geom_fips)

        # SCC Filtering & Re-aggregation
        # BUGFIX: Force aggregation for county/grid modes to match Batch aggregation behavior.
        # Without this, FF10 point-level data is merged with county geometries without aggregation,
        # causing higher statistics in GUI than Batch mode.
        # NOTE: Only apply aggregation for FF10 data; SMKREPORT is already aggregated at county level
        source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type', '')
        is_ff10 = source_type in ('ff10_point', 'ff10_nonpoint')
        
        if merge_on not in emis_for_merge.columns or use_scc_filter or (geometry_tag in ('county', 'grid') and is_ff10):
            # Prefer emis_for_merge if it already has merge_on column (e.g., FF10 with FIPS)
            # Only use raw_df if merge_on is missing or SCC filtering is needed
            if merge_on in emis_for_merge.columns and not use_scc_filter:
                # Use processed data that already has merge_on column
                data_to_agg = emis_for_merge
            else:
                # Fall back to raw_df for missing columns or SCC filtering
                data_to_agg = self.raw_df
            
            if use_scc_filter and data_to_agg is not None:
                scc_col = next((c for c in data_to_agg.columns if c.lower() in SCC_COLS), None)
                if scc_col:
                    if isinstance(sel_code, list):
                        data_to_agg = data_to_agg[data_to_agg[scc_col].astype(str).str.strip().isin(sel_code)].copy()
                    else:
                        data_to_agg = data_to_agg[data_to_agg[scc_col].astype(str).str.strip() == sel_code].copy()
            
            # Apply generic attribute filters
            if data_to_agg is not None:
                data_to_agg = _apply_attr_filters(data_to_agg)
            
            if isinstance(data_to_agg, pd.DataFrame) and merge_on in data_to_agg.columns:
                pols = list(self.pollutants or detect_pollutants(data_to_agg))
                if pols:
                    try:
                        subset = data_to_agg[[merge_on] + pols]
                        # Match Batch mode aggregation parameters exactly (batch.py line 946)
                        agg = subset.groupby(merge_on, dropna=False, sort=False, observed=False).sum(numeric_only=True).reset_index()
                        agg.attrs = dict(getattr(emis_for_merge, 'attrs', {}))
                        emis_for_merge = agg
                    except Exception as e:
                        logging.warning(f"Re-aggregation failed: {e}")

        if merge_on not in emis_for_merge.columns:
            _do_notify('WARNING', 'Missing Join Column', f"Data lacks '{merge_on}' required for {geometry_tag} plot.")
            raise ValueError("Handled")

        # DEBUG: Check FIPS match before merge
        if merge_on == 'FIPS' and 'FIPS' in emis_for_merge.columns and 'FIPS' in base_gdf.columns:
            emis_fips = set(emis_for_merge['FIPS'].dropna().unique().astype(str))
            geom_fips = set(base_gdf['FIPS'].dropna().unique().astype(str))
            matched = emis_fips & geom_fips
            emis_only = emis_fips - geom_fips
            geom_only = geom_fips - emis_fips
            logging.warning(f"DEBUG: FIPS Match Before Merge - Emis rows with matching FIPS: {len(emis_for_merge[emis_for_merge['FIPS'].isin(matched)])}/{len(emis_for_merge)}")
            logging.warning(f"DEBUG: FIPS in emis but not geom: {len(emis_only)} codes (sample: {list(emis_only)[:5]})")
            logging.warning(f"DEBUG: FIPS in geom but not emis: {len(geom_only)} codes")

        try:
            from data_processing import merge_emissions_with_geometry
            merged, prepared = merge_emissions_with_geometry(
                emis_for_merge, base_gdf, merge_on, sort=False, copy_geometry=False
            )

            # Post-Merge Cleanup: Consolidate suffixed columns (ROW_x/y -> ROW)
            # This ensures downstream consumers can rely on standard names.
            for base in ['ROW', 'COL', 'FIPS', 'region_cd']:
                x_col, y_col = f"{base}_x", f"{base}_y"
                if x_col in merged.columns and base not in merged.columns:
                    merged[base] = merged[x_col]
                elif y_col in merged.columns and base not in merged.columns:
                     merged[base] = merged[y_col]
                
                # Drop suffixes if we successfully mapped to base or if base already existed
                if base in merged.columns:
                    merged.drop(columns=[c for c in [x_col, y_col] if c in merged.columns], inplace=True, errors='ignore')
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

            # Spatial Filtering (use pre-captured filter_state, not widget)
            filter_mode = f_op
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

            # Mass Balance Check
            try:
                if prepared is not None and not prepared.empty and merged is not None:
                    # Find a numeric column to check (pollutant)
                    check_cols = [c for c in prepared.columns if c in merged.columns and pd.api.types.is_numeric_dtype(prepared[c])]
                    target = target_pol if target_pol in check_cols else (check_cols[0] if check_cols else None)
                    
                    if target:
                        total_mass = prepared[target].sum()
                        plotted_mass = merged[target].sum()
                        if total_mass > 0:
                            diff = total_mass - plotted_mass
                            pct = (diff / total_mass) * 100
                            if pct > 0.01: # > 0.01% discrepancy
                                _do_notify("WARNING", "Mass Mismatch", 
                                    f"Plot shows {plotted_mass:.2e} {target}, but computed total is {total_mass:.2e} {target}.\n"
                                    f"Approx. {pct:.1f}% ({diff:.2e}) of emissions fell outside the selected geometry (e.g. outside county boundaries) and are not plotted."
                                )
            except Exception: pass
            
            import gc; gc.collect()
            return merged
        except Exception as e:
            _do_notify('ERROR', 'Merge Failed', f"{e}")
            return None

    def _plot_worker(self, pollutant, plot_by_mode, scc_selection, scc_code_map, plot_crs_info, meta_fixed, filter_state=None):
        """Worker thread to prepare data and reproject, mirroring gui_qt.py exactly."""
        try:
            logging.warning(f"GUI_DEBUG: [PlotWorker] Starting for {pollutant}")
            print(f"GUI_DEBUG: [PlotWorker] Thread started for {pollutant}", flush=True)
            
            def safe_notify(level, title, msg, exc=None, **kwargs):
                try:
                    self.notify_signal.emit(level, f"{title}: {msg}")
                except RuntimeError:
                    pass

            try:
                logging.info(f"DEBUG: [PlotWorker] Calling _merged...")
                merged = self._merged(
                    plot_by_mode=plot_by_mode, 
                    scc_selection=scc_selection, 
                    scc_code_map=scc_code_map,
                    notify=safe_notify,
                    pollutant=pollutant,
                    fill_nan=meta_fixed.get('fill_nan', False),
                    filter_state=filter_state
                )
                if merged is not None:
                    print(f"DEBUG: [PlotWorker] _merged success. Rows: {len(merged)}")
                else:
                    print("DEBUG: [PlotWorker] _merged returned NONE.")
                logging.info(f"DEBUG: [PlotWorker] _merged returned. Type: {type(merged)}, Rows: {len(merged) if merged is not None else 0}")
            except ValueError as ve:
                print(f"DEBUG: [PlotWorker] VALUE ERROR: {ve}")
                logging.error(f"DEBUG: [PlotWorker] _merged raised ValueError: {ve}")
                if str(ve) == "Handled":
                     self.stop_progress_signal.emit()
                     return
                raise ve
            except Exception as e:
                print(f"DEBUG: [PlotWorker] ERROR in _merged: {e}")
                logging.error(f"DEBUG: [PlotWorker] _merged raised Exception: {e}")
                raise e

            if merged is None:
                logging.warning("DEBUG: [PlotWorker] Merged GDF is None. Aborting.")
                try:
                    self.notify_signal.emit('WARNING', 'Missing Data. Load smkreport and shapefile first.')
                except RuntimeError:
                    pass
                self.stop_progress_signal.emit()
                return

            # REPROJECTION LOGIC
            plot_crs, tf_fwd, tf_inv = plot_crs_info
            logging.info(f"DEBUG: [PlotWorker] Target CRS: {plot_crs}")
            
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
                          # IMPORTANT: still set the CRS property so the plotting tools know the target projection
                          if getattr(merged_plot, 'crs', None) != plot_crs:
                               try: merged_plot.set_crs(plot_crs, allow_override=True, inplace=True)
                               except: 
                                    try: merged_plot.crs = plot_crs
                                    except: pass
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
            meta['plot_crs'] = plot_crs # Pass for reference in UI thread
            
            # Calculate Zoom Bounds if requested
            if meta.get('zoom_to_data') and pollutant in merged_plot.columns:
                try:
                    # Filter for non-zero, non-null emissions to zoom into actual data
                    non_zero = merged_plot[merged_plot[pollutant].notna() & (merged_plot[pollutant] > 0)]
                    if not non_zero.empty:
                        meta['zoom_gdf'] = non_zero
                    else:
                        meta['zoom_gdf'] = merged_plot
                except Exception:
                    meta['zoom_gdf'] = merged_plot
            else:
                meta['zoom_gdf'] = merged_plot

            try:
                logging.info(f"DEBUG: [PlotWorker] Emitting plot_ready_signal for {pollutant}")
                self.plot_ready_signal.emit(merged_plot, pollutant, meta)
                logging.info(f"DEBUG: [PlotWorker] Signal emitted.")
            except RuntimeError:
                logging.error("DEBUG: [PlotWorker] RuntimeError emitting signal.")
            
        except Exception as e:
            logging.error(f"DEBUG: [PlotWorker] FAILED: {e}")
            logging.error(traceback.format_exc())
            try:
                self.notify_signal.emit("ERROR", f"Plot Preparation Error: {e}")
            except RuntimeError:
                pass
        finally:
            logging.info("DEBUG: [PlotWorker] Finished. Stopping progress.")
            self.stop_progress_signal.emit()

    @Slot(object, str, dict)
    def _render_plot_on_main(self, gdf, column, meta):
        """Main thread slot to update the matplotlib figure."""
        if getattr(self, '_smk_rendering', False): 
            return
        self._smk_rendering = True
        try:
            # Memory Safety: Detach hover handlers from all axes before clearing
            # This prevents 'zombie' closures from trying to access GC-locked geometries.
            for ax in self.figure.axes:
                ax.format_coord = lambda x, y: ""
                if hasattr(ax, '_smk_hover_gdf'): ax._smk_hover_gdf = None

            if gdf is None or gdf.empty:
                logging.warning("GDF is empty. Nothing to plot.")
                self.figure.clear()
                self.canvas.draw()
                self._stop_progress()
                return

            logging.info(f"DEBUG: [_render_plot_on_main] Entered. Pollutant: {column}, GDF Rows: {len(gdf)}")
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
                     
                 # Cursor Plot Dropdown
                 self.pc_layout.addSpacing(10)
                 self.pc_layout.addWidget(QLabel("| Cursor Plot:"))
                 self.cmb_cursor_mode = QComboBox()
                 self.cmb_cursor_mode.addItems(["by-TSTEP", "by-LAY"])
                 
                 # Dynamic Enablage Logic for Cursor Plot
                 has_multi_time = (self.cmb_ncf_time.count() > 4) if hasattr(self, 'cmb_ncf_time') else False
                 has_multi_lay = (self.cmb_ncf_layer.count() > 3) if hasattr(self, 'cmb_ncf_layer') else False
                 is_inline = (src_type == 'inline_point_lazy') or ('stack_groups_path' in attrs)
                 
                 if not has_multi_time:
                     self.cmb_cursor_mode.model().item(0).setEnabled(False)
                 if not has_multi_lay or is_inline:
                     self.cmb_cursor_mode.model().item(1).setEnabled(False)
                     
                 # Auto-select available option
                 if not has_multi_time and (has_multi_lay and not is_inline):
                     self.cmb_cursor_mode.setCurrentIndex(1)
                 
                 self.pc_layout.addWidget(self.cmb_cursor_mode)
                     
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
            
            # 3. Resolve actual CRS (needed for transfers and overlays)
            actual_crs = getattr(gdf, 'crs', None) or meta.get('plot_crs')
            if actual_crs is None:
                 actual_crs = "EPSG:4326"
            
            logging.info(f"DEBUG: Rendering plot with Actual CRS: {actual_crs}")
            logging.info(f"DEBUG: Calling create_map_plot...")

            # 4. Get transformers (needed for hover & graticule)
            tf_fwd = meta.get('tf_fwd')
            tf_inv = meta.get('tf_inv')
            if tf_fwd is None or tf_inv is None:
                tf_fwd, tf_inv = self._get_transformers(actual_crs)

            # Aspect ratio is handled by Figure size and create_map_plot management
            
            # 6. Render main plot
            self._merged_gdf = gdf
            
            # Defensive Column Resolution: Try strict, then flexible, then recovery
            actual_col = None
            if column in gdf.columns:
                actual_col = column
            else:
                # Find case-insensitive match
                actual_col = next((c for c in gdf.columns if c.strip().upper() == column.strip().upper()), None)
            
            # --- EMERGENCY RECOVERY ---
            # If the column is missing (e.g. was optimized away or lazy-load failed to sync),
            # try to re-fetch it directly into the GDF if we have grid info.
            if actual_col is None and self.input_files_list:
                path = self.input_files_list[0]
                if path.endswith(('.nc', '.ncf', '.timind')):
                    try:
                        logging.warning(f"Pollutant '{column}' missing from GDF. Attempting emergency recovery fetch...")
                        from ncf_processing import read_ncf_emissions
                        # Extract just the single pollutant for this layer/time
                        lay_idx = self.cmb_ncf_layer.currentIndex() if hasattr(self, 'cmb_ncf_layer') else 0
                        t_idx = self.cmb_ncf_time.currentIndex() if hasattr(self, 'cmb_ncf_time') else 0
                        
                        # Fix for layer index if aggregate mode (Sum/Avg)
                        if lay_idx < 0: lay_idx = 0
                        
                        rec_df = read_ncf_emissions(path, pollutants=[column], layer_idx=lay_idx, time_idx=t_idx)
                        if column in rec_df.columns:
                            gdf[column] = rec_df[column].values
                            actual_col = column
                            logging.info(f"Emergency recovery successful for {column}")
                    except Exception as ex:
                        logging.error(f"Emergency recovery failed: {ex}")

            if actual_col is None:
                raise KeyError(f"Pollutant '{column}' not found in dataset. Available columns: {list(gdf.columns)}")

            ax._smk_current_vals = gdf[actual_col].values
            ax._smk_pollutant = actual_col 
            ax._smk_time_lbl = None
            
            p_lw = 0.05
            p_ec = 'black'
            if len(gdf) > 20000:
                p_lw = 0.0
                p_ec = 'none'

            unit = self.units_map.get(actual_col, "")
            if not unit:
                vmeta = getattr(self.emissions_df, 'attrs', {}).get('variable_metadata', {})
                if isinstance(vmeta, dict) and actual_col in vmeta:
                     unit = vmeta[actual_col].get('units', '')

            # Fixed Scaling for NetCDF or Config
            extra_plot_kwargs = {'disable_quadmesh': False}
            
            # 1. Capture user intent from UI (passed in via meta)
            if meta.get('vmin') is not None:
                extra_plot_kwargs['vmin'] = meta['vmin']
            if meta.get('vmax') is not None:
                extra_plot_kwargs['vmax'] = meta['vmax']

            # 2. Config override (Backup fallback)
            if 'vmin' not in extra_plot_kwargs and self._json_arguments.get('vmin') is not None:
                try: extra_plot_kwargs['vmin'] = float(self._json_arguments['vmin'])
                except: pass
            if 'vmax' not in extra_plot_kwargs and self._json_arguments.get('vmax') is not None:
                try: extra_plot_kwargs['vmax'] = float(self._json_arguments['vmax'])
                except: pass
                
            # 3. Animation Cache (Auto-calculated global limits for NetCDF)
            if self._t_data_cache:
                if 'vmin' not in extra_plot_kwargs:
                    extra_plot_kwargs['vmin'] = self._t_data_cache.get('vmin')
                if 'vmax' not in extra_plot_kwargs:
                    extra_plot_kwargs['vmax'] = self._t_data_cache.get('vmax')

            logging.info(f"DEBUG: [_render_plot_on_main] Starting create_map_plot for {actual_col}...")
            collection = create_map_plot(
                gdf=gdf,
                column=actual_col,
                title="", 
                ax=ax,
                cmap_name=cmap,
                bins=bins,
                log_scale=use_log,
                overlay_counties=None, # Handle manually in GUI for zorder control
                overlay_shape=None,
                unit_label=unit,
                crs_proj=actual_crs,
                tf_fwd=None, # Handle graticule manually after zoom
                tf_inv=None,
                zoom_to_data=False,
                linewidth=p_lw,
                edgecolor=p_ec,
                **extra_plot_kwargs
            )
            logging.info(f"DEBUG: [_render_plot_on_main] create_map_plot finished.")
            
            # --- Robust Overlay Support (GUI Managed) ---
            # Ensure the plot is at lower zorder
            if collection:
                 collection.set_zorder(0)
            
            def _gui_add_overlays(_ax, _counties, _shapes, _target_crs):
                from plotting import _resolve_cmap_and_theme
                _, theme = _resolve_cmap_and_theme(cmap)
                if _counties is not None:
                    try:
                        c_gdf = _counties.to_crs(_target_crs) if _target_crs and getattr(_counties, 'crs', None) else _counties
                        c_gdf.boundary.plot(ax=_ax, color=theme['county_color'], linewidth=theme['county_lw'], alpha=0.8, zorder=5)
                    except: pass
                if _shapes is not None:
                    s_list = _shapes if isinstance(_shapes, list) else [_shapes]
                    clrs = ['cyan', 'magenta', 'yellow', 'red', 'lime', 'orange']
                    for idx, s_gdf in enumerate(s_list):
                        try:
                            s_out = s_gdf.to_crs(_target_crs) if _target_crs and getattr(s_gdf, 'crs', None) else s_gdf
                            s_out.boundary.plot(ax=_ax, color=clrs[idx % len(clrs)], linewidth=theme['overlay_lw'], alpha=0.9, linestyle='--', zorder=6)
                        except: pass
            
            _gui_add_overlays(ax, self.counties_gdf, self.overlay_gdf, actual_crs)

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
                        # Fallback: maintain current view or calculate from data
                        minx, miny, maxx, maxy = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]

                    padx = (maxx - minx) * 0.05
                    pady = (maxy - miny) * 0.05
                    if abs(padx) < 1e-7: padx = 0.5
                    if abs(pady) < 1e-7: pady = 0.5
                    ax.set_xlim(min(minx, maxx) - abs(padx), max(minx, maxx) + abs(padx))
                    ax.set_ylim(min(miny, maxy) - abs(pady), max(miny, maxy) + abs(pady))
                except Exception as ze:
                    logging.warning(f"Zoom calculation failed: {ze}")
            else:
                # Restore previous view to prevent "zoom out" reset
                if self._base_view:
                    try:
                        ax.set_xlim(self._base_view[0])
                        ax.set_ylim(self._base_view[1])
                    except: pass

            # 5b. Render Graticule (Handle manually after zoom to ensure coverage)
            if show_graticule and tf_fwd and tf_inv:
                 self._draw_graticule(ax, tf_fwd, tf_inv)
            
            # Initial Draw to register axes
            self.canvas.draw_idle()

            # 6. Install Interactions (Hover & Box-Zoom)
            self._setup_hover(gdf, actual_col, ax, tf_inv=tf_inv)
            self._install_box_zoom(ax)
            
            # Step-wise draw to register axes and allow tick generation
            self.canvas.draw()

            # --- FIX: Prevent graticule labels from spilling outside frame ---
            try:
                gx0, gx1 = ax.get_xlim()
                gy0, gy1 = ax.get_ylim()
                # Use a smaller buffer (1%) to avoid hiding labels in global views
                gbuf_x = 0.01 * abs(gx1 - gx0)
                gbuf_y = 0.01 * abs(gy1 - gy0)
                for txt in ax.texts:
                    if '°' in txt.get_text():
                        txt.set_clip_on(True)
                        tpos = txt.get_position()
                        # Only hide if the label center is actually outside the view bounds
                        # or extremely close to the edge (buffer)
                        if txt.get_ha() == 'center':
                            if tpos[0] < gx0 or tpos[0] > gx1:
                                txt.set_visible(False)
                        if txt.get_va() == 'center':
                            if tpos[1] < gy0 or tpos[1] > gy1:
                                txt.set_visible(False)
            except Exception: pass
            
            # 7. Format Colorbar (Mirroring gui_qt.py)
            # Identify the colorbar axis
            cbar_ax = None
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

                # --- REPAIR COLORBAR CONTENT ---
                # Check if the axis is empty (missing the color mesh/patches)
                if len(cbar_ax.collections) == 0 and len(cbar_ax.patches) == 0:
                    try:
                        # Use the Figure's colorbar method to ensure it draws on correctly
                        cbar_repair = cbar_ax.figure.colorbar(
                            collection, cax=cbar_ax, 
                            orientation='vertical' if orient_vertical else 'horizontal'
                        )
                        collection.colorbar = cbar_repair
                    except Exception: pass

                # 7b. Label colorbar with units if available (Use local unit variable)
                try:
                    if unit:
                         if orient_vertical: cbar_ax.set_ylabel(unit, fontweight='bold', fontsize=9)
                         else: cbar_ax.set_xlabel(unit, fontweight='bold', fontsize=9)
                except Exception: pass
                    
                # 7c. Robust Colorbar Formatting (Fix for small values and custom bins)
                try:
                    from matplotlib.ticker import FixedLocator, FuncFormatter, ScalarFormatter, LogFormatter, FixedFormatter, AutoLocator
                    from matplotlib.colors import LogNorm, BoundaryNorm
                    is_log = isinstance(norm, LogNorm) if norm is not None else False
                    bins_ticks = (bins if bins is not None else [])

                    if bins_ticks:
                        is_boundary = isinstance(norm, BoundaryNorm) if norm else False
                        ticks = [t for t in bins_ticks if (t > 0) ] if is_log else list(bins_ticks)
                        if ticks:
                            if is_boundary:
                                # For BoundaryNorm, use the colorbar object to map ticks correctly
                                cbar = getattr(collection, 'colorbar', None)
                                if cbar:
                                    cbar.set_ticks(ticks)
                                    # Format labels to avoid scientific notation for standard ranges
                                    labels = []
                                    for v in ticks:
                                        # Use standard float format then strip trailing zeros
                                        s = f"{float(v):.10f}".rstrip('0').rstrip('.')
                                        # If it's zero or too small for .10f, use .10g
                                        if not s or s == "0": s = f"{v:.10g}"
                                        # If still scientific or too long, use human-friendly g
                                        if 'e' in s or len(s) > 12: s = f"{v:g}"
                                        labels.append(s)
                                    
                                    if orient_vertical:
                                        cbar_ax.yaxis.set_ticklabels(labels)
                                    else:
                                        cbar_ax.xaxis.set_ticklabels(labels)
                                    cbar.update_ticks()
                            else:
                                # Determine formatter that shows requested precision
                                def _fmt_precise(x, pos=None):
                                    if x == 0: return "0"
                                    return f"{x:.6g}"
                                
                                fmt = FuncFormatter(_fmt_precise)
                                
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
                            # Use a custom formatter to ensure consistent decimal formatting
                            # and avoid the 1e-02 vs 10^-2 math/scientific flipping.
                            def _fmt_log_readable(x, pos=None):
                                if x <= 0: return ""
                                # Prioritize plain decimal/integer formatting up to 10^10 to avoid scientific notation
                                if 1e-4 <= x <= 1e10:
                                    s = f"{x:f}".rstrip('0').rstrip('.')
                                    if s: return s
                                return f"{x:g}"
                            
                            fmt = FuncFormatter(_fmt_log_readable)
                            if orient_vertical: 
                                cbar_ax.yaxis.set_major_formatter(fmt)
                            else: 
                                cbar_ax.xaxis.set_major_formatter(fmt)
                        else:
                            # For linear scales, prioritize plain decimals for standard ranges
                            fmt = ScalarFormatter(useOffset=False)
                            fmt.set_scientific(False) # Prioritize plain formatting
                            
                            if orient_vertical: 
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
                base_xlim = ax.get_xlim()
                base_ylim = ax.get_ylim()
                
                # Only update the stored "Base View" if we are NOT in a zoomed-to-data mode.
                # This ensures that when Zoom to Data is toggled off, we can accurately 
                # restore the last known "Full" view rather than staying stuck in a zoomed state.
                if not zoom_to_data:
                    self._base_view = (base_xlim, base_ylim)
                
                # B. Home Button Override (Direct Logic)
                def _home_override(*args, **kwargs):
                    try:
                        ax.set_xlim(base_xlim)
                        ax.set_ylim(base_ylim)
                        self.canvas.draw_idle()
                        self._update_plot_title(ax, immediate=True)
                        
                        # Explicitly trigger graticule redraw if setup
                        gr_cb = getattr(ax, '_smk_graticule_callback', None)
                        if gr_cb:
                            gr_cb(ax)
                    except: pass
                
                # Override the instance method
                self.toolbar.home = _home_override
                
                # C. Refresh Navigation Toolbar (Preserve stack if possible)
                # IMPORTANT: In Qt, we must refresh the button states
                if hasattr(self.toolbar, 'update'):
                    self.toolbar.update()
                elif hasattr(self.toolbar, 'set_history_buttons'):
                    self.toolbar.set_history_buttons()

                # D. Axis Callbacks for Live Stats (Robust Update)
                if hasattr(ax, '_smk_stats_cids'):
                    for cid in ax._smk_stats_cids:
                        ax.callbacks.disconnect(cid)
                
                c1 = ax.callbacks.connect('xlim_changed', lambda a: self._update_plot_title(a))
                c2 = ax.callbacks.connect('ylim_changed', lambda a: self._update_plot_title(a))
                ax._smk_stats_cids = [c1, c2]

            except Exception as he:
                logging.debug(f"Interaction sync failed: {he}")
 
            self.notify_signal.emit("INFO", f"Plotted {actual_col}")
            self._update_stats_panel(gdf, actual_col)
            self.canvas.draw_idle()
            
            # Memory Cleanup: explicitly trigger GC after successfully rendering everything
            import gc
            self._smk_rendering = False
            self.canvas.draw()
            gc.collect()
            
        except Exception as e:
            self._smk_rendering = False
            self._stop_progress()
            logging.error(f"Render failed: {e}")
            import traceback
            traceback.print_exc()
            self.notify_signal.emit("ERROR", f"Render failed: {e}")
            import gc; gc.collect()
        finally:
            self._smk_rendering = False
            self._stop_progress()



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
             
             # Debug Click Value vs TS Value
             try:
                 # Check current value in merged GDF
                 t_val = "N/A"
                 c_idx = -1
                 if self.emissions_df is not None:
                      # Find index in emissions_df where ROW=row_look and COL=col_look
                      # Standard Gridded
                      mask = (self.emissions_df['ROW'] == row_look) & (self.emissions_df['COL'] == col_look)
                      if mask.any():
                           # Get first match
                           rec = self.emissions_df[mask].iloc[0]
                           pol = self.cmb_pollutant.currentText()
                           if pol in rec:
                                t_val = rec[pol]
                 
                 logging.info(f"DEBUG: Click Cell ({row_look}, {col_look}). Main Plot Value: {t_val}")
             except Exception as e:
                 logging.debug(f"Click debug check failed: {e}")
                 
             if not is_ncf: return
             
             from ncf_processing import get_ncf_timeseries
             self.notify_signal.emit("INFO", f"Extracting Time Series for Cell ({row_look}, {col_look})...")
             
             # Layer/Op
             l_idx = 0; l_op = 'select'
             if hasattr(self, 'cmb_ncf_layer'):
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
             cursor_mode = "by-TSTEP"
             if hasattr(self, 'cmb_cursor_mode'):
                 cursor_mode = self.cmb_cursor_mode.currentText()
                 
             if cursor_mode == "by-TSTEP":
                 from ncf_processing import get_ncf_timeseries
                 # Support for Inline TS
                 sg_path = getattr(self.emissions_df, 'attrs', {}).get('stack_groups_path')
                 
                 res = get_ncf_timeseries(
                    self.input_files_list[0],
                    self.cmb_pollutant.currentText(),
                    [r-1], [c-1],
                    layer_idx=l_idx, layer_op=l_op, op='mean',
                    stack_groups_path=sg_path
                 )
                 
                 if res:
                     self._show_ts_window(res, f"Time Series @ Cell ({r}, {c})")
                     self.notify_signal.emit("INFO", f"Plotted Cell ({r}, {c}) Time Series")
                 else:
                     self.notify_signal.emit("WARNING", "No data found for cell.")
                     
             elif cursor_mode == "by-LAY":
                 from ncf_processing import get_ncf_profile
                 # Get current timestep index
                 t_idx = self.cmb_ncf_time.currentIndex() if hasattr(self, 'cmb_ncf_time') else 0
                 
                 # Adjust layer/time op strings if they are aggregates
                 t_op = 'select'
                 if hasattr(self, 'cmb_ncf_time'):
                     if "Sum" in self.cmb_ncf_time.currentText(): t_op = 'sum'
                     elif "Avg" in self.cmb_ncf_time.currentText(): t_op = 'mean'
                 
                 res = get_ncf_profile(
                     self.input_files_list[0],
                     self.cmb_pollutant.currentText(),
                     [r-1], [c-1],
                     time_idx=t_idx, time_op=t_op, op='mean',
                     stack_groups_path=None # Inline not supported for by-LAY currently
                 )
                 
                 if res:
                     self._show_ts_window(res, f"Vertical Profile @ Cell ({r}, {c})")
                     self.notify_signal.emit("INFO", f"Plotted Cell ({r}, {c}) Profile")
                 else:
                     self.notify_signal.emit("WARNING", "No vertical profile data found.")

         except Exception as e:
             self.notify_signal.emit("ERROR", f"Extraction failed: {e}")

    def _show_ts_window(self, data, title):
        try:
             win = TimeSeriesPlotWindow(data, title, self.cmb_pollutant.currentText(), self.units_map.get(self.cmb_pollutant.currentText(), ""), self)
             # Adjust X-ax label if it's a layer profile
             if "Profile" in title and hasattr(win, 'ax'):
                 win.ax.set_xlabel("Layer", fontsize=10, fontweight='bold')
                 win.canvas.draw()
                 
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
        
        p_arr = gdf[pollutant].values if pollutant in gdf_cols else None
        
        # Memory Safety Check: Keep a light reference to verify GDF hasn't been swapped
        # during mouse motion (prevents Segfaults in background GC)
        target_ax._smk_hover_gdf = gdf
        
        # Robustly handle missing geometry for optimized grids
        geom_arr = None
        try:
            if hasattr(gdf, 'geometry') and gdf.geometry is not None:
                geom_arr = gdf.geometry.values
        except: 
            geom_arr = None

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
            # CRITICAL SAFETY: Abort if application is busy rendering a new view
            if getattr(self, '_smk_rendering', False): return ""
            if getattr(target_ax, '_smk_hover_gdf', None) is not None and target_ax._smk_hover_gdf is not gdf:
                return ""

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
                    # For optimized grids, geom_arr might be None or empty. 
                    # If we found indices via Grid Math (Strategy 1), we can trust them if geometry is missing.
                    geom = None
                    if geom_arr is not None and idx < len(geom_arr):
                        geom = geom_arr[idx]

                    # Condition: If geometry exists, must contain point. If missing, trust Strategy 1.
                    # MEMORY SAFETY: Check if geometry is still a valid object before calling GEOS methods
                    if geom is None or (hasattr(geom, 'contains') and geom.contains(pt)):
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



    def _open_scc_multi(self):
        """Open the multi-selection dialog for SCC codes."""
        if not hasattr(self, '_scc_full_list') or not self._scc_full_list:
            # Try to capture from combo if post_load was skipped or partial
            items = [self.cmb_scc.itemText(i) for i in range(self.cmb_scc.count())]
            if not items or items == ["All SCC"]:
                QMessageBox.information(self, "No SCCs", "Load emissions data first to populate the SCC list.")
                return
            self._scc_full_list = items
            
        dlg = MultiSelectionDialog("Select SCCs", self._scc_full_list, selected=self.selected_sccs, parent=self)
        if dlg.exec():
            self.selected_sccs = dlg.get_selected()
            if self.selected_sccs:
                # Update combo to show state
                self.cmb_scc.blockSignals(True)
                self.cmb_scc.clear()
                self.cmb_scc.addItem(f"Multiple ({len(self.selected_sccs)} selected)")
                self.cmb_scc.setCurrentIndex(0)
                self.cmb_scc.setToolTip("\n".join(self.selected_sccs)) # Show selected in tooltip
                self.cmb_scc.setEnabled(True)
                self.cmb_scc.blockSignals(False)
            else:
                # Revert to standard
                self.cmb_scc.blockSignals(True)
                self.cmb_scc.clear()
                self.cmb_scc.addItems(self._scc_full_list)
                self.cmb_scc.setCurrentText("All SCC")
                self.cmb_scc.setToolTip("")
                self.cmb_scc.blockSignals(False)
            
            logging.info(f"Multi-SCC selection updated: {len(self.selected_sccs)} items.")

    def _filter_pollutant_list(self, text):
        """Filter the Pollutant ComboBox based on search text."""
        if not hasattr(self, '_full_pollutant_list') or self._full_pollutant_list is None:
            self._full_pollutant_list = [self.cmb_pollutant.itemText(i) for i in range(self.cmb_pollutant.count())]
        
        current = self.cmb_pollutant.currentText()
        self.cmb_pollutant.blockSignals(True)
        self.cmb_pollutant.clear()
        query = text.lower().strip()
        
        if not query:
            filtered = self._full_pollutant_list
        else:
            filtered = [it for it in self._full_pollutant_list if query in it.lower()]
            
        self.cmb_pollutant.addItems(filtered)
        
        # Restore selection if it still exists in filtered list
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

    def export_configuration(self):
        """Standardize and export current GUI settings to a clean YAML configuration."""
        import yaml
        from datetime import datetime
        
        path, _ = QFileDialog.getSaveFileName(self, "Export Configuration", "", "YAML Config (*.yaml)")
        if not path: return
        
        self._start_progress("Exporting configuration...")
        try:
            # SCC selection logic (merge into filter-val if filter-col is SCC)
            cur_filter_col = self.cmb_filter_col.currentText()
            filter_vals = self.txt_filter_val.text().strip()
            
            # If SCC is the primary filter or selected in the SCC box, handle it
            scc_text = self.cmb_scc.currentText()
            if scc_text != "All SCC":
                if cur_filter_col == "scc" or not cur_filter_col:
                    cur_filter_col = "scc"
                    # Combine manual entries with SCC selection if needed
                    if self.selected_sccs:
                        filter_vals = ",".join(self.selected_sccs)
                    elif scc_text:
                        filter_vals = scc_text

            config = {
                'timestamp_utc': datetime.utcnow().isoformat(),
                'arguments': {
                    'filepath': self.txt_input.text() or None,
                    'sector': 'exported_session', 
                    
                    # File Parsing
                    'delim': self.cmb_delim.currentText().lower(),
                    'skiprows': self.spin_skip.value(),
                    'comment': self.txt_comment.text() or None,
                    
                    # Geometry & Paths
                    'griddesc': self.txt_griddesc.text() or None,
                    'gridname': self.cmb_gridname.currentText() if self.cmb_gridname.currentIndex() > 0 else None,
                    'county-shapefile': self.txt_counties.text() or None,
                    'overlay-shapefile': self.txt_overlay_shp.text() or None,
                    
                    # Filtering
                    'filter-col': cur_filter_col or None,
                    'filter-val': filter_vals or None,
                    'filter-start': self.txt_range_min.text() or None,
                    'filter-end': self.txt_range_max.text() or None,
                    'filter-shapefile': self.txt_filter_shp.text() if hasattr(self, 'txt_filter_shp') and self.txt_filter_shp.text() else None,
                    'filter-shapefile-opt': self.cmb_filter_op.currentText().lower() if hasattr(self, 'cmb_filter_op') else None,
                    
                    # Plotting
                    'pollutant': self.cmb_pollutant.currentText() if self.cmb_pollutant.count() > 0 else None,
                    'pltyp': self.cmb_pltyp.currentText().lower(),
                    'projection': self.cmb_proj.currentText().lower(),
                    'vmin': float(self.txt_rmin.text()) if self.txt_rmin.text() else None,
                    'vmax': float(self.txt_rmax.text()) if self.txt_rmax.text() else None,
                    'cmap': self.cmb_cmap.currentText() + ('_r' if self.chk_rev_cmap.isChecked() else ''),
                    'bins': self.txt_bins.text() or None,
                    
                    # Boolean Flags
                    'log-scale': self.chk_log.isChecked(),
                    'show-grid': True,
                    'zoom-to-data': self.chk_zoom.isChecked(),
                    'fill-nan': self.chk_nan0.isChecked() if hasattr(self, 'chk_nan0') else False,
                    
                    # NetCDF Persistence (Map UI labels to batch-friendly keywords)
                    'ncf-zdim': self._batch_dim(self.cmb_ncf_layer.currentText()) if self.ncf_frame.isVisible() else None,
                    'ncf-tdim': self._batch_dim(self.cmb_ncf_time.currentText()) if self.ncf_frame.isVisible() else None,
                }
            }
            
            # Remove None or empty strings to keep it clean
            args = config['arguments']
            config['arguments'] = {k: v for k, v in args.items() if v not in [None, ""]}
            
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
            self.notify_signal.emit("INFO", f"Configuration saved to {path}")
            QMessageBox.information(self, "Success", f"Configuration saved to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not save configuration:\n{e}")
            logging.error(f"Config export failed: {e}")
        finally:
            self._stop_progress()

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

    def _draw_graticule(self, ax, tf_fwd, tf_inv, lon_step=None, lat_step=None, with_labels=True):
        """Draw longitude/latitude lines on the given axes using provided transformers."""
        if ax is None:
            return _draw_graticule_fn(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)

        def _remove_existing():
            artists = getattr(ax, '_smk_graticule_artists', None)
            if isinstance(artists, dict):
                for key in ('lines', 'texts'):
                    try:
                        group = artists.get(key, [])
                        for art in group:
                            try: art.remove()
                            except: pass
                    except: pass
            ax._smk_graticule_artists = {'lines': [], 'texts': []}

        def _disconnect():
            cids = getattr(ax, '_smk_graticule_cids', [])
            for cid in cids:
                try: ax.callbacks.disconnect(cid)
                except: pass
            ax._smk_graticule_cids = []
            ax._smk_graticule_callback = None

        try: _remove_existing()
        except: pass
        try: _disconnect()
        except: pass

        artists = _draw_graticule_fn(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)
        if not isinstance(artists, dict):
            artists = {'lines': [], 'texts': []}
        ax._smk_graticule_artists = artists

        def _on_limits(_axis):
            if getattr(ax, '_smk_drawing_graticule', False): return
            
            # Use a timer to debounce the redraw. This prevents infinite recursion
            # when scale transitions (Linear -> Log) trigger rapid-fire limit changes.
            def _debounced_redraw():
                if getattr(ax, '_smk_drawing_graticule', False): return
                try:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    if not (np.isfinite(xlim).all() and np.isfinite(ylim).all()):
                         return
                except: return
                
                ax._smk_drawing_graticule = True
                try:
                    _remove_existing()
                    refreshed = _draw_graticule_fn(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)
                    if not isinstance(refreshed, dict):
                        refreshed = {'lines': [], 'texts': []}
                    ax._smk_graticule_artists = refreshed
                    try: ax.figure.canvas.draw_idle()
                    except: pass
                except Exception as e:
                    logging.debug(f"Debounced redraw failed: {e}")
                finally:
                    ax._smk_drawing_graticule = False

            # Increase delay to 50ms to ensure Matplotlib view state is committed
            QTimer.singleShot(50, _debounced_redraw)

        try:
            cid1 = ax.callbacks.connect('xlim_changed', _on_limits)
            cid2 = ax.callbacks.connect('ylim_changed', _on_limits)
            ax._smk_graticule_cids = [cid1, cid2]
            ax._smk_graticule_callback = _on_limits
        except: pass
        return artists

    def preview_data(self):
        """Show a table preview of the current data with integrated view switching."""
        if self.emissions_df is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        # Prepare modes (Lazy loading via lambdas where appropriate)
        # [Lazy Load Handling]
        # If emissions_df is a lazy skeleton (NetCDF), load the full dataset now so the user can see it.
        try:
            is_lazy = getattr(self.emissions_df, 'attrs', {}).get('source_type') == 'gridded_netcdf' and \
                      getattr(self.emissions_df, 'attrs', {}).get('_smk_xr_ds') is not None
            
            # Check if pollutants are actually loaded as columns
            if is_lazy:
                available = getattr(self.emissions_df, 'attrs', {}).get('available_pollutants', [])
                missing = [p for p in available if p not in self.emissions_df.columns]
                
                if missing:
                    self.notify_signal.emit("INFO", f"Loading full dataset for table view ({len(missing)} variables)...")
                    try: QApplication.processEvents() # maintain responsiveness
                    except: pass
                    
                    # Re-use read_ncf_emissions to fetch data
                    from ncf_processing import read_ncf_emissions
                    path = self.input_files_list[0] if getattr(self, 'input_files_list', None) else None
                    if path:
                        ds = self.emissions_df.attrs['_smk_xr_ds']
                        ncf_params = self.emissions_df.attrs.get('ncf_params', {})
                        
                        # We load ALL missing pollutants
                        new_data = read_ncf_emissions(path, pollutants=missing, xr_ds=ds, **ncf_params)
                        
                        # Merge columns into main DF
                        for col in missing:
                            if col in new_data.columns:
                                self.emissions_df[col] = new_data[col].values
                                # Also update units map if needed
                                if col not in self.units_map:
                                    v_meta = new_data.attrs.get('variable_metadata', {}).get(col, {})
                                    self.units_map[col] = v_meta.get('units', '')
                        
                        self.notify_signal.emit("INFO", "Dataset fully loaded.")
        except Exception as e:
            logging.error(f"Failed to lazy-load full dataset: {e}")

        modes = {
            "Raw Data (Full File)": self.raw_df if self.raw_df is not None else self.emissions_df,
        }
        
        if self._merged_gdf is not None:
             modes["Plotted Data (Filtered)"] = self._merged_gdf
             
        # Add Summaries (computed on demand)
        # Note: We pass the bound method or lambda
        modes["Summary by State"] = lambda: self._summarize_by_geo(self.raw_df if self.raw_df is not None else self.emissions_df, 'state')
        modes["Summary by County"] = lambda: self._summarize_by_geo(self.raw_df if self.raw_df is not None else self.emissions_df, 'county')
        modes["Summary by SCC (Top 2000)"] = lambda: self._summarize_by_scc(self.raw_df if self.raw_df is not None else self.emissions_df)
        modes["Summary by Grid Cell"] = lambda: self._summarize_by_grid(self.emissions_df)

        self.preview_win = TableWindow(title="Data Preview", parent=self, modes=modes)
        self.preview_win.show()

    def _summarize_by_grid(self, df):
        """Aggregate emissions by Grid Cell (ROW, COL)."""
        pols = self.pollutants
        cols = []
        if 'ROW' in df.columns and 'COL' in df.columns:
            cols = ['ROW', 'COL']
        elif 'GRID_RC' in df.columns:
            cols = ['GRID_RC']
            
        if not cols:
             # Fallback for NetCDF or unknown structure
             if hasattr(df, 'attrs') and df.attrs.get('source_type') == 'gridded_netcdf':
                  # Already gridded, return full dataset
                  return df
             raise ValueError("No grid (ROW/COL) columns found.")
             
        g = df.groupby(cols)[pols].sum().reset_index()
        return g.sort_values(pols[0], ascending=False)

    def _summarize_by_scc(self, df):
        """Aggregate emissions by SCC."""
        # Find SCC column (case-insensitive) using standard aliases and stripping whitespace
        col = next((c for c in df.columns if c.strip().lower() in SCC_COLS), None)
        if not col:
            raise ValueError("No SCC column found.")
            
        pols = self.pollutants
        if not pols: return df
        
        # Group
        g = df.groupby(col)[pols].sum().reset_index()
        
        # Add descriptions if lookup available
        desc_col = next((c for c in df.columns if c.strip().lower() in ['scc description', 'scc_description']), None)
        if desc_col:
            # Get first description for each SCC to keep it in the summary
            descs = df.groupby(col)[desc_col].first().reset_index()
            g = g.merge(descs, on=col, how='left')
            # Reorder to put description after SCC
            cols = [col, desc_col] + [p for p in pols if p in g.columns]
            g = g[cols]
            
        return g.sort_values(pols[0], ascending=False).head(2000)

    def _augment_with_county_mapping(self, df):
        """Map grid cells (ROW, COL) to counties using simple centroid mapping."""
        if self.counties_gdf is None:
             raise ValueError("County shapefile not loaded.")
        
        # 1. Get Grid Info
        info = getattr(df, 'attrs', {}).get('_smk_grid_info')
        if not info and self.grid_gdf is not None:
             # Fallback to loaded grid definition if generic
             info = getattr(self.grid_gdf, 'attrs', {}).get('_smk_grid_info')
        
        if not info:
             raise ValueError("Grid definition (XORIG, XCELL, etc) not found in dataset.")

        # 2. Get Unique Cells (for performance)
        unique_cells = df[['ROW', 'COL']].drop_duplicates()
        if unique_cells.empty: return df
        
        # 3. Calculate Centroids (Vectorized)
        # Note: SMOKE/IOAPI ROW/COL are 1-based.
        # Centroid X = XORIG + (COL_idx + 0.5) * XCELL
        # COL_idx = COL - 1
        r = unique_cells['ROW'].values - 1
        c = unique_cells['COL'].values - 1
        
        # Guard against invalid indices if any
        mask = (r >= 0) & (c >= 0) 
        # (Technically we should check max bounds too but let's assume valid data)
        
        x = info['xorig'] + (c + 0.5) * info['xcell']
        y = info['yorig'] + (r + 0.5) * info['ycell']
        
        # 4. Reproject Points to County CRS
        # Construct Grid CRS
        proj_str = info.get('proj_str')
        if not proj_str:
             if info.get('proj_type') == 2: # LCC Default Fallback (SMKPLOT constant)
                  # Re-construct simple LCC
                  proj_str = "+proj=lcc +a=6370000.0 +b=6370000.0 +units=m +no_defs" 
                  # Warning: This is a wild guess without params. Ideally we have proj_str.
        
        if not proj_str:
             raise ValueError("Grid CRS projection string missing.")
        
        # Create Points
        import pyproj
        from shapely.geometry import Point
        
        src_crs = pyproj.CRS(proj_str)
        tgt_crs = self.counties_gdf.crs or "EPSG:4326"
        
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
        lon, lat = transformer.transform(x, y)
        
        geoms = [Point(xy) for xy in zip(lon, lat)]
        points_gdf = gpd.GeoDataFrame(unique_cells, geometry=geoms, crs=tgt_crs)
        
        # 5. Spatial Join
        # Only keep necessary columns from counties (FIPS, NAME)
        # Check available columns in counties_gdf (loaded by read_shpfile)
        c_cols = ['geometry']
        fips_col = 'FIPS' if 'FIPS' in self.counties_gdf.columns else ('region_cd' if 'region_cd' in self.counties_gdf.columns else None)
        if fips_col: c_cols.append(fips_col)
        
        name_col = next((c for c in self.counties_gdf.columns if c.lower() in ['name', 'namelsad', 'county_name', 'county']), None)
        if name_col: c_cols.append(name_col)
        
        counties_subset = self.counties_gdf[c_cols]
        
        joined = gpd.sjoin(points_gdf, counties_subset, how='left', predicate='within')
        
        # 6. deduplicate (one cell -> one county). Take first match.
        joined = joined.drop_duplicates(subset=['ROW', 'COL'])
        
        # 7. Merge Map back to full DF
        # Drop geometry/index_right before merge
        merge_cols = ['ROW', 'COL']
        if fips_col in joined.columns: merge_cols.append(fips_col)
        if name_col and name_col in joined.columns: merge_cols.append(name_col)
        
        mapping = joined[merge_cols]
        
        # Merge
        result = df.merge(mapping, on=['ROW', 'COL'], how='left')
        
        # Cache for Animation Aggregation
        try:
             # Make a dict: (row, col) -> fips
             # Optimisation: Store as dataframe or simply keep the 'joined' table
             # We need FIPS column name
             if fips_col in joined.columns:
                 self._grid_to_county_map = joined[['ROW', 'COL', fips_col]].copy()
                 self._grid_to_county_key = fips_col
             else:
                 self._grid_to_county_map = None
        except: pass
        
        return result

    def _summarize_by_geo(self, df, mode):
        """Aggregate by State/County (Replicating gui_qt logic)."""
        pols = self.pollutants
        raw = df.copy()
        lower_map = {c.lower(): c for c in raw.columns}
        mode_l = (mode or '').strip().lower()
        
        group_cols = []
        add_cols = {} # Extra columns to add to result
        
        # --- COUNTY MODE ---
        if mode_l == 'county':
            key = None
            if 'fips' in lower_map: key = lower_map['fips']
            elif 'region_cd' in lower_map: key = lower_map['region_cd']
            
            # --- Grid Centroid Mapping Logic ---
            if not key and ('ROW' in df.columns and 'COL' in df.columns):
                 try:
                     df_mapped = self._augment_with_county_mapping(df)
                     # Re-scan columns on the augmented DF
                     lower_map = {c.lower(): c for c in df_mapped.columns}
                     if 'fips' in lower_map: 
                         key = lower_map['fips']
                         raw = df_mapped # Use the mapped DF for subsequent logic
                     elif 'region_cd' in lower_map: 
                         key = lower_map['region_cd']
                         raw = df_mapped
                     
                     # Ensure we have FIPS even if they are NaN (unmapped)
                     # The groupby(dropna=False) below will handle them.
                 except Exception as e:
                     logging.warning(f"Grid-to-County mapping failed: {e}")
            # -----------------------------------

            if not key:
                raise ValueError("Cannot summarize by county: Missing FIPS/region_cd column and Grid Mapping failed/unavailable (Load County Shapefile).")
                
            group_cols = [key]
            
            # Derive STATEFP and names when possible (like gui_qt)
            try:
                fips_series = raw[key].astype(str).str.zfill(6)
                add_cols['COUNTRY_ID'] = fips_series.str[0]
                add_cols['STATEFP'] = fips_series.str[1:3]
                add_cols['STATE_NAME'] = add_cols['STATEFP'].map(US_STATE_FIPS_TO_NAME)
            except Exception: pass
            
            # If COUNTY or COUNTY_NAME column exists in input, carry it (like gui_qt)
            try:
                county_name_col = lower_map.get('county name') or lower_map.get('county') or lower_map.get('county_name')
                if county_name_col and county_name_col in raw.columns:
                     # Take most frequent name per FIPS to avoid duplicates
                     tmp_cn = raw[[key, county_name_col]].copy()
                     name_map = tmp_cn.dropna(subset=[county_name_col]).drop_duplicates(subset=[key]).set_index(key)[county_name_col]
                     add_cols['COUNTY_NAME'] = raw[key].map(name_map)
            except Exception: pass

        # --- STATE MODE ---
        elif mode_l == 'state':
            fips_col = None
            if 'fips' in lower_map: fips_col = lower_map['fips']
            elif 'region_cd' in lower_map: fips_col = lower_map['region_cd']
            
            # --- Grid Centroid Mapping Logic ---
            if not fips_col and 'state_cd' not in lower_map and ('ROW' in df.columns and 'COL' in df.columns):
                 try:
                     df_mapped = self._augment_with_county_mapping(df)
                     lower_map = {c.lower(): c for c in df_mapped.columns}
                     if 'fips' in lower_map: 
                         fips_col = lower_map['fips']
                         raw = df_mapped
                     elif 'region_cd' in lower_map: 
                         fips_col = lower_map['region_cd']
                         raw = df_mapped
                 except Exception: pass
            # -----------------------------------

            if not fips_col:
                # Try state code directly
                if 'state_cd' in lower_map:
                    group_cols = [lower_map['state_cd']]
                else:
                    raise ValueError("Cannot summarize by state: FIPS/region_cd/state_cd not available and Grid Mapping failed.")
            else:
                # Derive STATEFP from FIPS (gui_qt logic)
                # FIPS is typically 6 chars (C+SS+CCC). We want SS.
                raw['STATEFP'] = raw[fips_col].astype(str).str.strip().str.zfill(6).str[1:3]
                group_cols = ['STATEFP']
                
            # Add Name
            try:
                col_to_map = group_cols[0]
                # If grouping by STATEFP, map it. If state_cd, assume it might be state code
                raw['STATE_NAME'] = raw[col_to_map].map(US_STATE_FIPS_TO_NAME)
                add_cols['STATE_NAME'] = raw['STATE_NAME']
            except Exception: pass
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Perform Aggregation
        if not group_cols: return df
        
        g = raw.groupby(group_cols, dropna=False)[pols].sum().reset_index()
        
        # Attach enriched columns
        # We need to map back the enriched data to the grouped result
        # For STATEFP/Name, since we grouped BY it (in state mode), it's already there or easy to map
        # For County mode, we grouped by FIPS, so we map back attributes based on FIPS
        
        try:
            if mode_l == 'state':
                 # If we grouped by STATEFP, add STATE_NAME
                 if 'STATE_NAME' in add_cols and 'STATE_NAME' not in g.columns:
                      # Re-map using the group key
                      key_col = group_cols[0]
                      g['STATE_NAME'] = g[key_col].map(US_STATE_FIPS_TO_NAME)
                      
            elif mode_l == 'county':
                 # We grouped by FIPS (key).
                 # We need to reconstruct the attributes for each FIPS in g
                 key_col = group_cols[0]
                 
                 # Helper to get first valid value from original derived cols
                 # Actually, simpler: just re-derive from the grouped key since it's 1:1
                 f_series = g[key_col].astype(str).str.zfill(6)
                 
                 if 'STATE_NAME' in add_cols:
                      statefp = f_series.str[1:3]
                      g['STATE_NAME'] = statefp.map(US_STATE_FIPS_TO_NAME)
                      
                 if 'COUNTY_NAME' in add_cols:
                      # We need the map we built earlier
                      # Re-build map from raw data as we did before
                      county_name_col = lower_map.get('county name') or lower_map.get('county') or lower_map.get('county_name')
                      if county_name_col:
                           tmp_cn = raw[[key_col, county_name_col]].copy()
                           name_map = tmp_cn.dropna(subset=[county_name_col]).drop_duplicates(subset=[key_col]).set_index(key_col)[county_name_col]
                           g['COUNTY_NAME'] = g[key_col].map(name_map)

        except Exception as e:
            logging.warning(f"Failed to attach enriched summaries: {e}")
            
        # Reorder for niceness
        cols_out = group_cols + [c for c in ['STATE_NAME', 'COUNTY_NAME'] if c in g.columns] + pols
        # Filter to only existing cols
        cols_out = [c for c in cols_out if c in g.columns]
        g = g[cols_out]
        
        # Fill NaNs in key columns for display
        for col in group_cols:
             if g[col].isna().any():
                 g[col] = g[col].fillna("Unmapped")
        if 'COUNTY_NAME' in g.columns:
             g['COUNTY_NAME'] = g['COUNTY_NAME'].fillna("Outside Domain")
        if 'STATE_NAME' in g.columns:
             g['STATE_NAME'] = g['STATE_NAME'].fillna("Unknown")

        return g.sort_values(pols[0], ascending=False)

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
            fips_col = next((c for c in raw.columns if c.strip().lower() in REGION_COLS), None)
            if not fips_col: raise ValueError("No FIPS column found.")
            group_cols = [fips_col]
        elif mode == 'state':
            fips_col = next((c for c in raw.columns if c.strip().lower() in REGION_COLS), None)
            if not fips_col: 
                # Fallback to checking any column that might look like FIPS
                fips_col = next((c for c in raw.columns if 'fips' in c.lower()), None)
            
            if not fips_col: raise ValueError("No FIPS column found for state summary.")
            # Local copy to add state
            raw = raw.copy()
            raw['STATEFP'] = raw[fips_col].astype(str).str.zfill(6).str[1:3]
            group_cols = ['STATEFP']
        elif mode == 'scc':
            scc_col = next((c for c in raw.columns if c.strip().lower() in SCC_COLS), None)
            if not scc_col: raise ValueError("No SCC column found.")
            desc_cols = [c for c in raw.columns if c.strip().lower() in ['scc description', 'scc_description']]
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
        self.is_profile = "Profile" in title
        self.setWindowTitle(title if self.is_profile else f"Time Series - {title}")
        w, h = _get_adaptive_window_size(800, 500, min_w=700, min_h=450)
        self.resize(w, h)
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
        
        # Convert times to datetime if string (only for regular TS)
        if not self.is_profile:
            try:
                if times and isinstance(times[0], str):
                    try:
                        # Attempt detailed SMOKE format parsing first: YYYYDDD_HHMMSS
                        times = [pd.to_datetime(t, format='%Y%j_%H%M%S') for t in times]
                    except:
                        # Fallback to auto-detection
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
                if self.is_profile:
                    ax.plot(series, range(len(times)), marker='o', markersize=ms, label=label, linewidth=lw, alpha=alpha)
                else:
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

            if self.is_profile:
                ax.plot(vals, range(len(times)), marker='o', linestyle='-', markersize=4)
            else:
                ax.plot(times, vals, marker='o', linestyle='-', markersize=4)

        u_str = str(self.unit or '').strip()
        
        if self.is_profile:
            val_lbl = f"{self.pollutant} ({u_str})" if u_str else self.pollutant
            ax.set_xlabel(val_lbl)
            ax.set_ylabel("Layer")
            ax.set_yticks(range(len(times)))
            ax.set_yticklabels(times)
        else:
            y_lbl = f"{self.pollutant} ({u_str})" if u_str else self.pollutant
            ax.set_ylabel(y_lbl)
            ax.set_xlabel("Time Step")
            # Date formatting
            import matplotlib.dates as mdates
            if len(times) > 0:
                 try:
                     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                     self.figure.autofmt_xdate()
                 except: pass

        ax.grid(True, linestyle='--', alpha=0.7)
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
