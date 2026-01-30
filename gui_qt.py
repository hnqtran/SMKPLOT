#!/proj/ie/proj/SMOKE/htran/Emission_Modeling_Platform/utils/smkplot/.venv/bin/python
"""GUI components for SMKPLOT.

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
import copy
import logging
import threading
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
    DEFAULT_ONLINE_COUNTIES_URL
)
from utils import normalize_delim, is_netcdf_file
from data_processing import (
    read_inputfile, read_shpfile, extract_grid, create_domain_gdf, detect_pollutants, map_latlon2grd,
    merge_emissions_with_geometry, filter_dataframe_by_range, filter_dataframe_by_values, get_emis_fips, apply_spatial_filter
)
from plotting import _plot_crs, _draw_graticule as _draw_graticule_fn, create_map_plot

# Backend selection: try Tk if DISPLAY exists, otherwise Agg

# --- Qt Compatibility Layer (Antigravity v2) ---
import os
import logging
# Headless detection to prevent hanging
if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    logging.info("No display detected, setting Qt to offscreen mode")

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, 
                               QFileDialog, QMessageBox, QProgressBar, QTabWidget, 
                               QSplitter, QFrame, QSizePolicy, QScrollArea, QGridLayout,
                               QMenu, QMenuBar, QStatusBar, QListWidget, QTextEdit,
                               QLayout, QTreeWidget, QTreeWidgetItem, QStyle, QListView)
from PySide6.QtCore import Qt, Signal, QObject, QThread, QTimer, QSize
from PySide6.QtGui import QAction, QIcon, QFont, QIntValidator, QDoubleValidator, QTextCursor

import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as _FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

def _qt_align(sticky):
    if not sticky: return Qt.AlignCenter
    s = str(sticky).lower()
    
    # Initialize with empty alignment
    a = Qt.Alignment()
    
    # Horizontal
    if 'w' in s and 'e' in s: pass 
    elif 'w' in s: a |= Qt.AlignLeft
    elif 'e' in s: a |= Qt.AlignRight
    else: a |= Qt.AlignHCenter
    
    # Vertical
    if 'n' in s and 's' in s: pass
    elif 'n' in s: a |= Qt.AlignTop
    elif 's' in s: a |= Qt.AlignBottom
    else: a |= Qt.AlignVCenter
    
    return a

class QtTkMixin:
    """Mixin to add Tk-like methods (grid, pack, configure, bind) to QWidgets."""
    def __init__(self):
        self._grid_info = {}
        
    def grid(self, row=0, column=0, rowspan=1, columnspan=1, sticky='', padx=0, pady=0, **kwargs):
        p = self.parentWidget()
        if isinstance(p, QMainWindow):
             cw = p.centralWidget()
             if cw: p = cw
        if p:
            if p.layout() is None:
                l = QGridLayout()
                l.setContentsMargins(0, 0, 0, 0)
                l.setSpacing(1)
                p.setLayout(l)
            if isinstance(p.layout(), QGridLayout):
                align = _qt_align(sticky)
                p.layout().addWidget(self, row, column, rowspan, columnspan, align)

    def pack(self, side='top', fill='none', expand=False, padx=0, pady=0, **kwargs):
        # Map pack to vertical or horizontal stack in a Grid
        p = self.parentWidget()
        if isinstance(p, QMainWindow):
             cw = p.centralWidget()
             if cw: p = cw
        if p:
            if p.layout() is None: p.setLayout(QGridLayout())
            layout = p.layout()
            if isinstance(layout, QGridLayout):
                if layout.count() == 0:
                    row = 0
                    col = 0
                else:
                    r, c, rs, cs = layout.getItemPosition(layout.count() - 1)
                    if str(side).lower() in ('left', 'right'):
                        row = r
                        col = c + cs
                    else:
                        row = r + rs
                        col = 0
                
                st = ''
                if fill in ('both', 'x'): st += 'ew'
                if fill in ('both', 'y'): st += 'ns'
                if expand: st += 'nsew'
                
                self.grid(row=row, column=col, sticky=st, padx=padx, pady=pady)
                
                # If expand is requested, we must tell the layout to grow this row/column
                if expand:
                    self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    layout.setRowStretch(row, 1)
                    layout.setColumnStretch(col, 1)

    def configure(self, **kwargs):
        if 'state' in kwargs: self.setEnabled(kwargs['state'] != "disabled")
        if 'text' in kwargs and hasattr(self, 'setText'): self.setText(kwargs['text'])
        if 'command' in kwargs:
             try: self.clicked.disconnect() 
             except: pass
             if kwargs['command']: self.clicked.connect(kwargs['command'])
        if 'values' in kwargs and hasattr(self, 'clear') and hasattr(self, 'addItems'):
             self.clear()
             self.addItems([str(v) for v in kwargs['values']])
        if 'image' in kwargs: pass # skip images for now

    def config(self, **kwargs):
        self.configure(**kwargs)
             
    def columnconfigure(self, index, weight=0, minsize=0):
        l = self.layout()
        if l and isinstance(l, QGridLayout): l.setColumnStretch(index, weight)
    def rowconfigure(self, index, weight=0, minsize=0):
        l = self.layout()
        if l and isinstance(l, QGridLayout): l.setRowStretch(index, weight)
    def grid_remove(self): self.hide()
    def grid_forget(self): self.hide()
    def bind(self, sequence=None, func=None, add=None):
        # Basic mapping for Entry events
        if hasattr(self, 'returnPressed') and sequence == '<Return>':
            self.returnPressed.connect(lambda: func(None))
        if hasattr(self, 'editingFinished') and sequence == '<FocusOut>':
            self.editingFinished.connect(lambda: func(None))
        if hasattr(self, 'itemSelectionChanged') and 'Select' in sequence:
             self.itemSelectionChanged.connect(lambda: func(None))

# --- Widget Shims ---
class Frame(QFrame, QtTkMixin):
    def __init__(self, master=None, **kwargs):
        super().__init__(master)
        QtTkMixin.__init__(self)
        self.setLayout(QGridLayout())
        self.configure(**kwargs)

class Label(QLabel, QtTkMixin):
    def __init__(self, master=None, text='', textvariable=None, **kwargs):
        super().__init__(text, master)
        QtTkMixin.__init__(self)
        if textvariable:
            self.setText(str(textvariable.get()))
            if hasattr(textvariable, 'changed'):
                textvariable.changed.connect(lambda v: self.setText(str(v)))
            else:
                textvariable.trace_add('write', lambda *_: self.setText(str(textvariable.get())))
        self.configure(**kwargs)

class Button(QPushButton, QtTkMixin):
    def __init__(self, master=None, text='', command=None, **kwargs):
        super().__init__(text, master)
        QtTkMixin.__init__(self)
        if command: self.clicked.connect(command)
        self.configure(**kwargs)

class Checkbutton(QCheckBox, QtTkMixin):
    def __init__(self, master=None, text='', variable=None, command=None, **kwargs):
        super().__init__(text, master)
        QtTkMixin.__init__(self)
        if variable:
            self.setChecked(bool(variable.get()))
            if hasattr(variable, 'changed'):
                variable.changed.connect(lambda _: self.setChecked(bool(variable.get())))
            else:
                variable.trace_add('write', lambda *_: self.setChecked(bool(variable.get())))
            self.toggled.connect(lambda b: variable.set(b))
        if command: self.clicked.connect(command)
        self.configure(**kwargs)
        
class Entry(QLineEdit, QtTkMixin):
    def __init__(self, master=None, width=None, textvariable=None, **kwargs):
        super().__init__(master)
        QtTkMixin.__init__(self)
        if hasattr(self, 'setFixedWidth') and width:
            self.setFixedWidth(int(width)*8) # approx char width
        if textvariable:
            self.setText(str(textvariable.get()))
            self.textChanged.connect(lambda t: textvariable.set(t))
            if hasattr(textvariable, 'changed'):
                textvariable.changed.connect(lambda v: self.setText(str(v)) if self.text() != str(v) else None)
            else:
                textvariable.trace_add('write', lambda *_: self.setText(str(textvariable.get())) if self.text() != str(textvariable.get()) else None)
    def delete(self, first, last=None): self.setText("")
    def insert(self, index, string): self.setText(string)
    def get(self): return self.text()

class Combobox(QComboBox, QtTkMixin):
    def __init__(self, master=None, values=None, textvariable=None, state=None, **kwargs):
        super().__init__(master)
        QtTkMixin.__init__(self)
        if values: self.addItems(values)
        if textvariable:
            self.currentTextChanged.connect(lambda t: textvariable.set(t))
            if hasattr(textvariable, 'changed'):
                textvariable.changed.connect(lambda v: self.setCurrentText(str(v)) if self.currentText() != str(v) else None)
            else:
                textvariable.trace_add('write', lambda *_: self.setCurrentText(str(textvariable.get())) if self.currentText() != str(textvariable.get()) else None)
    def current(self, newindex=None):
        if newindex is None: return self.currentIndex()
        self.setCurrentIndex(newindex)
    def get(self): return self.currentText()
    def set(self, value): self.setCurrentText(value) # Not standard Tk, but useful

    def _check_multi_column(self):
        if self.count() > 25:
            v = QListView()
            v.setFlow(QListView.LeftToRight)
            v.setWrapping(True)
            v.setResizeMode(QListView.Adjust)
            v.setSpacing(2)
            fm = self.fontMetrics()
            max_w = 120
            for i in range(self.count()):
                max_w = max(max_w, fm.horizontalAdvance(str(self.itemText(i))) + 20)
            v.setGridSize(QSize(max_w, 24))
            self.setView(v)
            popup_w = min(1200, max_w * 4 + 40)
            v.setMinimumWidth(popup_w)

    def addItem(self, *args, **kwargs):
        super().addItem(*args, **kwargs)
        self._check_multi_column()

    def addItems(self, *args, **kwargs):
        super().addItems(*args, **kwargs)
        self._check_multi_column()

class OptionMenuBuf:
    def __init__(self, combo):
        self.combo = combo
    def delete(self, first, last=None):
        self.combo.clear()
    def add_command(self, label=None, command=None):
        if label: self.combo.addItem(str(label))

class OptionMenu(QComboBox, QtTkMixin):
    def __init__(self, master, variable, value, *values, command=None, **kwargs):
        super().__init__(master)
        QtTkMixin.__init__(self)
        items = list(values)
        self.addItems([str(v) for v in items])
        self.setCurrentText(str(value))
        if variable:
            variable.set(value)
            self.currentTextChanged.connect(lambda t: variable.set(t))
            if hasattr(variable, 'changed'):
                variable.changed.connect(lambda v: self.setCurrentText(str(v)) if self.currentText() != str(v) else None)
            else:
                variable.trace_add('write', lambda *_: self.setCurrentText(str(variable.get())) if self.currentText() != str(variable.get()) else None)
        if command:
            self.currentTextChanged.connect(lambda t: command(t))
        self.configure(**kwargs)
        self._check_multi_column()

    def _check_multi_column(self):
        # If many items, use a grid-like wrapping view
        if self.count() > 25:
            v = QListView()
            v.setFlow(QListView.LeftToRight)
            v.setWrapping(True)
            v.setResizeMode(QListView.Adjust)
            v.setSpacing(2)
            # Find max width of text to set grid size
            fm = self.fontMetrics()
            max_w = 120
            for i in range(self.count()):
                max_w = max(max_w, fm.horizontalAdvance(self.itemText(i)) + 20)
            v.setGridSize(QSize(max_w, 24))
            self.setView(v)
            # Make the popup wide enough (e.g. 4 columns or max 1000px)
            popup_w = min(1200, max_w * 4 + 40)
            v.setMinimumWidth(popup_w)

    def addItem(self, *args, **kwargs):
        super().addItem(*args, **kwargs)
        self._check_multi_column()

    def addItems(self, *args, **kwargs):
        super().addItems(*args, **kwargs)
        self._check_multi_column()
    def __getitem__(self, key):
        if key == 'menu': return OptionMenuBuf(self)
        raise KeyError(key)

class Text(QTextEdit, QtTkMixin):
    def __init__(self, master=None, height=10, width=90, **kwargs):
        super().__init__(master)
        QtTkMixin.__init__(self)
    def delete(self, start, end=None): self.clear()
    def insert(self, index, chars): self.insertPlainText(chars if chars else "")
    def see(self, index): pass
    def get(self, start, end=None): return self.toPlainText()

class Scrollbar(QWidget, QtTkMixin):
    def __init__(self, master=None, orient='vertical', command=None, **kwargs):
        super().__init__(master)
        QtTkMixin.__init__(self)
        self.hide()
    def set(self, *args): pass

class Style:
    def __init__(self, master=None): pass
    def configure(self, style, **kwargs): pass
    def map(self, style, **kwargs): pass
    def theme_use(self, theme): pass

class Treeview(QTreeWidget, QtTkMixin):
    def __init__(self, master=None, columns=None, show=None, **kwargs):
        super().__init__(master)
        QtTkMixin.__init__(self)
        if columns:
            self.setColumnCount(len(columns))
            self.setHeaderLabels(list(columns))
    def heading(self, column, text=None, command=None): pass
    def column(self, column, width=None, minwidth=None, stretch=None, anchor=None, **kwargs): pass
    def insert(self, parent, index, iid=None, **kwargs):
        values = kwargs.get('values')
        item = QTreeWidgetItem(self)
        if values:
            for i, v in enumerate(values):
                item.setText(i, str(v))
        return item
    def delete(self, *items): self.clear()
    def get_children(self, item=None): return []
    def selection(self): return self.selectedItems()
    def yview(self, *args): pass
    def xview(self, *args): pass


# --- Matplotlib Shims ---
class NavigationToolbar2Tk(NavigationToolbar, QtTkMixin):
    def __init__(self, canvas, window, **kwargs):
        super().__init__(canvas, window)
        QtTkMixin.__init__(self)
        self.update()

class FigureCanvas(_FigureCanvasQTAgg, QtTkMixin):
    def __init__(self, figure, master=None, **kwargs):
        _FigureCanvasQTAgg.__init__(self, figure)
        QtTkMixin.__init__(self)
        if master: self.setParent(master)
    def get_tk_widget(self): return self

FigureCanvasTkAgg = FigureCanvas

# --- Variable Shims ---
class StringVar(QObject):
    changed = Signal(str)
    def __init__(self, master=None, value=None, name=None):
        super().__init__()
        self._val = str(value) if value is not None else ''
        self._callbacks = {}
    def get(self): return self._val
    def set(self, value):
        if self._val != str(value):
            self._val = str(value)
            self.changed.emit(self._val)
            for cb in self._callbacks.values(): cb()
    def trace_add(self, mode, callback):
        # We ignore mode, assume 'write'
        cb_id = str(id(callback))
        self._callbacks[cb_id] = callback
        return cb_id
class BooleanVar(StringVar):
    def get(self): return bool(self._val == 'True' or self._val == '1')
    def set(self, value): super().set(str(value))
class IntVar(StringVar):
    def get(self): return int(float(self._val or 0))
    def set(self, value): super().set(str(value))
class DoubleVar(StringVar):
    def get(self): return float(self._val or 0.0)

# --- Root/Toplevel Shim ---
class Root(QMainWindow, QtTkMixin):
    notify_signal = Signal(str, str)
    finalize_signal = Signal(bool, object, object)
    run_on_main_signal = Signal(object)
    def __init__(self, *args, **kwargs):
        master = args[0] if args else kwargs.get('master')
        super().__init__(master)
        if QApplication.instance():
             QApplication.instance().setStyle("Fusion")
             self._apply_modern_theme()
        QtTkMixin.__init__(self)
        self.run_on_main_signal.connect(lambda f: f())
        self._central = QWidget()
        self.setCentralWidget(self._central)
        l = QGridLayout()
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        self._central.setLayout(l)
        self._protocols = {}
        if master:
            self.show()

    def _apply_modern_theme(self):
        qss = """
        QMainWindow, QWidget {
            background-color: #f8f9fa;
            color: #212529;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 10pt;
        }
        QPushButton {
            background-color: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 5px 15px;
            color: #212529;
        }
        QPushButton:hover {
            background-color: #e9ecef;
            border-color: #adb5bd;
        }
        QPushButton:pressed {
            background-color: #dee2e6;
        }
        QLineEdit, QComboBox, QSpinBox {
            background-color: #ffffff;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 4px 8px;
            color: #212529;
        }
        QLineEdit:focus, QComboBox:focus {
            border: 1px solid #007bff;
            background-color: #ffffff;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #ced4da;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #6c757d;
            margin-top: 2px;
        }
        QComboBox QAbstractItemView, QListView {
            background-color: #ffffff;
            selection-background-color: #007bff;
            selection-color: white;
            color: #212529;
            border: 1px solid #ced4da;
            outline: none;
        }
        QListView::item {
            padding: 4px;
            border-radius: 2px;
        }
        QListView::item:hover {
            background-color: #e9ecef;
        }
        QProgressBar {
            border: none;
            background-color: #e9ecef;
            height: 4px;
        }
        QProgressBar::chunk {
            background-color: #007bff;
        }
        QStatusBar {
            background-color: #f1f3f5;
            color: #495057;
            border-top: 1px solid #dee2e6;
        }
        QToolTip {
            background-color: #ffffff;
            color: #212529;
            border: 1px solid #ced4da;
            padding: 5px;
        }
        """
        QApplication.instance().setStyleSheet(qss)
    def title(self, text): self.setWindowTitle(text)
    def geometry(self, g): 
        try: 
            w, h = map(int, g.split('+')[0].split('x'))
            self.resize(w, h)
        except: pass
    def protocol(self, n, f): 
        if n == "WM_DELETE_WINDOW": self._protocols['close'] = f
    def closeEvent(self, e):
        if 'close' in self._protocols:
            try:
                self._protocols['close']()
            except SystemExit:
                e.accept()
                return
            except Exception:
                pass
            e.ignore() 
        else:
            e.accept()
    def withdraw(self): self.hide()
    def deiconify(self): self.show()
    def update_idletasks(self): QApplication.processEvents()
    def rowconfigure(self, i, weight=0): 
        if self._central.layout(): self._central.layout().setRowStretch(i, weight)
    def columnconfigure(self, i, weight=0):
        if self._central.layout(): self._central.layout().setColumnStretch(i, weight)
    def winfo_screenwidth(self):
        s = QApplication.primaryScreen()
        return s.size().width() if s else 1600
    def destroy(self): self.close()
    def after(self, ms, func): 
        QTimer.singleShot(ms, func)
        return "id"
    def minsize(self, w, h): self.setMinimumSize(int(w), int(h))
    def winfo_width(self): return self.width()
    def option_add(self, *args): pass
    def resizable(self, *args): pass

# --- Namespace Shims ---
class tk_ns:
    Tk = Root
    Toplevel = Root
    Frame=Frame; Label=Label; Button=Button; Checkbutton=Checkbutton; Entry=Entry; Combobox=Combobox; OptionMenu=OptionMenu; Text=Text
    StringVar=StringVar; BooleanVar=BooleanVar; IntVar=IntVar; DoubleVar=DoubleVar
    END="end"; W="w"; E="e"; N="n"; S="s"; EW="ew"; NSEW="nsew"
    NORMAL="normal"; DISABLED="disabled"

class ttk_ns:
    Frame=Frame; Label=Label; Button=Button; Checkbutton=Checkbutton; Entry=Entry; Combobox=Combobox; OptionMenu=OptionMenu; Treeview=Treeview; Scrollbar=Scrollbar; Style=Style

class messagebox:
    @staticmethod
    def showerror(t, m): QMessageBox.critical(None, t, m)
    @staticmethod
    def showinfo(t, m): QMessageBox.information(None, t, m)
    @staticmethod
    def showwarning(t, m): QMessageBox.warning(None, t, m)

def _tk_ft(ft):
    if not ft: return ""
    if isinstance(ft, str): return ft
    try:
        return ";;".join([f"{n} ({p})" for n, p in ft])
    except: return ""

class filedialog:
    @staticmethod
    def askopenfilename(**k): 
        return QFileDialog.getOpenFileName(None, k.get('title','Open'), k.get('initialdir',''), _tk_ft(k.get('filetypes')))[0]
    @staticmethod
    def askopenfilenames(**k): 
        f = QFileDialog.getOpenFileNames(None, k.get('title','Open'), k.get('initialdir',''), _tk_ft(k.get('filetypes')))[0]
        return list(f) if f else []
    @staticmethod
    def asksaveasfilename(**k): 
        return QFileDialog.getSaveFileName(None, k.get('title','Save'), k.get('initialdir',''), _tk_ft(k.get('filetypes')))[0]

# --- Global Aliases ---
tk = tk_ns
ttk = ttk_ns
USING_TK = True
class EmissionGUI:
    def __init__(self, root, inputfile_path: Optional[str], counties_path: Optional[str], emissions_delim: Optional[str] = None, *, cli_args=None, app_version: str = "1.0"):
        self.root = root
        if hasattr(self.root, 'notify_signal'):
             self.root.notify_signal.connect(self._loader_notify)
        if hasattr(self.root, 'finalize_signal'):
             self.root.finalize_signal.connect(lambda p, l, s: self._finalize_loaded_emissions(show_preview=p, source_label=l, scc_data=s))
        self.root.title(f"SMKPLOT version {app_version} (Author: tranhuy@email.unc.edu)")
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
        
        # Prioritize arguments from CLI before configuration snapshot        
        self.emissions_delim = getattr(cli_args, 'delim', None) if cli_args else None or self._json_arguments.get('delim', None)
        self.emissions_delim = normalize_delim(self.emissions_delim)
        self.sector = getattr(cli_args, 'sector', None) if cli_args else None or self._json_arguments.get('sector', None)
        self.skiprows = getattr(cli_args, 'skiprows', None) if cli_args else None or self._json_arguments.get('skiprows', None)
        self.comment_token = getattr(cli_args, 'comment', None) if cli_args else None or self._json_arguments.get('comment', None)
        self.encoding = getattr(cli_args, 'encoding', None) if cli_args else None or self._json_arguments.get('encoding', None)
        
        # fill_nan logic
        self.fill_nan_arg = getattr(cli_args, 'fill_nan', None) if cli_args else None or self._json_arguments.get('fill_nan', None)
        
        # Fill NaN with Zero checkbox state
        # If fill_nan_arg is approximately 0, default to checked
        default_zero = False
        if self.fill_nan_arg is not None:
            try:
                if abs(float(self.fill_nan_arg)) < 1e-9:
                    default_zero = True
            except Exception:
                pass
        self.fill_zero_var = tk.BooleanVar(value=default_zero)

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

        # NetCDF Dimension Config (from CLI/JSON)
        self.ncf_tdim = getattr(cli_args, 'ncf_tdim', 'avg') if cli_args else 'avg'
        self.ncf_zdim = getattr(cli_args, 'ncf_zdim', '0') if cli_args else '0'

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

        # State for "Update, Don't Replot" optimization
        # Mapping: pollutant -> { 'win': Toplevel, 'fig': Figure, 'ax': Axes, 'canvas': Canvas, 'stats_cbids': list }
        self._plot_windows = {}
        
        # Restore UI state first
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

        # --- Pre-calculate settings for UI initialization ---
        # Load optional overlay shapefile
        self.overlay_path  = getattr(cli_args, 'overlay_shapefile', None) if cli_args else None or self._json_arguments.get('overlay_shapefile', None)
        
        # Load filter shapefile (new parameter)
        self.filter_path = getattr(cli_args, 'filter_shapefile', None) if cli_args else None or self._json_arguments.get('filter_shapefile', None)

        # Load filter_shapefile_opt setting (new parameter)
        val_filter_opt = getattr(cli_args, 'filter_shapefile_opt', None)
        if val_filter_opt is None:
            val_filter_opt = self._json_arguments.get('filter_shapefile_opt', None)

        # Load filtered_by_overlay setting (deprecated, backward compatibility)
        val_filter = getattr(cli_args, 'filtered_by_overlay', None) 
        if val_filter is None:
            val_filter = self._json_arguments.get('filtered_by_overlay', None)
        
        # Determine initial filter mode
        initial_filter_mode = 'False'
        raw_filter = val_filter_opt if val_filter_opt is not None else val_filter
        if raw_filter:
            if isinstance(raw_filter, bool):
                initial_filter_mode = 'intersect' if raw_filter else 'False'
            elif isinstance(raw_filter, str):
                s_filter = raw_filter.strip().lower()
                if s_filter in ('true', 'yes', 'on'):
                    initial_filter_mode = 'intersect'
                elif s_filter in ('false', 'no', 'off', 'none', 'null', ''):
                    initial_filter_mode = 'False'
                else:
                    initial_filter_mode = raw_filter
        self.initial_filter_overlay = initial_filter_mode
        # --- End pre-calculation ---

        self._build_layout()

        # Restore UI choices (linear/log, bins, cmap, delim, etc.)
        if self._json_arguments.get('log_scale') is True:
            self.scale_var.set('log')
        elif self._json_arguments.get('log_scale') is False:
            self.scale_var.set('linear')
        else:
            self.scale_var.set('linear')

        bins_val = self._json_arguments.get('bins')
        if bins_val is not None:
            self.class_bins_var.set(str(bins_val))
        else:
            self.class_bins_var.set('')

        cmap_val = self._json_arguments.get('cmap')
        if cmap_val:
            self.cmap_var.set(str(cmap_val))
        else:
            self.cmap_var.set('viridis')

        # Restore from CLI or JSON snapshot
        self.inputfile_path = inputfile_path or (getattr(cli_args, 'filepath', None) if cli_args else None) or self._json_arguments.get('filepath')

        # --- Initialize logic-heavy fields and trigger loaders ---
        
        # 1. Emissions Input
        if isinstance(self.inputfile_path, (list, tuple)):
            self.inputfile_path = "; ".join([str(p) for p in self.inputfile_path if p])
            
        if self.inputfile_path:
            try:
                self.emis_entry.delete(0, tk.END)
                self.emis_entry.insert(0, str(self.inputfile_path))
            except Exception:
                pass
            # Auto-load if coming from CLI or JSON snapshot
            if self.json_payload or (cli_args and getattr(cli_args, 'filepath', None)):
                self.root.after(200, lambda: self.load_inputfile(show_preview=False))

        # 2. GRIDDESC
        self.griddesc_path = getattr(cli_args, 'griddesc', None) or self._json_arguments.get('griddesc')
        if self.griddesc_path == "(NetCDF Attributes)":
            self.griddesc_path = None

        if self.griddesc_path:
            try:
                self.griddesc_entry.delete(0, tk.END)
                self.griddesc_entry.insert(0, str(self.griddesc_path))
            except Exception:
                pass
            self.load_griddesc()
            
            # Restore selected grid name (Priority: CLI/Config)
            grid_name = getattr(cli_args, 'gridname', None) or self._json_arguments.get('gridname')
            if self.grid_name_var and grid_name and grid_name != "Select Grid":
                self.grid_name_var.set(grid_name)
                self.load_grid_shape()

        # 3. SCC Selection (Handled by load_inputfile if present)

        # 4. Counties Shapefile
        self.counties_path = counties_path or getattr(cli_args, 'county_shapefile', None) or self._json_arguments.get('county_shapefile')
        if self.counties_path:
            try:
                self.county_entry.delete(0, tk.END)
                self.county_entry.insert(0, str(self.counties_path))
            except Exception:
                pass
            self.load_shpfile()
        else:
            self.use_online_counties()

        # 5. Overlays and Filters
        if self.overlay_path:
            ov_str = ";".join(self.overlay_path) if isinstance(self.overlay_path, list) else str(self.overlay_path)
            try:
                self.overlay_entry.delete(0, tk.END)
                self.overlay_entry.insert(0, ov_str)
            except Exception:
                pass
            
        if self.filter_path:
            fl_str = ";".join(self.filter_path) if isinstance(self.filter_path, list) else str(self.filter_path)
            try:
                self.filter_entry.delete(0, tk.END)
                self.filter_entry.insert(0, fl_str)
            except Exception:
                pass

        # Trigger loader for overlays/filters
        if self.overlay_path or self.filter_path:
            self.load_overlay()

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
            if not self.griddesc_path or self.griddesc_path == "(NetCDF Attributes)":
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
        grid name is selected. Otherwise,        fallback to WGS84 (EPSG:4326). This ensures geographic coordinate rendering
        when no specific grid context is available.
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
            if self.griddesc_path and self.griddesc_path != "(NetCDF Attributes)" and getattr(self, 'grid_name_var', None):
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

    def _draw_graticule(self, ax, tf_fwd: pyproj.Transformer, tf_inv: pyproj.Transformer, lon_step=None, lat_step=None, with_labels=True):
        """Draw longitude/latitude lines (in degrees) on the given axes using provided transformers.
        The axes must be in the projected (LCC) coordinates. Optionally label lines along edges.
        """


        if ax is None:
            return _draw_graticule_fn(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)

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

        artists = _draw_graticule_fn(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)
        if not isinstance(artists, dict):
            artists = {'lines': [], 'texts': []}
        ax._smk_graticule_artists = artists  # type: ignore[attr-defined]

        def _on_limits(_axis):
            try:
                _remove_existing()
                refreshed = _draw_graticule_fn(ax, tf_fwd, tf_inv, lon_step, lat_step, with_labels)
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



    def _set_status(self, message: str, level: Optional[str] = None) -> None:
        """Update the GUI status bar, collapsing whitespace and capping length."""
        lvl = (level or 'INFO').upper()
        # Ensure logging occurs
        if lvl == 'ERROR': logging.error(message)
        elif lvl == 'WARNING': logging.warning(message)
        else: logging.info(message)
        
        if not USING_TK:
            return

        def _do_update():
            try:
                if self.status_var is None:
                    self.status_var = tk.StringVar(master=self.root, value='')
                
                # Apply premium styling to status bar based on level (Light Mode)
                style = "QStatusBar { background: #f1f3f5; color: #495057; font-family: 'Segoe UI', sans-serif; font-weight: 500; border-top: 1px solid #dee2e6; }"
                if lvl == 'ERROR':
                    style = "QStatusBar { background: #fee2e2; color: #991b1b; font-weight: bold; border-top: 1px solid #fecaca; }"
                elif lvl == 'WARNING':
                    style = "QStatusBar { background: #fef3c7; color: #92400e; font-weight: bold; border-top: 1px solid #fde68a; }"
                elif lvl == 'SUCCESS' or 'success' in message.lower():
                    style = "QStatusBar { background: #dcfce7; color: #166534; font-weight: bold; border-top: 1px solid #bbf7d0; }"
                
                if hasattr(self.root, 'statusBar'):
                    self.root.statusBar().setStyleSheet(style)

                prefix = f"{lvl}: " if lvl else ''
                collapsed = ' '.join(message.strip().split()) if message else ''
                self.status_var.set((prefix + collapsed)[:512])
                
                # Progress Bar Logic
                if hasattr(self, 'progress_bar'):
                    c_low = collapsed.lower()
                    busy_keys = ["loading", "processing", "building", "rendering", "extracting", "preparing", "fetching", "animation"]
                    done_keys = ["ready", "error", "success", "cleared", "loaded", "complete", "finished", "done"]
                    
                    if any(x in c_low for x in busy_keys):
                         self.progress_bar.show()
                         self.progress_bar.setRange(0, 0)
                    elif any(x in c_low for x in done_keys) and not any(x in c_low for x in ["loading", "rendering"]):
                         self.progress_bar.hide()
                         self.progress_bar.setRange(0, 100)
                         self.progress_bar.setValue(0)
                         
                if self.status_label is not None:
                    self.status_label.update_idletasks()
            except Exception:
                pass

        if hasattr(self.root, 'run_on_main_signal'):
             self.root.run_on_main_signal.emit(_do_update)
        else:
             _do_update()

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
        if QThread.currentThread() != QApplication.instance().thread():
             self.root.after(0, lambda: self._loader_notify(level, message))
             return
        lvl = (level or 'INFO').upper()
        msg = ' '.join(str(message).strip().split()) if message else ''
        if not msg:
            return
        self._loader_messages.append((lvl, msg))
        if len(self._loader_messages) > 10:
            self._loader_messages = self._loader_messages[-10:]
        self._set_status(msg, level=lvl)
        
        if hasattr(self, 'progress_bar'):
            if any(x in msg for x in ["Lazy-loading", "Loading", "Processing", "Building"]):
                 self.progress_bar.show()
                 self.progress_bar.setRange(0, 0)
            elif any(x in msg for x in ["Ready", "Error", "Warning", "Success"]):
                 self.progress_bar.hide()

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
        """Harden shutdown: close plots, and force exit."""
        if getattr(self, '_in_close_callback', False):
            return
        self._in_close_callback = True
        try:
            # Close all matplotlib figures to release memory and resources
            plt.close('all')
        except Exception:
            pass
        try:
            # Shutdown Qt loop
            QApplication.quit()
        except Exception:
            pass
        # Force exit to ensure the process terminates immediately, 
        # bypassing any hanging background threads or matplotlib artifacts.
        import os
        os._exit(0)

    # ---- Unified notification / logging helper ----
    def _notify(self, level: str, title: str, message: str, exc: Optional[Exception] = None, *, popup: Optional[bool] = None):
        """Log the message and show a GUI dialog (if Tk available). Levels: INFO, WARNING, ERROR.
        If an exception object is supplied, logs stack trace at ERROR level."""
        lvl = level.upper()
        if popup is None:
            # Default: Show popups for WARNING and ERROR, but NOT for INFO
            popup = (lvl != 'INFO')
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
                # from tkinter import messagebox
                if lvl == 'INFO':
                    messagebox.showinfo(title, message)
                elif lvl == 'WARNING':
                    messagebox.showwarning(title, message)
                elif lvl == 'ERROR':
                    messagebox.showerror(title, message)
            except Exception:
                pass  # Suppress any UI errors after logging

    def _build_layout(self):
        frm = ttk.Frame(self.root, padding=4)
        frm.grid(row=0, column=0, sticky='nsew')
        # keep a reference to the main frame for later configuration
        self.frm = frm
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # SMOKE report + delimiter controls
        ttk.Label(frm, text="SMKREPORT/FF10/NETCDF Input:").grid(row=0, column=0, sticky='w')
        self.emis_entry = ttk.Entry(frm, width=self._w_chars(60))
        self.emis_entry.grid(row=0, column=1, sticky='we')
        self.emis_entry.bind('<Return>', self._on_emissions_entry_change)
        self.emis_entry.bind('<FocusOut>', self._on_emissions_entry_change)
        btn_browse_in = ttk.Button(frm, text="Browse", command=self.browse_inputfile)
        btn_browse_in.grid(row=0, column=2, padx=4)
        if QApplication.instance():
            btn_browse_in.setIcon(QApplication.instance().style().standardIcon(QStyle.SP_DirHomeIcon))

        # Delimiter state (widgets will be placed in button row)
        self.delim_var = tk.StringVar()
        self.custom_delim_var = tk.StringVar()
        # Determine initial delimiter token from provided CLI arg
        def _initial_delim_token(raw: Optional[str]) -> str:
            if not raw:
                return 'auto'
            raw_norm = normalize_delim(raw)
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
        self.county_entry.bind('<Return>', self._on_counties_entry_change)
        self.county_entry.bind('<FocusOut>', self._on_counties_entry_change)
        btn_browse_sh = ttk.Button(frm, text="Browse", command=self.browse_shpfile)
        btn_browse_sh.grid(row=1, column=2, padx=4)
        if QApplication.instance():
             btn_browse_sh.setIcon(QApplication.instance().style().standardIcon(QStyle.SP_FileDialogStart))

        # Online year selector and button
        self.counties_year_var = tk.StringVar(value='2020')
        ttk.OptionMenu(frm, self.counties_year_var, '2020', '2020', '2023', command=lambda *_: self.use_online_counties()).grid(row=1, column=3, sticky='we')

        # Overlay Shapefile (Optional)
        ttk.Label(frm, text="Overlay Shapefile (optional):").grid(row=2, column=0, sticky='w')
        self.overlay_entry = ttk.Entry(frm, width=self._w_chars(60))
        self.overlay_entry.grid(row=2, column=1, sticky='we')
        self.overlay_entry.bind('<Return>', self._on_overlay_entry_change)
        self.overlay_entry.bind('<FocusOut>', self._on_overlay_entry_change)
        btn_browse_ov = ttk.Button(frm, text="Browse", command=self.browse_overlay_shpfile)
        btn_browse_ov.grid(row=2, column=2, padx=4)

        # Filter by Overlay Shapefile
        self.filter_overlay_var = tk.StringVar(value=str(getattr(self, 'initial_filter_overlay', 'False')))
        ttk.Label(frm, text="Filter Operation:").grid(row=3, column=3, sticky='e', padx=(10, 2))
        self.filter_overlay_menu = ttk.OptionMenu(frm, self.filter_overlay_var, self.filter_overlay_var.get(), "False", "clipped", "intersect", "within")
        self.filter_overlay_menu.grid(row=3, column=4, sticky='we')

        # Filter Shapefile (Optional - for spatial filtering)
        ttk.Label(frm, text="Filter Shapefile (optional):").grid(row=3, column=0, sticky='w')
        self.filter_entry = ttk.Entry(frm, width=self._w_chars(60))
        self.filter_entry.grid(row=3, column=1, sticky='we')
        self.filter_entry.bind('\u003cReturn\u003e', self._on_filter_entry_change)
        self.filter_entry.bind('\u003cFocusOut\u003e', self._on_filter_entry_change)
        ttk.Button(frm, text="Browse", command=self.browse_filter_shpfile).grid(row=3, column=2, padx=4)

        # GRIDDESC file
        ttk.Label(frm, text="GRIDDESC File (optional):").grid(row=4, column=0, sticky='w')
        self.griddesc_entry = ttk.Entry(frm, width=self._w_chars(60))
        self.griddesc_entry.grid(row=4, column=1, sticky='we')
        self.griddesc_entry.bind('<Return>', self._on_griddesc_entry_change)
        self.griddesc_entry.bind('<FocusOut>', self._on_griddesc_entry_change)
        ttk.Button(frm, text="Browse", command=self.browse_griddesc).grid(row=4, column=2, padx=4)

        # Grid Name selector
        self.grid_name_var = tk.StringVar()
        self.grid_name_menu = ttk.OptionMenu(frm, self.grid_name_var, "Select Grid", command=lambda *_: self.load_grid_shape())
        self.grid_name_menu.grid(row=4, column=3, sticky='we')

        # Pollutant selector
        ttk.Label(frm, text="Pollutant:").grid(row=5, column=0, sticky='w')
        self.pollutant_var = tk.StringVar()
        self.pollutant_menu = ttk.OptionMenu(frm, self.pollutant_var, None)
        self.pollutant_menu.grid(row=5, column=1, sticky='we')

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
        btn_frame.grid(row=6, column=0, columnspan=5, pady=6, sticky='we')
        # Column stretching: let the bins entry grow (Column 18)
        for c in (18,):
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
            self.load_inputfile(show_preview=False)
        self.custom_delim_entry.bind('<FocusOut>', _reload_event)
        self.custom_delim_entry.bind('<Return>', _reload_event)
        # Zoom to Data next to Delim
        self.zoom_var = tk.BooleanVar(value=True)
        self.zoom_check = ttk.Checkbutton(btn_frame, text='Zoom to Data', variable=self.zoom_var)
        self.zoom_check.grid(row=0, column=5, sticky='w', padx=(12,0))
        
        # Fill NaN=0 Checkbox
        self.fill_zero_check = ttk.Checkbutton(btn_frame, text='Fill NaN=0', variable=self.fill_zero_var)
        self.fill_zero_check.grid(row=0, column=6, sticky='w', padx=(8,0))
        
        # NCF Controls (Layer / TStep) - initially enabled but state set to disabled
        self.ncf_layer_var = tk.StringVar(value='')
        self.ncf_tstep_var = tk.StringVar(value='')
        
        self.ncf_layer_label = ttk.Label(btn_frame, text='Lay:')
        self.ncf_layer_menu = ttk.Combobox(btn_frame, textvariable=self.ncf_layer_var, state='disabled', width=8)
        self.ncf_tstep_label = ttk.Label(btn_frame, text='Time:')
        self.ncf_tstep_menu = ttk.Combobox(btn_frame, textvariable=self.ncf_tstep_var, state='disabled', width=26)
        
        # Bind events
        def _on_ncf_param_change(event):
            if not self.inputfile_path: return
            # Only trigger if NCF and valid selections
            if not (self.ncf_layer_menu.get() and self.ncf_tstep_menu.get()): return
            self.load_inputfile(show_preview=False)

        self.ncf_layer_menu.bind('<<ComboboxSelected>>', _on_ncf_param_change)
        self.ncf_tstep_menu.bind('<<ComboboxSelected>>', _on_ncf_param_change)
        
        # Initially grid them (they will be disabled)
        self.ncf_layer_label.grid(row=0, column=7, sticky='e', padx=(12,2))
        self.ncf_layer_menu.grid(row=0, column=8, sticky='w')
        self.ncf_tstep_label.grid(row=0, column=9, sticky='e', padx=(12,2))
        self.ncf_tstep_menu.grid(row=0, column=10, sticky='w')

        # Row 1: Plotting Config (Scale, Plot By, Proj, Bins, Colormap)
        self.scale_label = ttk.Label(btn_frame, text="Scale:")
        self.scale_label.grid(row=1, column=0, sticky='e', padx=(4,2))
        self.scale_var = tk.StringVar(value='linear')
        self.scale_menu_widget = ttk.OptionMenu(btn_frame, self.scale_var, 'linear', 'linear', 'log')
        self.scale_menu_widget.grid(row=1, column=1, sticky='w')
        
        self.plotby_label = ttk.Label(btn_frame, text='By:')
        self.plotby_label.grid(row=1, column=2, sticky='e', padx=(12,2))
        self.plot_by_var = tk.StringVar(value='auto')
        self.plotby_menu_widget = ttk.OptionMenu(btn_frame, self.plot_by_var, 'auto', 'auto', 'county', 'grid')
        self.plotby_menu_widget.grid(row=1, column=3, sticky='w')
        
        self.proj_label = ttk.Label(btn_frame, text='Proj:')
        self.proj_label.grid(row=1, column=4, sticky='e', padx=(12,2))
        self.projection_var = tk.StringVar(value='lcc')
        self.proj_menu_widget = ttk.OptionMenu(btn_frame, self.projection_var, 'lcc', 'auto', 'wgs84', 'lcc')
        self.proj_menu_widget.grid(row=1, column=5, sticky='w')
        
        self.bins_label = ttk.Label(btn_frame, text='Bins:')
        self.bins_label.grid(row=1, column=6, sticky='e', padx=(12,2))
        self.bins_entry = ttk.Entry(btn_frame, width=self._w_chars(28, min_chars=18, max_chars=40), textvariable=self.class_bins_var)
        self.bins_entry.grid(row=1, column=7, sticky='we')
        
        self.cmap_label = ttk.Label(btn_frame, text='Map:')
        self.cmap_label.grid(row=1, column=8, sticky='e', padx=(12,2))
        self.cmap_menu_widget = ttk.OptionMenu(btn_frame, self.cmap_var, self.cmap_var.get(), *self._cmap_choices)
        self.cmap_menu_widget.grid(row=1, column=9, sticky='w')

        # Row 2 (Optional/Context): SCC
        self.scc_select_var = tk.StringVar(value='All SCC')
        self.scc_label = ttk.Label(btn_frame, text='SCC Filter:')
        self.scc_entry = ttk.Combobox(btn_frame, textvariable=self.scc_select_var, values=[], width=self._w_chars(60, min_chars=30, max_chars=100), state='disabled')
        self.scc_label.grid(row=2, column=0, sticky='e', padx=(4,2), pady=(4,0))
        self.scc_entry.grid(row=2, column=1, columnspan=7, sticky='we', pady=(4,0))
        
        try:
            btn_frame.columnconfigure(7, weight=1)
            self.scc_entry.state(['disabled'])
            self.scc_label.state(['disabled'])
        except Exception: pass

        # The 3-row layout is now stable; disable dynamic reflow to prevent layout collapse
        self._btn_frame = btn_frame
        self._layout_mode = 'static'
        # try:
        #     self.root.bind('<Configure>', self._on_resize)
        # except Exception:
        #     pass

        # Text preview widget (persistent)
        self.preview = tk.Text(frm, height=6, width=90)
        self.preview.grid(row=7, column=0, columnspan=5, pady=2, sticky='nsew')

        # Embedded plot frame (row 8) - map will display here
        self.plot_frame = ttk.Frame(frm)
        self.plot_frame.grid(row=8, column=0, columnspan=5, sticky='nsew', pady=(2, 0))
        # Ensure plot area has a semi-reasonable minimum vertical expansion
        if hasattr(self.plot_frame, 'setMinimumHeight'):
            self.plot_frame.setMinimumHeight(400)

        if USING_TK:
            try:
                if self.status_var is None:
                    self.status_var = tk.StringVar(master=self.root, value='Ready.')
            except Exception:
                self.status_var = None
            if self.status_var is not None:
                # Modern Status Bar with Progress
                status_frame = ttk.Frame(frm)
                status_frame.grid(row=9, column=0, columnspan=5, sticky='we', pady=(4, 0))
                
                self.status_label = ttk.Label(status_frame, textvariable=self.status_var, relief='sunken', anchor='w')
                self.status_label.grid(row=0, column=0, sticky='we')
                
                self.progress_bar = QProgressBar()
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(0)
                self.progress_bar.hide()
                status_frame.layout().addWidget(self.progress_bar, 0, 1)
                
                status_frame.columnconfigure(0, weight=1)
                status_frame.columnconfigure(1, weight=0)

        # Configure stretch rows/columns for the main frame
        try:
            frm.rowconfigure(7, weight=1)   # text preview
            frm.rowconfigure(8, weight=3)   # plot area larger
            frm.rowconfigure(9, weight=0)
            frm.columnconfigure(1, weight=1)
        except Exception:
            pass

    def _configure_button_layout(self, mode: str):
        # Adjust grid positions depending on mode
        if mode == 'wide':
            try:
                self._btn_frame.columnconfigure(18, weight=1) # Bins entry at col 18
                for c in (1, 22):
                    self._btn_frame.columnconfigure(c, weight=0)
            except Exception:
                pass
            # Put all on row 0
            # Note: NCF widgets (7-10), Scale (11-12), PlotBy (13-14) are typically roughly static or managed by init
            # We explicitly place the ones that move between rows 0 and 1
            widgets_map = [
                (self.ncf_layer_label, 7), (self.ncf_layer_menu, 8),
                (self.ncf_tstep_label, 9), (self.ncf_tstep_menu, 10),
                (self.scale_label, 11), (self.scale_menu_widget, 12),
                (self.plotby_label, 13), (self.plotby_menu_widget, 14),
                (self.proj_label, 15), (self.proj_menu_widget, 16),
                (self.bins_label, 17), (self.bins_entry, 18), 
                (self.cmap_label, 19), (self.cmap_menu_widget, 20),
                (self.scc_label, 21), (self.scc_entry, 22)
            ]
            for widget, col in widgets_map:
                try:
                    sticky_val = 'we' if widget in (self.bins_entry, self.scc_entry) else 'w'
                    if widget in (self.proj_label, self.bins_label, self.cmap_label, self.scc_label, self.scale_label, self.plotby_label, self.ncf_layer_label, self.ncf_tstep_label):
                        sticky_val = 'e'
                    widget.grid_configure(row=0, column=col, sticky=sticky_val)
                except Exception:
                    pass
        else:  # compact
            try:
                # Make bins entry and scc entry stretch on compact row
                self._btn_frame.columnconfigure(3, weight=1)
                # Reset weights for columns used in wide mode
                for c in (18, 22):
                    self._btn_frame.columnconfigure(c, weight=0)
            except Exception:
                pass
            # Move Bins/Colormap/SCC/Proj to row 1
            # Leave Scale and PlotBy on Row 0? 
            # Row 0: Plot, Preview, Delim, Custom, Zoom, FillZero, Scale, PlotBy
            # Cols: 0, 1, 2-3, 4, 5, 6, 7-8, 9-10 (shifted indices)
            
            # Reposition items that stay on Row 0 but need shifting in compact mode?
            # Actually, standardizing: keep NCF on row 1 in compact mode for space?
            
            mapping_row1 = [
                (self.ncf_layer_label, 0), (self.ncf_layer_menu, 1), 
                (self.ncf_tstep_label, 2), (self.ncf_tstep_menu, 3),
                (self.proj_label, 4), (self.proj_menu_widget, 5),
                (self.bins_label, 6), (self.bins_entry, 7),
                (self.cmap_label, 8), (self.cmap_menu_widget, 9),
                (self.scc_label, 10), (self.scc_entry, 11),
            ]
            for widget, col in mapping_row1:
                try:
                    # Allow both entry widgets to stretch horizontally
                    sticky_val = 'we' if widget in (self.bins_entry, self.scc_entry) else 'w'
                    if widget in (self.proj_label, self.bins_label, self.cmap_label, self.scc_label, self.ncf_layer_label, self.ncf_tstep_label):
                        sticky_val = 'e'
                    widget.grid_configure(row=1, column=col, sticky=sticky_val)
                except Exception:
                    pass
                    
            # Ensure Scale/PlotBy are on Row 0
            # Scale (11,12 in wide), PlotBy (13,14 in wide) -> might be 7,8 and 9,10 in compact row 0?
            # Let's just put them after FillZero (col 6)
            row0_extra = [
                (self.scale_label, 7), (self.scale_menu_widget, 8),
                (self.plotby_label, 9), (self.plotby_menu_widget, 10)
            ]
            for widget, col in row0_extra:
                try:
                    widget.grid_configure(row=0, column=col, sticky='w' if 'label' not in str(widget) else 'e')
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
            cand = self.inputfile_path
            if cand and os.path.exists(cand):
                init = os.path.dirname(cand)
                if os.path.isdir(init):
                    init_dir = init
        except Exception:
            init_dir = None
        if init_dir is None:
            init_dir = DEFAULT_INPUTS_INITIALDIR if os.path.isdir(DEFAULT_INPUTS_INITIALDIR) else None
        paths = filedialog.askopenfilenames(initialdir=init_dir, filetypes=[("CSV/List/NetCDF", "*.csv *.txt *.lst *.ncf *.nc"), ("All", "*.*")])
        if not paths:
            return
        
        # Join with semicolon for display
        joined_path = "; ".join(paths)
        self.emis_entry.delete(0, tk.END)
        self.emis_entry.insert(0, joined_path)
        self.inputfile_path = joined_path
        self.load_inputfile(show_preview=False)

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

    def browse_overlay_shpfile(self):
        init_dir = DEFAULT_SHPFILE_INITIALDIR if os.path.isdir(DEFAULT_SHPFILE_INITIALDIR) else None
        path = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("Shapefile / Geopackage / Zip", "*.shp *.gpkg *.zip"), ("All", "*.*")])
        if not path:
            return
        self.overlay_entry.delete(0, tk.END)
        self.overlay_entry.insert(0, path)
        self.overlay_path = path
        self.load_overlay()

    def use_online_counties(self):
        """Set and load the default online US counties shapefile. User selection can override later."""
        try:
            year = self.counties_year_var.get()
        except Exception:
            year = '2020'
        if year:
            url = f"https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_county_500k.zip"        
        else:
            url = DEFAULT_ONLINE_COUNTIES_URL
        self.counties_path = url
        try:
            self.county_entry.delete(0, tk.END)
            self.county_entry.insert(0, self.counties_path)
        except Exception:
            pass
        self.load_shpfile()

    def _on_emissions_entry_change(self, event=None):
        path = self.emis_entry.get().strip()
        if path != getattr(self, 'inputfile_path', None):
            self.inputfile_path = path
            self.load_inputfile(show_preview=False)

    def _on_counties_entry_change(self, event=None):
        path = self.county_entry.get().strip()
        if path != getattr(self, 'counties_path', None):
            self.counties_path = path
            self.load_shpfile()

    def _on_overlay_entry_change(self, event=None):
        path = self.overlay_entry.get().strip()
        if path != getattr(self, 'overlay_path', None):
            self.overlay_path = path
            self.load_overlay()

    def browse_filter_shpfile(self):
        init_dir = DEFAULT_SHPFILE_INITIALDIR if os.path.isdir(DEFAULT_SHPFILE_INITIALDIR) else None
        path = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("Shapefile / Geopackage / Zip", "*.shp *.gpkg *.zip *.geojson *.json"), ("All", "*.*")])
        if not path:
            return
        self.filter_entry.delete(0, tk.END)
        self.filter_entry.insert(0, path)
        self.filter_path = path
        self.load_filter()

    def _on_filter_entry_change(self, event=None):
        path = self.filter_entry.get().strip()
        if path != getattr(self, 'filter_path', None):
            self.filter_path = path
            self.load_filter()

    def _on_griddesc_entry_change(self, event=None):
        path = self.griddesc_entry.get().strip()
        if path != getattr(self, 'griddesc_path', None):
            self.griddesc_path = path
            self.load_griddesc()

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
                self.load_inputfile(show_preview=False)

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
        
        
        if not self._json_arguments:
            return

        

    def _finalize_loaded_emissions(self, *, show_preview: bool, source_label: Optional[str] = None, scc_data=None) -> None:
        logging.info("DEBUG: Inside _finalize_loaded_emissions")
        if not isinstance(self.emissions_df, pd.DataFrame):
            self._notify('ERROR', 'Invalid Data', 'Emissions dataset is not a DataFrame; cannot continue.')
            return

        self._invalidate_merge_cache()

        try:
            self._source_type = getattr(self.emissions_df, 'attrs', {}).get('source_type')
        except Exception:
            self._source_type = None
        self._ff10_ready = bool(self._source_type == 'ff10_point')

        # No-op: Path persistence disabled

        if self._ff10_ready and self.grid_gdf is not None:
            self._ensure_ff10_grid_mapping()

        logging.info("DEBUG: Updating SCC widgets")
        self._update_scc_widgets(scc_data=scc_data)

        logging.info("DEBUG: Detecting pollutants")
        self.pollutants = detect_pollutants(self.emissions_df)
        if not self.pollutants and isinstance(self.emissions_df, pd.DataFrame):
            self.pollutants = self.emissions_df.attrs.get('available_pollutants', [])
        try:
            self.units_map = dict(self.emissions_df.attrs.get('units_map', {}))
        except Exception:
            self.units_map = {}
        if not self.pollutants:
            self._notify('WARNING', 'No Pollutants', 'No pollutant columns detected.')
            return

        # Sort pollutants alphabetically for easier navigation
        self.pollutants = sorted(self.pollutants, key=lambda s: s.lower())

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
        # Sync path from entry widget if changed manually
        try:
            val = self.emis_entry.get().strip()
            if val:
                self.inputfile_path = val
        except Exception:
            pass

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

        # Capture NCF dropdown state to preserve selection across reload
        ncf_preserve = {}
        try:
            if hasattr(self, 'ncf_layer_var'):
                ncf_preserve['layer'] = self.ncf_layer_var.get()
            if hasattr(self, 'ncf_tstep_var'):
                ncf_preserve['tstep'] = self.ncf_tstep_var.get()
        except Exception:
            pass

        self._set_status("Loading data...", level="INFO")

        # Reset UI elements in main thread before starting background loader
        try:
            if hasattr(self, 'ncf_layer_menu'):
                self.ncf_layer_menu.set('')
                self.ncf_tstep_menu.set('')
                self.ncf_layer_menu['values'] = []
                self.ncf_tstep_menu['values'] = []
                self.ncf_layer_menu.state(['disabled'])
                self.ncf_tstep_menu.state(['disabled'])
            if not self.inputfile_path:
                self.pollutant_menu['menu'].delete(0, 'end')
        except Exception:
            pass
        
        threading.Thread(
            target=self._load_inputfile_worker, 
            args=(show_preview, effective_delim, current_delim_state, ncf_preserve), 
            daemon=True
        ).start()

    def _load_inputfile_worker(self, show_preview, effective_delim, current_delim_state, ncf_preserve):
        logging.info("DEBUG: Worker Thread Started")
        if not self.inputfile_path:
            self.emissions_df = None
            self.raw_df = None
            self.pollutants = []
            self.units_map = {}
            self._invalidate_merge_cache()
            self.status_var.set("Emissions data cleared.")
            return

        try:
            if True: # Unified loading path (reimport deprecated)
                try:
                    # Thread-safe notify wrapper
                    def safe_notify(level, message):
                        logging.info(f"DEBUG: safe_notify called: {message}")
                        if hasattr(self.root, 'notify_signal'):
                            self.root.notify_signal.emit(level, message)
                        else:
                            self.root.after(0, lambda: self._loader_notify(level, message))
                    
                    # Handle multiple files (semicolon separated)
                    f_input = self.inputfile_path
                    if f_input and ";" in f_input:
                        f_input = [x.strip() for x in f_input.split(";") if x.strip()]

                    # Prepare NetCDF Params
                    ncf_params = {}
                    # Time Dimension
                    t_arg = getattr(self, 'ncf_tdim', 'avg')
                    if str(t_arg).isdigit():
                        ncf_params['tstep_idx'] = int(t_arg)
                        ncf_params['tstep_op'] = 'select'
                    else:
                        op = str(t_arg).lower()
                        if op in ('avg', 'mean', 'average'): ncf_params['tstep_op'] = 'mean'
                        elif op in ('sum', 'total'): ncf_params['tstep_op'] = 'sum'
                        elif op in ('max', 'maximum'): ncf_params['tstep_op'] = 'max'
                        elif op in ('min', 'minimum'): ncf_params['tstep_op'] = 'min'
                        else: ncf_params['tstep_op'] = 'mean'

                    # Layer Dimension
                    z_arg = getattr(self, 'ncf_zdim', '0')
                    if str(z_arg).isdigit():
                        ncf_params['layer_idx'] = int(z_arg)
                        ncf_params['layer_op'] = 'select'
                    else:
                        op = str(z_arg).lower()
                        ncf_params['layer_idx'] = None
                        if op in ('avg', 'mean', 'average'): ncf_params['layer_op'] = 'mean'
                        elif op in ('sum', 'total'): ncf_params['layer_op'] = 'sum'
                        elif op in ('max', 'maximum'): ncf_params['layer_op'] = 'max'
                        elif op in ('min', 'minimum'): ncf_params['layer_op'] = 'min'
                        else: 
                            ncf_params['layer_idx'] = 0
                            ncf_params['layer_op'] = 'select'

                    emissions_df, raw_df = read_inputfile(
                        f_input,
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
                        ncf_params=ncf_params,
                        lazy=True
                    )
                    logging.info("DEBUG: read_inputfile returned.")
                    if isinstance(emissions_df, pd.DataFrame):
                        emissions_df.attrs['ncf_params'] = ncf_params

                    # Build FIPS to ensure compatibility and consistent attributes
                    if isinstance(emissions_df, pd.DataFrame):
                        logging.info("DEBUG: Building FIPS...")
                        try:
                            is_ncf = isinstance(f_input, str) and is_netcdf_file(f_input)
                            emissions_df = get_emis_fips(emissions_df, verbose=(not is_ncf))
                        except ValueError:
                            # Ignore if FIPS columns not found (e.g. gridded data without region_cd)
                            pass
                        except Exception as e:
                            safe_notify('WARNING', f"Error building FIPS: {e}")
                    
                    # Apply units_map from JSON arguments if available (overrides input data)
                    units_map_arg = self._json_arguments.get('units_map')
                    if isinstance(units_map_arg, dict) and isinstance(emissions_df, pd.DataFrame):
                        existing_map = emissions_df.attrs.get('units_map', {})
                        if not isinstance(existing_map, dict): existing_map = {}
                        for k, v in units_map_arg.items():
                            existing_map[k] = v
                        emissions_df.attrs['units_map'] = existing_map

                    # NCF Auto-Grid Logic
                    if isinstance(f_input, str) and is_netcdf_file(f_input):
                        try:
                            # Lazy import
                            from ncf_processing import create_ncf_domain_gdf, read_ncf_grid_params, get_ncf_dims, read_ncf_emissions
                            safe_notify('INFO', 'Detected NetCDF input. Auto-configuring grid geometry...')
                            
                            # Get dimensions and update UI
                            dims_info = get_ncf_dims(f_input) # {'n_tsteps': X, 'n_layers': Y, ...}
                            n_lay = dims_info.get('n_layers', 1)
                            n_ts = dims_info.get('n_tsteps', 1)
                            tflags = dims_info.get('tflag_values', [])
                            
                            def _update_ncf_ui():
                                # Enable widgets (must clear disabled state first)
                                self.ncf_layer_menu.state(['!disabled', 'readonly'])
                                self.ncf_tstep_menu.state(['!disabled', 'readonly'])
                                
                                # Populate values
                                lay_vals = ['Sum All Layers', 'Average All Layers'] + [f"Layer {i+1}" for i in range(n_lay)]
                                self.ncf_layer_menu['values'] = lay_vals
                                
                                # Attempt to restore preserved selection or default to first
                                target_lay = ncf_preserve.get('layer')
                                if target_lay and target_lay in lay_vals:
                                    self.ncf_layer_menu.set(target_lay)
                                elif self.ncf_layer_var.get() not in lay_vals:
                                    # Default to Layer 1 for backward compatibility/preference? Or Sum?
                                    # Let's keep Layer 1 as default as it's often surface
                                    if n_lay > 0:
                                        self.ncf_layer_menu.set(lay_vals[2]) # 0 is Sum, 1 is Avg, 2 is Layer 1
                                
                                # Use TFLAG values if available
                                if tflags and len(tflags) == n_ts:
                                    ts_vals = ['Sum All Time', 'Average All Time'] + [f"{t}" for i, t in enumerate(tflags)]
                                else:
                                    ts_vals = ['Sum All Time', 'Average All Time'] + [f"{i+1}" for i in range(n_ts)]
                                
                                self.ncf_tstep_menu['values'] = ts_vals

                                # Attempt to restore preserved selection or default
                                target_ts = ncf_preserve.get('tstep')
                                if target_ts and target_ts in ts_vals:
                                    self.ncf_tstep_menu.set(target_ts)
                                elif self.ncf_tstep_var.get() not in ts_vals:
                                    self.ncf_tstep_menu.set(ts_vals[1])

                            if hasattr(self.root, 'run_on_main_signal'):
                                self.root.run_on_main_signal.emit(_update_ncf_ui)
                            else:
                                self.root.after(0, _update_ncf_ui)

                            # Determine read params - use preserved values as they reflect user intent (which was cleared from var)
                            curr_lay_str = ncf_preserve.get('layer') or ''
                            curr_ts_str = ncf_preserve.get('tstep') or ''
                            
                            l_idx = 0
                            l_op = 'select'
                            if 'Sum All' in curr_lay_str:
                                l_idx = None
                                l_op = 'sum'
                            elif 'Average' in curr_lay_str:
                                l_idx = None
                                l_op = 'mean'
                            elif 'Layer' in curr_lay_str:
                                try: l_idx = int(curr_lay_str.split()[-1]) - 1
                                except: l_idx = 0
                                l_op = 'select'
                            
                            ts_idx = None
                            ts_op = 'mean'
                            if 'Sum All' in curr_ts_str:
                                ts_idx = None
                                ts_op = 'sum'
                            elif 'Average' in curr_ts_str:
                                ts_idx = None
                                ts_op = 'mean'
                            elif curr_ts_str:
                                try:
                                    # Re-construct expected list to find index
                                    # n_ts and tflags are available here
                                    flag_list = []
                                    if tflags and len(tflags) == n_ts:
                                        flag_list = [f"{t}" for t in tflags]
                                    else:
                                        flag_list = [f"{i+1}" for i in range(n_ts)]
                                     
                                    if curr_ts_str in flag_list:
                                        ts_idx = flag_list.index(curr_ts_str)
                                        ts_op = 'select'
                                    else:
                                        # Fallback for old format "Time X ..."
                                        if 'Time' in curr_ts_str:
                                                token = curr_ts_str.replace('Time', '', 1).strip().split()[0]
                                                ts_idx = int(token) - 1
                                                ts_op = 'select'
                                except: 
                                        ts_idx = None
                                        ts_op = 'sum'
                            
                            # Re-read emissions with specific params
                            # Note: read_inputfile called read_ncf_emissions default inside data_processing.
                            # We update to use lazy skeleton logic by default
                            msg = f"Reading NetCDF (Layer: {curr_lay_str or 'Layer 1'}, Time: {curr_ts_str or 'Average'})..."
                            safe_notify('INFO', msg)
                            # Pass existing xr_ds if available
                            old_ds = getattr(emissions_df, 'attrs', {}).get('_smk_xr_ds')
                            emissions_df = read_ncf_emissions(f_input, layer_idx=l_idx, tstep_idx=ts_idx, layer_op=l_op, tstep_op=ts_op, xr_ds=old_ds, lazy=True)
                            # Record current params for future lazy fetches
                            emissions_df.attrs['ncf_params'] = {
                                'layer_idx': l_idx, 'tstep_idx': ts_idx, 'layer_op': l_op, 'tstep_op': ts_op
                            }
                            # Update raw_df too for consistency
                            raw_df = emissions_df
                            
                            # ... (existing grid logic) ...
                            ncf_grid_gdf = create_ncf_domain_gdf(f_input, full_grid=True)
                            if not ncf_grid_gdf.empty:
                                def _set_ncf_grid(gdf):
                                    self.grid_gdf = gdf
                                    self.plot_by_var.set('grid')
                                    # Update UI labels to reflect NCF source
                                    if self.griddesc_entry:
                                        self.griddesc_entry.delete(0, tk.END)
                                        self.griddesc_entry.insert(0, "(NetCDF Attributes)")
                                    # Extract GRID name for UI
                                    _, gp = read_ncf_grid_params(f_input)
                                    if self.grid_name_var:
                                        self.grid_name_var.set(gp[0])
                                    
                                    self.status_var.set("Grid geometry loaded from NetCDF.")
                                
                                if hasattr(self.root, 'run_on_main_signal'):
                                    self.root.run_on_main_signal.emit(lambda g=ncf_grid_gdf: _set_ncf_grid(g))
                                else:
                                    self.root.after(0, lambda g=ncf_grid_gdf: _set_ncf_grid(g))
                        except Exception as grid_err:
                            safe_notify('WARNING', f"Failed to auto-configure grid from NetCDF: {grid_err}")

                except Exception as e:
                    err_msg = str(e)
                    if "No valid FIPS code columns found" in err_msg:
                        if hasattr(self.root, 'run_on_main_signal'):
                             self.root.run_on_main_signal.emit(lambda: self._notify('ERROR', 'Input Not Supported', 'The input file format is not supported (missing FIPS/Region columns).', exc=None))
                        else:
                             self.root.after(0, lambda: self._notify('ERROR', 'Input Not Supported', 'The input file format is not supported (missing FIPS/Region columns).', exc=None))
                    else:
                        if hasattr(self.root, 'run_on_main_signal'):
                             self.root.run_on_main_signal.emit(lambda e=e: self._notify('ERROR', 'Emissions Load Error', str(e), exc=e))
                        else:
                             self.root.after(0, lambda e=e: self._notify('ERROR', 'Emissions Load Error', str(e), exc=e))
                    return
                
                # Apply fill-nan if configured
                fn_arg = getattr(self, 'fill_nan_arg', None)
                if fn_arg is not None:
                    try:
                        should_fill = False
                        fill_num = 0.0
                        if isinstance(fn_arg, bool) and not fn_arg:
                            should_fill = False
                        elif isinstance(fn_arg, str) and fn_arg.lower() == 'false':
                            should_fill = False
                        else:
                            try:
                                fill_num = float(fn_arg)
                                should_fill = True
                            except Exception:
                                pass
                        
                        if should_fill:
                            # Fill emissions_df
                            cols = emissions_df.select_dtypes(include=[np.number]).columns
                            if len(cols) > 0:
                                emissions_df[cols] = emissions_df[cols].fillna(fill_num)
                            # Fill raw_df
                            rcols = raw_df.select_dtypes(include=[np.number]).columns
                            if len(rcols) > 0:
                                raw_df[rcols] = raw_df[rcols].fillna(fill_num)
                    except Exception as e:
                        logging.warning(f"Fill NaN error: {e}")

                self.emissions_df = emissions_df
                self.raw_df = raw_df
                if effective_delim is not None:
                    self.emissions_delim = effective_delim
                self._last_loaded_delim_state = current_delim_state
                
                # Pre-compute SCC data in thread
                logging.info("DEBUG: Computing SCC data...")
                scc_data = self._compute_scc_data(self.raw_df)
                
                logging.info("DEBUG: Finalizing loaded emissions...")
                if hasattr(self.root, 'finalize_signal'):
                     self.root.finalize_signal.emit(show_preview, None, scc_data)
                else:
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
        if not self.griddesc_path or self.griddesc_path == "(NetCDF Attributes)":
            self.grid_gdf = None
            try:
                menu = self.grid_name_menu["menu"]
                menu.delete(0, "end")
                self.grid_name_var.set("Select Grid")
            except Exception:
                pass
            self._invalidate_merge_cache()
            return
        self._ff10_grid_ready = False
        try:
            grid_names = sorted(extract_grid(self.griddesc_path, grid_id=None), key=lambda s: s.lower())
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
        if not self.griddesc_path or not self.grid_name_var.get() or self.grid_name_var.get() == "Select Grid":
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
            # Path persistence disabled
            self._invalidate_merge_cache()
        except Exception as e:
            self._notify('ERROR', 'Counties Load Error', str(e), exc=e)
            return

    def load_overlay(self):
        """Load optional overlay shapefile(s) for visual reference. Also includes filter shapefiles."""
        try:
            # 1. Clear existing visual overlay list
            self.overlay_gdf = None
            
            # 2. Get paths for both overlays and filters
            ov_paths = getattr(self, 'overlay_path', None)
            filter_paths = getattr(self, 'filter_path', None)
            
            all_parts = []
            
            # Helper to load paths into parts list
            def _load_into(paths, parts_list):
                if not paths: return
                file_list = []
                if isinstance(paths, str):
                    file_list = [p.strip() for p in paths.split(';') if p.strip()]
                elif isinstance(paths, (list, tuple)):
                    for item in paths:
                        if isinstance(item, str):
                            file_list.extend([p.strip() for p in item.split(';') if p.strip()])
                        else:
                            file_list.append(str(item))
                else:
                    file_list = [str(paths)]
                
                for fpath in file_list:
                    try:
                        part = read_shpfile(fpath, False)
                        if part is not None and not part.empty:
                            parts_list.append(part)
                    except Exception:
                        self._notify("WARNING", "Load Warning", f"Failed to load shapefile: {fpath}")

            # Load primary overlays
            _load_into(ov_paths, all_parts)
            
            # Load filters (for visual display)
            filter_parts = []
            _load_into(filter_paths, filter_parts)
            
            # Update filter_gdf for spatial filtering logic
            self._update_filter_gdf(filter_parts)
            
            # Combine all for visual display
            all_parts.extend(filter_parts)
            
            if all_parts:
                self.overlay_gdf = all_parts
                  
        except Exception as e:
            self._notify('ERROR', 'Overlay Load Error', str(e), exc=e)

    def _update_filter_gdf(self, loaded_filter_parts):
        """Update the internal filter_gdf used for spatial filtering."""
        if not loaded_filter_parts:
            self.filter_gdf = None
        elif len(loaded_filter_parts) == 1:
            self.filter_gdf = loaded_filter_parts[0]
        else:
            target_crs = loaded_filter_parts[0].crs
            combined_parts = [loaded_filter_parts[0]]
            for other in loaded_filter_parts[1:]:
                if target_crs and other.crs and other.crs != target_crs:
                    other = other.to_crs(target_crs)
                combined_parts.append(other)
            self.filter_gdf = pd.concat(combined_parts, ignore_index=True)

    def load_filter(self):
        """Update filter data and refresh overlays."""
        self.load_overlay()


    def _merged(self, plot_by_mode=None, scc_selection=None, scc_code_map=None, notify=None, pollutant=None) -> Optional[gpd.GeoDataFrame]:
        if self.emissions_df is None:
            return None

        def _do_notify(level, title, msg, exc=None, **kwargs):
            if notify:
                notify(level, title, msg, exc, **kwargs)
            else:
                self._notify(level, title, msg, exc=exc, **kwargs)

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
                raise ValueError("Handled")
            self._ensure_ff10_grid_mapping(notify_success=False)
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
                raise ValueError("Handled")

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
                    raise ValueError("Handled")

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
            pollutant or '',
            getattr(self, 'fill_zero_var', None) and self.fill_zero_var.get(),
            getattr(self, 'fill_nan_arg', None),
            # Filter overlay state
            getattr(self, 'filter_overlay_var', None) and self.filter_overlay_var.get(),
            id(self.overlay_gdf) if getattr(self, 'overlay_gdf', None) is not None else 0
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

        emis_for_merge = self.emissions_df if isinstance(self.emissions_df, pd.DataFrame) else None
        if emis_for_merge is None:
            _do_notify('ERROR', 'No Emissions Data', 'Emissions dataset is not loaded or invalid.')
            return None

        # ON-DEMAND LAZY NETCDF FETCH:
        # If the requested pollutant is in the NetCDF variables but not in the DataFrame yet
        target_pol = pollutant if pollutant else (self.pollutant_var.get() if hasattr(self, 'pollutant_var') else None)
        if target_pol and target_pol not in emis_for_merge.columns:
                ds = emis_for_merge.attrs.get('_smk_xr_ds')
                if ds is not None:
                    _do_notify('INFO', 'Fetching Data', f"Lazy-extracting {target_pol} from NetCDF dataset...")
                    try:
                        from ncf_processing import read_ncf_emissions
                        ncf_params = emis_for_merge.attrs.get('ncf_params', {})
                        # Re-read just this pollutant using the cached xarray handle
                        new_data = read_ncf_emissions(
                            self.inputfile_path, 
                            pollutants=[target_pol], 
                            xr_ds=ds, 
                            **ncf_params
                        )
                        if target_pol in new_data.columns:
                            # Cache it in the main dataframe so we don't fetch it again
                            emis_for_merge[target_pol] = new_data[target_pol].values
                            # Also update units_map if missing
                            if target_pol not in self.units_map:
                                v_meta = new_data.attrs.get('variable_metadata', {}).get(target_pol, {})
                                self.units_map[target_pol] = v_meta.get('units', '')
                    except Exception as e:
                        _do_notify('WARNING', 'Fetch Failed', f"Could not lazy-load {target_pol}: {e}")

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
                # Enhanced Warning Diagnostics
                msg_lines = ["Plot by Grid requires 'GRID_RC' column (implied X/Y logic)."]
                
                # Check for presence of row/col columns
                cols_low = [c.lower() for c in emis_for_merge.columns]
                has_row = any(x in cols_low for x in ['row', 'r', 'y', 'ycell', 'y_cell'])
                has_col = any(x in cols_low for x in ['col', 'c', 'x', 'xcell', 'x_cell'])
                
                if has_row and has_col:
                    msg_lines.append("Found potential Row/Col columns, but GRID_RC was not generated.")
                    msg_lines.append("Check that your GRIDDESC file is loaded and the correct Grid Name is selected.")
                else:
                    msg_lines.append("Input data appears to be missing 'row'/'col' or 'x'/'y' identifier columns.")
                
                if self.grid_gdf is None or self.grid_gdf.empty:
                    msg_lines.append("CRITICAL: No Grid Definition loaded. You must load a GRIDDESC file.")

                _do_notify('WARNING', 'No GRID_RC in data', "\n".join(msg_lines))
            else:
                if source_type in ('ff10_point', 'ff10_nonpoint'):
                    _do_notify('WARNING', 'No region_cd in data', 'FF10 data requires region_cd for county plots.')
                else:
                    _do_notify('WARNING', 'No FIPS in data', 'Plot by County requires emissions with FIPS codes.')
            
            # Signal to caller that warning is handled to avoid generic "Missing Data" msg
            raise ValueError("Handled")

        try:
            merged, prepared_emis = merge_emissions_with_geometry(
                emis_for_merge,
                base_gdf,
                merge_on,
                sort=False,
                copy_geometry=False,
            )
            # Propagate attributes from emissions and geometry
            if hasattr(merged, 'attrs'):
                for src in [emis_for_merge, base_gdf]:
                    if hasattr(src, 'attrs'):
                        merged.attrs.update(src.attrs)
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
        
        # Apply fill_nan if configured (for map regions with no data)
        try:
            fill_arg = getattr(self, 'fill_nan_arg', None)
            fill_zero = getattr(self, 'fill_zero_var', None) and self.fill_zero_var.get()

            do_fill = False
            fill_val = 0.0

            if fill_zero:
                do_fill = True
                fill_val = 0.0
            elif fill_arg is not None:
                if isinstance(fill_arg, bool) and not fill_arg:
                    do_fill = False
                elif isinstance(fill_arg, str) and str(fill_arg).lower() == 'false':
                    do_fill = False
                else:
                    try:
                        fill_val = float(fill_arg)
                        do_fill = True
                    except Exception:
                        pass
                
            if do_fill:
                # Identify pollutant columns to fill
                pols = getattr(self, 'pollutants', [])
                if pols:
                    cols_to_fill = [c for c in pols if c in merged.columns]
                    if cols_to_fill:
                            merged[cols_to_fill] = merged[cols_to_fill].fillna(fill_val)
        except Exception as e:
            logging.warning(f"Error filling NaNs in merged geometry: {e}")

        # Filter by Filter Shapefile if enabled (new logic with backward compatibility)
        try:
            filter_mode = None
            filter_gdf_to_use = getattr(self, 'filter_gdf', None)
            
            # Check new filter_shapefile_opt first
            if getattr(self, 'filter_overlay_var', None):
                val = self.filter_overlay_var.get()
                if isinstance(val, str):
                    s = val.strip().lower()
                    if s not in ('false', 'none', 'off', 'null', ''):
                        filter_mode = 'intersect' if s == 'true' else s
                elif isinstance(val, bool) and val:
                    filter_mode = 'intersect'
            
            # Backward compatibility: if filter_gdf not set but filtering requested, use overlay_gdf
            if filter_mode and filter_gdf_to_use is None:
                ov_gdf = getattr(self, 'overlay_gdf', None)
                if ov_gdf is not None:
                    _do_notify('WARNING', 'Deprecated', 'Using overlay for filtering. Please use filter_shapefile instead.', popup=False)
                    # Convert overlay list to combined GeoDataFrame for filtering
                    if isinstance(ov_gdf, list):
                        if ov_gdf:
                            target_crs = ov_gdf[0].crs
                            combined_parts = [ov_gdf[0]]
                            for other in ov_gdf[1:]:
                                if target_crs and other.crs and other.crs != target_crs:
                                    other = other.to_crs(target_crs)
                                combined_parts.append(other)
                            filter_gdf_to_use = pd.concat(combined_parts, ignore_index=True)
                    else:
                        filter_gdf_to_use = ov_gdf
            
            # Apply filtering
            if filter_mode and filter_gdf_to_use is not None:
                _do_notify('INFO', 'Filtering', f'Filtering data by shapefile ({filter_mode})...', popup=False)
                try:
                    merged = apply_spatial_filter(merged, filter_gdf_to_use, filter_mode)
                except Exception as e:
                    _do_notify('WARNING', 'Filter Failed', f"Could not filter by shapefile ({filter_mode}): {e}")
        except Exception as e:
            _do_notify('WARNING', 'Filter Logic Error', str(e))

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
                cand_idx = []
                
                # OPTIMIZATION: High-speed grid math for large datasets
                # This bypasses expensive SIndex intersection for 100k+ polygons
                info = getattr(gdf, 'attrs', {}).get('_smk_grid_info')
                if info and 'ROW' in gdf.columns:
                    try:
                        # 1. Coordinate to Native Grid space (e.g. LCC)
                        native_crs_str = info.get('proj_str')
                        lookup_x, lookup_y = x, y
                        
                        if native_crs_str:
                            try:
                                import pyproj
                                native_crs = pyproj.CRS.from_user_input(native_crs_str)
                                # The Axes CRS is recorded by the plotter
                                axes_crs = getattr(target_ax, '_smk_ax_crs', None) or getattr(gdf, 'crs', None)
                                
                                if axes_crs is not None and not native_crs.equals(axes_crs):
                                    if not hasattr(target_ax, '_hover_lookup_transformer'):
                                        target_ax._hover_lookup_transformer = pyproj.Transformer.from_crs(axes_crs, native_crs, always_xy=True)
                                    lookup_x, lookup_y = target_ax._hover_lookup_transformer.transform(x, y)
                            except Exception: pass
                        
                        # 2. Native space to Row/Col (1-based)
                        col_look = int(np.floor((lookup_x - info['xorig']) / info['xcell'])) + 1
                        row_look = int(np.floor((lookup_y - info['yorig']) / info['ycell'])) + 1
                        
                        # 3. Fast lookup in GDF
                        # Identify candidate index based on Row/Col
                        matches = gdf.index[(gdf['ROW'] == row_look) & (gdf['COL'] == col_look)].tolist()
                        if matches:
                            cand_idx = matches
                    except Exception: cand_idx = []
                
                # FALLBACK: Use Spatial Index (Intersection) if math failed or no match
                if not cand_idx:
                    if sindex is not None:
                        cand_idx = list(sindex.intersection((x, y, x, y)))
                    else:
                        cand_idx = range(len(gdf))

                best_parts = None
                best_val = -1.0
                
                for idx in cand_idx:
                    row = gdf.iloc[idx]
                    geom = row.geometry
                    if geom is not None and geom.contains(pt):
                        parts = [base]
                        # Identify region
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
                        
                        # Pull value
                        val = row.get(pollutant)
                        # Check for current values on axes (useful for NetCDF animation)
                        current_vals = getattr(target_ax, '_smk_current_vals', None)
                        if current_vals is not None:
                            try:
                                # Use the positional index if within bounds
                                if idx < len(current_vals):
                                    val = current_vals[idx]
                            except Exception:
                                pass
                        try:
                            f_val = float(val) if not pd.isna(val) else 0.0
                        except Exception:
                            f_val = 0.0
                            
                        if not pd.isna(val):
                            try:
                                parts.append(f"{pollutant}={f_val:.4g}")
                            except Exception:
                                parts.append(f"{pollutant}={val}")
                        
                        # If this is the best (highest non-zero) value we've seen, remember it
                        if f_val > best_val or best_parts is None:
                            best_val = f_val
                            best_parts = parts
                
                if best_parts:
                    return ", ".join(best_parts)
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
            if self.inputfile_path:
                self.load_inputfile(show_preview=False)
                return
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
            def safe_notify(level, title, msg, exc=None, **kwargs):
                if hasattr(self.root, 'run_on_main_signal'):
                    self.root.run_on_main_signal.emit(lambda l=level, t=title, m=msg, ex=exc, kw=kwargs: self._notify(l, t, m, exc=ex, **kw))
                else:
                    self.root.after(0, lambda l=level, t=title, m=msg, ex=exc, kw=kwargs: self._notify(l, t, m, exc=ex, **kw))

            try:
                merged = self._merged(
                    plot_by_mode=plot_by_mode, 
                    scc_selection=scc_selection, 
                    scc_code_map=scc_code_map,
                    notify=safe_notify,
                    pollutant=pollutant
                )
            except ValueError as ve:
                if str(ve) == "Handled":
                    # Diagnostic notification already sent
                    self.root.after(0, self._enable_plot_btn)
                    return
                raise ve
            
            if merged is None:
                if hasattr(self.root, 'run_on_main_signal'):
                    self.root.run_on_main_signal.emit(lambda: self._notify('WARNING', 'Missing Data', 'Load smkreport and shapefile first.'))
                    self.root.run_on_main_signal.emit(self._enable_plot_btn)
                else:
                    self.root.after(0, lambda: self._notify('WARNING', 'Missing Data', 'Load smkreport and shapefile first.'))
                    self.root.after(0, self._enable_plot_btn)
                return

            # Reproject
            plot_crs, tf_fwd, tf_inv = plot_crs_info
            try:
                merged_plot = merged.to_crs(plot_crs) if plot_crs is not None and getattr(merged, 'crs', None) is not None else merged
                # Ensure attributes needed for time series and QuadMesh are preserved
                if getattr(merged, 'attrs', None) and getattr(merged_plot, 'attrs', None) is not None:
                    for k in ['stack_groups_path', 'proxy_ncf_path', 'original_ncf_path', '_smk_grid_info', '_smk_is_native']:
                        if k in merged.attrs:
                                merged_plot.attrs[k] = merged.attrs[k]
            except Exception:
                merged_plot = merged
            
            if hasattr(self.root, 'run_on_main_signal'):
                self.root.run_on_main_signal.emit(lambda m=merged, mp=merged_plot, p=pollutant, pi=plot_crs_info: self._finalize_plot(m, mp, p, pi))
            else:
                self.root.after(0, lambda m=merged, mp=merged_plot, p=pollutant, pi=plot_crs_info: self._finalize_plot(m, mp, p, pi))
            
        except Exception as e:
            if hasattr(self.root, 'run_on_main_signal'):
                self.root.run_on_main_signal.emit(lambda e=e: self._notify('ERROR', 'Plot Prep Error', str(e), exc=e))
                self.root.run_on_main_signal.emit(self._enable_plot_btn)
            else:
                self.root.after(0, lambda e=e: self._notify('ERROR', 'Plot Prep Error', str(e), exc=e))
                self.root.after(0, self._enable_plot_btn)

    def _finalize_plot(self, merged, merged_plot, pollutant, plot_crs_info):
        import time
        _t_start = time.time()
        _t_last = [_t_start]  # Use list to avoid nonlocal issues
        def _log_time(msg):
            elapsed = time.time() - _t_last[0]
            total = time.time() - _t_start
            logging.info(f"[PLOT TIMING] {msg}: {elapsed:.2f}s (total: {total:.2f}s)")
            _t_last[0] = time.time()
        
        self._enable_plot_btn()
        self._set_status("Rendering plot...", level="INFO")
        _log_time("Plot initialization")
        # Persistence disabled

        # Lazy import for embedding
        from matplotlib.figure import Figure
        try:
            pass; # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except Exception as e:
            self._notify('ERROR', 'Backend Error', f'Failed to load TkAgg backend: {e}', exc=e)
            return

        # Determine if we can reuse an existing window for THIS pollutant.
        plot_state = self._plot_windows.get(pollutant)
        reuse_window = False
        
        if plot_state and plot_state.get('win'):
            try:
                if plot_state['win'].winfo_exists():
                    reuse_window = True
                else:
                    self._plot_windows.pop(pollutant, None)
                    plot_state = None
            except Exception:
                self._plot_windows.pop(pollutant, None)
                plot_state = None

        win = None
        local_fig = None
        local_ax = None
        canvas = None
        toolbar = None

        plot_container = None
        if reuse_window:
            # Reuse existing window assets
            try:
                local_ax = plot_state['ax']
                local_fig = plot_state['fig']
                canvas = plot_state['canvas']
                win = plot_state['win']
                toolbar = plot_state.get('toolbar')
                plot_container = plot_state.get('plot_container')

                # Clear the main axis
                local_ax.clear()
                
                # Clean up UI (Export, Time Series controls)
                if plot_container:
                    for child in plot_container.winfo_children():
                        # Keep Canvas and Toolbar
                        if child not in (canvas.get_tk_widget(), toolbar):
                                try: child.destroy()
                                except: pass

                # Cleanup old stats callbacks specific to this window
                old_cbids = plot_state.get('stats_cbids', [])
                for cid in old_cbids:
                    try: local_ax.callbacks.disconnect(cid)
                    except: pass
                plot_state['stats_cbids'] = []
                
                # Remove any other axes (colorbars) but keep main ax
                for ax in local_fig.axes:
                    if ax != local_ax:
                        local_fig.delaxes(ax)
                
                # Bring to front
                win.deiconify()
                win.lift()
                logging.info(f"Reusing existing plot window for {pollutant}.")
            except Exception as e:
                logging.warning("Failed to reuse window: %s", e)
                reuse_window = False
                self._plot_windows.pop(pollutant, None)
                plot_state = None

        if not reuse_window:
            # Create a new pop-out window for this plot
            win = tk.Toplevel(self.root)
            win.title(f"Map: {pollutant}")
            
            # Setup close handler to remove from dict
            def _on_destroy(event, p=pollutant, w=win):
                if event.widget == w:
                    try: self._plot_windows.pop(p, None)
                    except: pass
            win.bind("<Destroy>", _on_destroy)

            plot_container = ttk.Frame(win)
            plot_container.pack(side='top', fill='both', expand=True)
            # Scale figure size modestly with UI scale
            try:
                base_w, base_h = 9.0, 5.0
                fig_w = max(6.0, min(16.0, base_w * getattr(self, '_ui_scale', 1.0)))
                fig_h = max(3.5, min(10.0, base_h * getattr(self, '_ui_scale', 1.0)))
            except Exception:
                fig_w, fig_h = 9.0, 5.0
            
            try:
                # Force window size based on figure size + padding for toolbar
                w_px = int(fig_w * 100)
                h_px = int(fig_h * 100) + 100
                win.geometry(f"{w_px}x{h_px}")
            except Exception:
                pass

            local_fig = Figure(figsize=(fig_w, fig_h), dpi=100)
            local_ax = local_fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(local_fig, master=plot_container)
            canvas.draw()
            _log_time("Initial canvas draw")
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
                        if text == 'Zoom': return tk.Button(master=self, text='')
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
                    # Use attached data if available (fixes stale closure on reuse)
                    try:
                        gdf_src = getattr(local_ax, '_current_data', merged_plot)
                    except:
                        gdf_src = merged_plot
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
        # Save state to dictionary if this was a new window
        if not reuse_window:
            plot_state = {
                'win': win,
                'fig': local_fig,
                'ax': local_ax,
                'canvas': canvas,
                'toolbar': toolbar if 'toolbar' in locals() else None,
                'plot_container': plot_container,
                'stats_cbids': []
            }
            self._plot_windows[pollutant] = plot_state

        plot_crs, tf_fwd, tf_inv = plot_crs_info if plot_crs_info else (None, None, None)
        data = merged[pollutant]
        positive = data[data > 0].dropna()
        vmin = float(positive.min()) if not positive.empty else 0.0
        vmax_val = data.max(skipna=True)
        vmax = float(vmax_val) if pd.notna(vmax_val) else 0.0

        # Colormap selection
        cmap_name = self.cmap_var.get() if hasattr(self, 'cmap_var') and self.cmap_var else 'jet'

        # Auto-select boundary colors for contrast
        # Heuristic: dark-start maps (Jet, Viridis) => Use medium gray for context and bright Cyan for overlay highlight
        #            light-start maps (Greys, Reds) => Use darker gray for context and Black/Red for overlay
        _cmap_lower = cmap_name.lower()
        _dark_cmaps = {'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'turbo', 'nipy_spectral', 'gnuplot', 'gnuplot2'}
        
        if _cmap_lower in _dark_cmaps:
            county_color = 'black'    # User requested persistent black
            overlay_color = 'cyan'    # High contrast highlight (Neon)
            county_lw = 0.6           # Increased visibility
            overlay_lw = 0.8
        else:
            county_color = 'black'    # User requested persistent black
            overlay_color = 'black'   # Max contrast on light maps
            county_lw = 0.6           # Increased visibility
            overlay_lw = 0.8

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
            # Discrete mapping using boundaries
            # Handle user request for transparency below first bin:
            # - Use clip=False so values < bins[0] map to -1 (Under)
            # - Use extend='neither' to preserve original bin-to-color mapping (extend='max' shifts colors)
            # - Explicitly set under/over colors
            
            cmap = copy.copy(cmap)
            cmap.set_under('none')            # Transparent for values < bins[0]
            try:
                # Ensure values > bins[-1] stay opaque (last color) like clip=True
                cmap.set_over(cmap(cmap.N - 1)) 
            except Exception:
                pass
            
            norm = BoundaryNorm(bins, ncolors=cmap.N, clip=False, extend='neither')
        else:
            # Continuous; honor linear/log selection
            if self.scale_var.get() == 'log' and vmax > 0 and vmin > 0:
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = None

        # Plotting
        # Capture axes before plotting so we can detect the newly created colorbar axis
        pre_axes = set(local_fig.axes)
        
        # Optimization: Disable edges for large datasets to speed up rendering
        p_lw = 0.05
        p_ec = 'black'
        if len(merged_plot) > 20000:
            p_lw = 0.0
            p_ec = 'none'
            if getattr(self, '_notify', None):
                # Update status without popup
                try: self.status_var.set("Rendering optimized (edges disabled due to density)...")
                except: pass
        
        # Ensure uniform collection size for animation (plot all geometries, even if NaN)
        if pollutant in merged_plot.columns:
            try:
                merged_plot[pollutant] = merged_plot[pollutant].fillna(0.0)
            except Exception: pass

        # Ensure 'bad' values (NaNs) are plotted in the same collection using set_bad
        # This prevents GeoPandas from creating a separate collection for missing values,
        # which is required for the animation updates (set_array) to work correctly on the full grid.
        try:
            if not (len(bins) >= 2): # already copied if discrete
                cmap = copy.copy(cmap)
            cmap.set_bad('#f0f0f0', alpha=1.0)
        except Exception: pass

        # Attach data for export (closure workaround)
        local_ax._current_data = merged_plot
        
        # Use shared plotting function
        self._data_collection = create_map_plot(
            gdf=merged_plot,
            column=pollutant,
            title="", # GUI handles title via window logic
            ax=local_ax,
            cmap_name=cmap, # Pass object
            bins=bins, 
            unit_label=None, # GUI handles labeling manually below
            overlay_counties=None, # GUI handles layers separately
            overlay_shape=None,
            crs_proj=plot_crs, # needed for overlays if we added them, but helpful for context
            tf_fwd=None, # GUI handles graticule manually
            tf_inv=None,
            zoom_to_data=False, # GUI handles zoom manually
            # Style & Overrides
            linewidth=p_lw,
            edgecolor=p_ec,
            norm=norm,
            zorder=1
        )
        _log_time("GeoPandas plot() completed")

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
                        # Determine formatter that shows requested precision
                        def _fmt_precise(x, pos=None):
                            if x == 0: return "0"
                            # If very small, use up to 4 decimal places, else standard
                            if 0 < abs(x) < 0.1:
                                return f"{x:.4g}"
                            return f"{x:.4g}"
                        
                        from matplotlib.ticker import FuncFormatter

                        if orient_vertical:
                            try:
                                cbar_ax.yaxis.set_major_locator(FixedLocator(ticks))
                                cbar_ax.yaxis.set_major_formatter(FuncFormatter(_fmt_precise))
                            except Exception:
                                pass
                        else:
                            try:
                                cbar_ax.xaxis.set_major_locator(FixedLocator(ticks))
                                cbar_ax.xaxis.set_major_formatter(FuncFormatter(_fmt_precise))
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
                # Use a separate axis or just plain plot, but importantly do NOT make it a collection that looks like data
                # We save it to a variable that is NOT _data_collection
                overlay.plot(ax=local_ax, facecolor='none', edgecolor=county_color, linewidth=county_lw, alpha=0.6, zorder=20)
            except Exception:
                pass
        if self.overlay_gdf is not None:
            # Handle both single GeoDataFrame and list of GeoDataFrames
            overlay_list = self.overlay_gdf if isinstance(self.overlay_gdf, list) else [self.overlay_gdf]
            colors = ['cyan', 'magenta', 'yellow', 'red', 'lime', 'orange']
            
            for idx, ov_part in enumerate(overlay_list):
                try:
                    if plot_crs is not None and getattr(ov_part, 'crs', None) is not None:
                        ov_part = ov_part.to_crs(plot_crs)
                    
                    # Use unique color for each overlay
                    color = colors[idx % len(colors)]
                    ov_part.boundary.plot(ax=local_ax, facecolor='none', edgecolor=color, linewidth=overlay_lw, alpha=0.9, zorder=25, linestyle='--')
                except Exception as e:
                    logging.warning("Failed to plot overlay part %d: %s", idx, e)
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
        logging.info(f"DEBUG GUI Graticule Check: CRS={plot_crs}, FWD={tf_fwd}, INV={tf_inv}")
        try:
            if plot_crs is not None and tf_fwd is not None and tf_inv is not None:
                # Slightly behind the polygons (zorder=0 in draw function)
                self._draw_graticule(local_ax, tf_fwd, tf_inv, lon_step=None, lat_step=None)
                # Equal aspect to preserve shapes visually
                try:
                    local_ax.set_aspect('equal', adjustable='box')
                except Exception:
                    pass
            else:
                logging.warning("DEBUG GUI Graticule SKIPPED: Missing transformers or CRS")
        except Exception:
            pass
        # Record base view for this window and wire Home
        try:
            base_view = (local_ax.get_xlim(), local_ax.get_ylim())
            self._base_views[pollutant] = base_view # Store base view per pollutant
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
            attrs = getattr(self.emissions_df, 'attrs', {})
            src = attrs.get('source_name') if self.emissions_df is not None else None
            var_meta = attrs.get('variable_metadata', {}).get(pollutant, {})
        except Exception:
            src = None
            var_meta = {}

        # Build title and optional SCC subtitle
        long_name = var_meta.get('long_name')
        units = str(var_meta.get('units', '') or '').strip()
        
        if long_name:
            main_title = long_name
            if units:
                main_title += f" ({units})"
                # Also update units map for colorbar if not set
                if pollutant not in self.units_map:
                    self.units_map[pollutant] = units
        else:
            main_title = f"{pollutant} emissions from {src}" if src else f"{pollutant} emission"
        
        try:
            sel_disp = self.scc_select_var.get() if getattr(self, 'scc_select_var', None) else 'All SCC'
        except Exception:
            sel_disp = 'All SCC'

        # Append NetCDF Layer/Time info
        ncf_parts = []
        try:
            # Check if source is NetCDF
            src_type = attrs.get('source_type')
            is_ncf_source = (src_type == 'gridded_netcdf') or \
                            is_netcdf_file(self.inputfile_path) or \
                            ('proxy_ncf_path' in attrs)
            
            if is_ncf_source:
                if hasattr(self, 'ncf_layer_var') and self.ncf_layer_var.get():
                    ncf_parts.append(self.ncf_layer_var.get())
                if hasattr(self, 'ncf_tstep_var') and self.ncf_tstep_var.get():
                    ts_val = self.ncf_tstep_var.get()
                    if 'Sum' in ts_val:
                        ncf_parts.append(ts_val)
                    else:
                        ncf_parts.append(f"Time: {ts_val}")
        except Exception:
            pass

        title_lines = [main_title]
        if ncf_parts:
            title_lines.append(", ".join(ncf_parts))
        if sel_disp and sel_disp != 'All SCC':
            title_lines.append(f"SCC: {sel_disp}")
        
        # Apply Title to Axes (Fix for missing title)
        try:
            if title_lines:
                local_ax.set_title("\n".join(title_lines), fontsize=12)
        except Exception:
            pass

        # Add Time Series buttons if NCF
        is_ncf_source = False
        ts_click_handler = None
        try:
            src_type = attrs.get('source_type')
            is_ncf_source = (src_type == 'gridded_netcdf') or \
                            is_netcdf_file(self.inputfile_path) or \
                            ('proxy_ncf_path' in attrs)
        except Exception:
            pass
        
        # Text artist (Stats) anchored to bottom-left inside axes
        # Defined early so it can be updated by Time Nav
        stats_text = local_ax.text(
            0.01, 0.01,
            "",
            transform=local_ax.transAxes,
            ha='left', va='bottom', fontsize=9, color='#333333', zorder=30,
            bbox=dict(facecolor='white', edgecolor='#aaaaaa', alpha=0.7, pad=2)
        )

        if is_ncf_source:
            ts_frame = ttk.Frame(plot_container)
            ts_frame.pack(side='top', fill='x', pady=(2, 0))

            # --- Time Stepping Controls ---
            nav_frame = ttk.Frame(plot_container) 
            nav_frame.pack(side='top', fill='x', pady=(2, 0))
             
            self._t_data_cache = None
            self._t_idx = 0
            self._is_showing_agg = True
            # stats_box - removed, using stats_text
             
            lbl_time = ttk.Label(nav_frame, text="Time: Initial")
            lbl_time.pack(side='left', padx=5)

            def _ensure_time_data():
                if self._t_data_cache is not None: return True
                try:
                    from ncf_processing import get_ncf_animation_data
                except ImportError: return False
                 
                # Prepare indices for ALL cells in plot
                r_col = 'ROW'; c_col = 'COL'
                if 'ROW' not in merged_plot.columns and 'ROW_x' in merged_plot.columns: r_col = 'ROW_x'
                if 'COL' not in merged_plot.columns and 'COL_x' in merged_plot.columns: c_col = 'COL_x'
                 
                # 0-based indices
                try:
                    # Robust extraction: fillna(0) ensures no NaNs cause crash during casting
                    rows = merged_plot[r_col].fillna(0).astype(int) - 1
                    cols = merged_plot[c_col].fillna(0).astype(int) - 1
                except Exception as e:
                    logging.error(f"Failed to extract grid indices: {e}") 
                    return False
                 
                # Layer op
                l_idx = 0; l_op = 'select'
                try: 
                    lay_setting = self.ncf_layer_var.get()
                    if "Sum" in lay_setting: l_op = 'sum'
                    elif "Avg" in lay_setting: l_op = 'mean'
                    elif "Layer" in lay_setting:
                        try: l_idx = int(lay_setting.split()[-1]) - 1; l_op = 'select'
                        except: pass
                except: pass

                self._set_status("Loading full time series...", level="INFO")
                 
                # Determine variables before logging
                effective_path = self.inputfile_path
                stack_groups_path = None
                try:
                    if hasattr(self.emissions_df, 'attrs'):
                        if 'proxy_ncf_path' in self.emissions_df.attrs:
                                effective_path = self.emissions_df.attrs['proxy_ncf_path']
                        elif 'original_ncf_path' in self.emissions_df.attrs:
                                effective_path = self.emissions_df.attrs['original_ncf_path']
                          
                        if 'stack_groups_path' in self.emissions_df.attrs:
                                stack_groups_path = self.emissions_df.attrs['stack_groups_path']
                except Exception: pass

                logging.info(f"Loading Animation Data: Path={effective_path}, StackGroups={stack_groups_path}")
                if 'proxy_ncf_path' in getattr(self.emissions_df, 'attrs', {}):
                        logging.info("Using cached gridded file for animation: %s", effective_path)

                # Capture Current View Scale (Initial Scale)
                vmin_init, vmax_init = 0, 1
                init_coll = None
                try:
                    if hasattr(self, '_data_collection') and self._data_collection in local_ax.collections:
                        init_coll = self._data_collection
                    else:
                        for c in local_ax.collections:
                            if c.get_array() is not None:
                                init_coll = c
                                break
                    if init_coll is None and local_ax.collections: init_coll = local_ax.collections[0]
                    if init_coll:
                        vmin_init, vmax_init = init_coll.get_clim()
                except: pass

                try:
                    res = get_ncf_animation_data(
                        effective_path, 
                        pollutant, 
                        rows.tolist(), 
                        cols.tolist(), 
                        layer_idx=l_idx, 
                        layer_op=l_op,
                        stack_groups_path=stack_groups_path
                    )
                    if not res:
                        logging.error("get_ncf_animation_data returned None. Check logs for details.")
                        # Don't show popup error, just log it to avoid spamming if user clicks around
                        # But wait, this is _ensure_time_data, called once on creation.
                        # If it fails, animation won't work.
                        self._notify('WARNING', 'Animation Data', 'No time series data could be loaded for this layer.')
                        return False
                     
                    # Add Aggregates
                    # vals: (T, N)
                    vals = res['values']
                    times = res['times']
                     
                    # Total All Time (Sum across T)
                    tot = np.sum(vals, axis=0, keepdims=True)
                    # Average All Time (Mean across T)
                    avg = np.mean(vals, axis=0, keepdims=True)
                     
                    # Max/Min All Time
                    mx_agg = np.nanmax(vals, axis=0, keepdims=True)
                    mn_agg = np.nanmin(vals, axis=0, keepdims=True)
                     
                    # Store aggregates separately, do NOT stack into main playback loop
                    res['tot_val'] = tot.flatten()
                    res['avg_val'] = avg.flatten()
                    res['mx_val'] = mx_agg.flatten()
                    res['mn_val'] = mn_agg.flatten()
                     
                    res['values'] = vals
                    res['times'] = times
                     
                    # Calculate scale limits
                    # Use Global Min/Max across entire time dimension for consistency
                    # NOTE: 'vals' here is the original time-step data only, excluding Total/Avg aggregates.
                     
                    # Determine log scale: Prefer UI setting, then Norm type
                    is_log = False
                    try: 
                        if hasattr(self, 'scale_var') and self.scale_var.get() == 'log':
                            is_log = True
                        elif init_coll and isinstance(init_coll.norm, LogNorm):
                            is_log = True
                    except: pass
                     
                    if is_log:
                        pos_vals = vals[vals > 0]
                        res['vmin'] = pos_vals.min() if pos_vals.size > 0 else 0.001
                        res['vmax'] = vals.max() if vals.size > 0 else 1.0
                    else:
                        res['vmin'] = np.nanmin(vals)
                        res['vmax'] = np.nanmax(vals)

                    self._t_data_cache = res

                    # Sync _t_idx with the current UI selection (only on first load)
                    try:
                        # Attempt to align _t_idx with ncf_tstep_var
                        current_sel = self.ncf_tstep_var.get()
                        # Since aggregates are now separate, if user selected agg, we should maybe
                        # initialize plot to that agg? Or just defaut to time step 0?
                        # Let's default to time step 0 if an aggregate was selected, unless we add
                        # logic to immediately show the aggregate.
                        # The user asked to remove them from loop range.
                         
                        # If it's a specific time step, match it
                        found = False
                        for i, t in enumerate(times):
                            if str(t).strip() == str(current_sel).strip():
                                self._t_idx = i
                                self._is_showing_agg = False
                                found = True
                                break
                            if str(current_sel) in str(t):
                                self._t_idx = i
                                self._is_showing_agg = False
                                found = True
                                break
                         
                        if not found:
                            self._t_idx = 0
                            self._is_showing_agg = True
                             
                    except Exception:
                        pass

                    return True
                except Exception as e:
                    self._notify('ERROR', 'Time Data', str(e))
                    return False

            def _update_view(new_vals, time_lbl):
                coll = None
                try:
                    if hasattr(self, '_data_collection') and self._data_collection in local_ax.collections:
                        coll = self._data_collection
                    else:
                        if self._t_data_cache and 'values' in self._t_data_cache:
                            expected_size = self._t_data_cache['values'][0].size
                            for c in local_ax.collections:
                                arr = c.get_array()
                                if arr is not None and arr.size == expected_size:
                                    coll = c
                                    break
                    if coll is None and local_ax.collections:
                        c0 = local_ax.collections[0]
                        if c0.get_array() is not None: coll = c0
                except: pass

                try:
                    if coll:
                        coll.set_array(new_vals)
                        local_ax._smk_current_vals = new_vals
                        coll.set_clim(self._t_data_cache['vmin'], self._t_data_cache['vmax'])
                        if hasattr(coll, 'colorbar') and coll.colorbar:
                            coll.colorbar.update_normal(coll)

                        # Calculate Stats (View Dependent)
                        try:
                            x0, x1 = local_ax.get_xlim()
                            y0, y1 = local_ax.get_ylim()
                            from shapely.geometry import box
                            view_box = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                             
                            visible_idxs = list(merged_plot.sindex.intersection(view_box.bounds))
                             
                            if visible_idxs:
                                view_vals = new_vals[visible_idxs]
                                filtered = view_vals[~np.isnan(view_vals)]
                            else:
                                filtered = np.array([])

                            u_str = f" ({units})" if units else ""
                            new_title = f"{pollutant}{u_str}\nTime: {time_lbl}"
                            local_ax.set_title(new_title)

                            if filtered.size > 0:
                                mn = np.min(filtered); mx = np.max(filtered)
                                u = np.mean(filtered); s = np.sum(filtered)
                                def _fs(v): return f"{v:.4g}"
                                stats_txt = f"Min: {_fs(mn)}  Mean: {_fs(u)}  Max: {_fs(mx)}  Sum: {_fs(s)}"
                                if units: stats_txt += f" {units}"
                                try: stats_text.set_text(stats_txt)
                                except: pass
                            else:
                                try: stats_text.set_text("No Data in View")
                                except: pass
                        except Exception as ex:
                            print(f"Stats update error: {ex}")
                         
                        n_steps = len(self._t_data_cache['times'])
                        # Update label logic: if custom label (Agg), show it. If index-based, show index.
                        if "Sum" in time_lbl or "Mean" in time_lbl:
                            lbl_time.config(text=f"Time: {time_lbl}")
                        else:
                            lbl_time.config(text=f"Time: {time_lbl} ({self._t_idx+1}/{n_steps})")
                         
                        canvas.draw_idle()
                except Exception as e:
                    print(f"Update failed: {e}")

            def _step_time(delta):
                if not _ensure_time_data(): return
                n_steps = len(self._t_data_cache['times'])
                old_idx = self._t_idx
                if getattr(self, '_is_showing_agg', False):
                    self._t_idx = 0 if delta > 0 else n_steps - 1
                    self._is_showing_agg = False
                else:
                    self._t_idx = (self._t_idx + delta) % n_steps
                new_vals = self._t_data_cache['values'][self._t_idx]
                time_lbl = self._t_data_cache['times'][self._t_idx]
                logging.info(f"Animation Step: {old_idx} -> {self._t_idx}, ValRange=[{new_vals.min():.2f}, {new_vals.max():.2f}]")
                _update_view(new_vals, time_lbl)
            
            def _show_agg(mode):
                if not _ensure_time_data(): return
                self._is_showing_agg = True
                if mode == 'total':
                    new_vals = self._t_data_cache['tot_val']
                    lbl = "Total (Sum)"
                elif mode == 'avg':
                    new_vals = self._t_data_cache['avg_val']
                    lbl = "Average (Mean)"
                elif mode == 'max':
                    new_vals = self._t_data_cache['mx_val']
                    lbl = "Max All Time (Grid-wise)"
                elif mode == 'min':
                    new_vals = self._t_data_cache['mn_val']
                    lbl = "Min All Time (Grid-wise)"
                _update_view(new_vals, lbl)

            ttk.Button(nav_frame, text="< Prev", command=lambda: _step_time(-1)).pack(side='left', padx=2)
            ttk.Button(nav_frame, text="Next >", command=lambda: _step_time(1)).pack(side='left', padx=2)
            ttk.Button(nav_frame, text="Total", command=lambda: _show_agg('total')).pack(side='left', padx=5)
            ttk.Button(nav_frame, text="Avg", command=lambda: _show_agg('avg')).pack(side='left', padx=2)
            ttk.Button(nav_frame, text="Max", command=lambda: _show_agg('max')).pack(side='left', padx=2)
            ttk.Button(nav_frame, text="Min", command=lambda: _show_agg('min')).pack(side='left', padx=2)
             
            def _on_ts(mode):
                try:
                    from ncf_processing import get_ncf_timeseries
                except ImportError:
                    self._notify('ERROR', 'Time Series', 'Could not import NetCDF processing module.')
                    return
                 
                # 1. Get current bounds
                xmin, xmax = local_ax.get_xlim()
                ymin, ymax = local_ax.get_ylim()
                 
                # 2. Filter gdf
                try:
                    # merged_plot is in plot CRS, limits are in plot CRS
                    # Filter by intersection
                    from shapely.geometry import box
                    bbox = box(min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))
                     
                    sidx = merged_plot.sindex
                    cand_idxs = list(sidx.intersection(bbox.bounds))
                    subset = merged_plot.iloc[cand_idxs]
                    subset = subset[subset.intersects(bbox)]
                     
                    if subset.empty:
                        self._notify('WARNING', 'Time Series', 'No grid cells in current view.')
                        return
                     
                    # 3. Get indices
                    # We need ROW and COL columns. 
                    # They are 1-based in DF.
                    # Handle overlap suffix from merge (_x from geom, _y from emis)
                    r_col = 'ROW'
                    c_col = 'COL'
                    if 'ROW' not in subset.columns:
                        if 'ROW_x' in subset.columns: r_col = 'ROW_x'
                        else: r_col = None
                    if 'COL' not in subset.columns:
                        if 'COL_x' in subset.columns: c_col = 'COL_x'
                        else: c_col = None
                          
                    if not r_col or not c_col:
                        if 'GRID_RC' in subset.columns:
                            # Fallback: parse GRID_RC
                            try:
                                # assuming R_C
                                subset['_TS_ROW'] = subset['GRID_RC'].apply(lambda x: int(x.split('_')[0]))
                                subset['_TS_COL'] = subset['GRID_RC'].apply(lambda x: int(x.split('_')[1]))
                                r_col = '_TS_ROW'
                                c_col = '_TS_COL'
                            except:
                                self._notify('ERROR', 'Time Series', 'Grid ROW/COL could not be derived.')
                                return
                        else:
                            self._notify('ERROR', 'Time Series', 'Grid ROW/COL information missing from plot data.')
                            return
                     
                    rows_0 = subset[r_col].astype(int) - 1
                    cols_0 = subset[c_col].astype(int) - 1
                     
                    # Check bounds validity
                    if (rows_0 < 0).any() or (cols_0 < 0).any():
                        self._notify('ERROR', 'Time Series', 'Invalid grid indices found.')
                        return
                     
                except Exception as e:
                    self._notify('ERROR', 'Time Series Prep', f"Failed to filter grid: {e}")
                    return

                # 4. Get Layer info
                try:
                    lay_setting = self.ncf_layer_var.get()
                    l_idx = 0
                    l_op = 'select'
                    if 'Sum All' in lay_setting:
                        l_idx = None
                        l_op = 'sum'
                    elif 'Average' in lay_setting:
                        l_idx = None
                        l_op = 'mean'
                    elif 'Layer' in lay_setting:
                        try: l_idx = int(lay_setting.split()[-1]) - 1
                        except: l_idx = 0
                        l_op = 'select'
                except: # fallback
                    l_idx = 0
                    l_op = 'select'
                 
                # 5. Call backend
                self._set_status("Extracting time series...", level="INFO")

                # Prepare arguments for Inline support
                stack_groups_path = None
                effective_path = self.inputfile_path
                try:
                    stack_groups_path = merged_plot.attrs.get('stack_groups_path')
                    if 'proxy_ncf_path' in merged_plot.attrs:
                        effective_path = merged_plot.attrs['proxy_ncf_path']
                    if 'original_ncf_path' in merged_plot.attrs:
                        effective_path = merged_plot.attrs['original_ncf_path']
                except Exception:
                    pass

                try:
                    result = get_ncf_timeseries(
                        effective_path,
                        pollutant,
                        rows_0.tolist(),
                        cols_0.tolist(),
                        layer_idx=l_idx,
                        layer_op=l_op,
                        op=mode,
                        stack_groups_path=stack_groups_path
                    )
                except Exception as e:
                    self._notify('ERROR', 'Extract Error', str(e))
                    return
                     
                if not result:
                    self._notify('WARNING', 'Time Series', 'Extraction returned no data.')
                    return
                 
                # 6. Plot
                ts_view = tk.Toplevel(self.root)
                ts_title = f"{pollutant} Time Series ({mode.title()}) - {len(subset)} Cells"
                ts_view.title(ts_title)
                try:
                    # 800x500 base size for time series popups
                    w_ts = int(800 * getattr(self, '_ui_scale', 1.0))
                    h_ts = int(500 * getattr(self, '_ui_scale', 1.0))
                    ts_view.geometry(f"{w_ts}x{h_ts}")
                except Exception:
                    ts_view.geometry("800x500")
                 
                import matplotlib.pyplot as plt
                pass; # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
                 
                fig_ts, ax_ts = plt.subplots(figsize=(8, 4))
                times = result['times']
                vals = result['values']
                units = result['units']
                 
                # Parse time labels? YYYYDDD_HHMMSS
                # Maybe simplify X axis labels
                if isinstance(vals, dict):
                        # Multi-line plot (Inline Sources)
                        for label, series in vals.items():
                            lw = 2.5 if label == 'Total' else 1.0
                            alpha = 1.0 if label == 'Total' else 0.6
                            ms = 4 if label == 'Total' else 2
                            ax_ts.plot(series, marker='o', markersize=ms, label=label, linewidth=lw, alpha=alpha)
                      
                        # Only show legend if reasonable number of items
                        if len(vals) < 25:
                            ax_ts.legend(fontsize='small', loc='upper right')
                        else:
                            # If too many, maybe just Total in legend?
                            ax_ts.legend(['Total'], loc='upper right')
                else:
                        ax_ts.plot(vals, marker='o', markersize=4)
                      
                u_str = str(units or '').strip()
                y_lbl = f"{pollutant} ({u_str})" if u_str else pollutant
                ax_ts.set_ylabel(y_lbl)
                ax_ts.set_xlabel("Time Step")
                ax_ts.grid(True, linestyle='--', alpha=0.7)
                 
                if len(times) > 10:
                        # Sparse ticks
                        step = len(times) // 10
                        ax_ts.set_xticks(range(0, len(times), step))
                        ax_ts.set_xticklabels([times[i] for i in range(0, len(times), step)], rotation=30, ha='right')
                else:
                        ax_ts.set_xticks(range(len(times)))
                        ax_ts.set_xticklabels(times, rotation=30, ha='right')
                 
                fig_ts.tight_layout()
                 
                cv = FigureCanvasTkAgg(fig_ts, master=ts_view)
                cv.draw()
                cv.get_tk_widget().pack(side='top', fill='both', expand=True)
                 
                tb = NavigationToolbar2Tk(cv, ts_view)
                tb.update()
                tb.pack(side='top', fill='x')
                 
                self._set_status("Time series extracted.", level="INFO")
             
            ttk.Button(ts_frame, text="Time_Series_Total", command=lambda: _on_ts('sum')).pack(side='left', padx=4)
            ttk.Button(ts_frame, text="Time_Series_Averaged", command=lambda: _on_ts('mean')).pack(side='left', padx=4)

            def _display_ts_window(result, title_msg):
                ts_view = tk.Toplevel(self.root)
                ts_view.title(title_msg)
                try:
                    # 800x500 base size for time series popups
                    w_ts = int(800 * getattr(self, '_ui_scale', 1.0))
                    h_ts = int(500 * getattr(self, '_ui_scale', 1.0))
                    ts_view.geometry(f"{w_ts}x{h_ts}")
                except Exception:
                    ts_view.geometry("800x500")
                import matplotlib.pyplot as plt
                pass; # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
                 
                fig_ts, ax_ts = plt.subplots(figsize=(8, 4))
                times = result['times']
                vals = result['values']
                units = result['units']
                 
                if isinstance(vals, dict):
                        # Multi-line plot (Inline Sources)
                        for label, series in vals.items():
                            lw = 2.5 if label == 'Total' else 1.0
                            alpha = 1.0 if label == 'Total' else 0.6
                            ms = 4 if label == 'Total' else 2
                            ax_ts.plot(series, marker='o', markersize=ms, label=label, linewidth=lw, alpha=alpha)
                      
                        if len(vals) < 25:
                            ax_ts.legend(fontsize='small', loc='upper right')
                        else:
                            ax_ts.legend(['Total'], loc='upper right')
                else:
                        ax_ts.plot(vals, marker='o', markersize=4)

                u_str = str(units or '').strip()
                y_lbl = f"{pollutant} ({u_str})" if u_str else pollutant
                ax_ts.set_ylabel(y_lbl)
                ax_ts.set_xlabel("Time Step")
                ax_ts.grid(True, linestyle='--', alpha=0.7)
                 
                if len(times) > 10:
                        step = max(1, len(times) // 10)
                        ax_ts.set_xticks(range(0, len(times), step))
                        ax_ts.set_xticklabels([times[i] for i in range(0, len(times), step)], rotation=30, ha='right')
                else:
                        ax_ts.set_xticks(range(len(times)))
                        ax_ts.set_xticklabels(times, rotation=30, ha='right')
                 
                fig_ts.tight_layout()
                 
                cv = FigureCanvasTkAgg(fig_ts, master=ts_view)
                cv.draw()
                cv.get_tk_widget().pack(side='top', fill='both', expand=True)
                 
                tb = NavigationToolbar2Tk(cv, ts_view)
                tb.update()
                tb.pack(side='top', fill='x')

            def _on_click(event):
                # Click-to-plot logic
                if event.inaxes != local_ax: return
                if getattr(toolbar, 'mode', '') != '': return
                if event.button != 1: return
                 
                try:
                    from shapely.geometry import Point
                    x, y = event.xdata, event.ydata
                    pt = Point(x, y)
                     
                    target = None
                    # OPTIMIZATION: High-speed grid math check
                    info = getattr(merged_plot, 'attrs', {}).get('_smk_grid_info')
                    if info and 'ROW' in merged_plot.columns:
                        try:
                            lx, ly = x, y
                            # Transform click to native space if needed
                            native_crs_str = info.get('proj_str')
                            if native_crs_str:
                                import pyproj
                                native_crs = pyproj.CRS.from_user_input(native_crs_str)
                                axes_crs = getattr(local_ax, '_smk_ax_crs', None) or getattr(merged_plot, 'crs', None)
                                if axes_crs is not None and not native_crs.equals(axes_crs):
                                    if not hasattr(local_ax, '_click_lookup_transformer'):
                                        local_ax._click_lookup_transformer = pyproj.Transformer.from_crs(axes_crs, native_crs, always_xy=True)
                                    lx, ly = local_ax._click_lookup_transformer.transform(x, y)
                            
                            c_look = int(np.floor((lx - info['xorig']) / info['xcell'])) + 1
                            r_look = int(np.floor((ly - info['yorig']) / info['ycell'])) + 1
                            
                            # Lookup
                            matches = merged_plot[(merged_plot['ROW'] == r_look) & (merged_plot['COL'] == c_look)]
                            if not matches.empty:
                                target = matches.iloc[0]
                        except Exception: target = None
                    
                    if target is None:
                        # FALLBACK: Spatial lookup
                        if hasattr(merged_plot, 'sindex') and merged_plot.sindex:
                            cands = list(merged_plot.sindex.intersection((x, y, x, y)))
                            for idx in cands:
                                row = merged_plot.iloc[idx]
                                if row.geometry.contains(pt):
                                    target = row
                                    break
                        else:
                            matches = merged_plot[merged_plot.intersects(pt)]
                            if not matches.empty:
                                target = matches.iloc[0]
                         
                    if target is None: return

                    # Extract R/C
                    r = None; c = None
                    if 'ROW' in target: r = target['ROW']
                    elif 'ROW_x' in target: r = target['ROW_x']
                     
                    if 'COL' in target: c = target['COL']
                    elif 'COL_x' in target: c = target['COL_x']
                     
                    if (r is None or c is None) and 'GRID_RC' in target:
                        try:
                                parts = str(target['GRID_RC']).split('_')
                                r = int(parts[0])
                                c = int(parts[1])
                        except: pass
                          
                    if r is None or c is None: return

                    l_idx = 0; l_op = 'select'
                    try: 
                        if hasattr(self, 'ncf_layer_var'):
                            l_sel = self.ncf_layer_var.get()
                            if "Sum" in l_sel: l_op = 'sum'
                            elif "Avg" in l_sel: l_op = 'mean'
                            else:
                                parts = l_sel.split()
                                if len(parts) > 1 and parts[-1].isdigit():
                                    l_idx = int(parts[-1]) - 1
                    except: pass
                     
                    from ncf_processing import get_ncf_timeseries
                    self._set_status(f"Extracting Cell ({r}, {c})...", level="INFO")
                     
                    attrs = getattr(merged_plot, 'attrs', {})
                    effective_path = self.inputfile_path
                    if 'proxy_ncf_path' in attrs:
                        effective_path = attrs['proxy_ncf_path']
                    if 'original_ncf_path' in attrs:
                        effective_path = attrs['original_ncf_path']

                    stack_groups_path = None
                    if 'stack_groups_path' in attrs:
                        stack_groups_path = attrs['stack_groups_path']
                          
                    res = get_ncf_timeseries(
                        effective_path,
                        pollutant,
                        [int(r)-1],
                        [int(c)-1],
                        layer_idx=l_idx,
                        layer_op=l_op,
                        op='mean',
                        stack_groups_path=stack_groups_path
                    )
                     
                    if res:
                        _display_ts_window(res, f"{pollutant} Time Series Cell ({r}, {c})")
                        self._set_status(f"Plotted Cell {r}_{c}", level="INFO")
                     
                except Exception as e:
                    logging.error(f"Click handler failed: {e}")
             
            ts_click_handler = _on_click

        local_ax.set_title("\n".join(title_lines))
        # Install hover/status text formatter with emission values and WGS84 lon/lat
        try:
            # Initialize current values for hover lookup (NetCDF animation support)
            local_ax._smk_current_vals = merged_plot[pollutant].values
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
                    if ts_click_handler:
                        ts_click_handler(event)
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
            # Determine if fill was applied
            fill_active = False
            try:
                if getattr(self, 'fill_zero_var', None) and self.fill_zero_var.get():
                    fill_active = True
                else:
                    arg = getattr(self, 'fill_nan_arg', None)
                    if arg is not None and str(arg).lower() != 'false':
                        fill_active = True
            except Exception:
                pass

            if (
                not fill_active
                and isinstance(prepared_for_stats, pd.DataFrame)
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
            # stats_text created earlier

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
            # Connect to axis limit changes
            plot_state['stats_cbids'] = []
            try:
                plot_state['stats_cbids'].append(local_ax.callbacks.connect('xlim_changed', lambda ax: _update_stats_for_view()))
            except Exception:
                pass
            try:
                plot_state['stats_cbids'].append(local_ax.callbacks.connect('ylim_changed', lambda ax: _update_stats_for_view()))
            except Exception:
                pass
            # Run once initially
            _update_stats_for_view()
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
            if self.inputfile_path:
                self.load_inputfile(show_preview=True)
                return
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