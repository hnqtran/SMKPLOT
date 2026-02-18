# Author: tranhuy@email.unc.edu
"""Utility functions for SMKPLOT GUI."""

import os
import sys
import traceback
import json
from typing import Optional, List, Dict, Any, Union, Tuple
import matplotlib

import numpy as np
import pandas as pd



def normalize_delim(d: Optional[str]) -> Optional[str]:
    """Normalize common delimiter tokens to actual characters."""
    if d is None:
        return None
    if d == '\\t':  # literal backslash t from shell
        return '\t'
    low = d.lower()
    mapping = {
        'tab': '\t',
        'comma': ',',
        'semicolon': ';',
        'pipe': '|',
        'space': ' ',
        'whitespace': ' ',
        'csv': ',',
        'tsv': '\t',
    }
    return mapping.get(low, d)


def coerce_merge_key(series: pd.Series, pad: Optional[int] = None) -> pd.Series:
    """Standardize merge keys to strings with optional zero-padding.

    The implementation favors vectorized operations for common cases to
    avoid the per-row Python ``apply`` that can stall large merges. We fall
    back to a safe row-wise conversion only when necessary (mixed dtypes,
    exotic objects, etc.).
    """

    if not isinstance(series, pd.Series):  # defensive; should not happen
        series = pd.Series(series)

    original = series
    pad_len = pad or 0

    def _finalize(s: pd.Series) -> pd.Series:
        if pad_len > 0:
            try:
                obj = s.astype('string')
                obj = obj.mask(obj.isna(), None)
                padded = obj.fillna('').str.zfill(pad_len)
                padded = padded.mask(obj.isna(), None)
                return padded.astype('object')
            except Exception:
                pass
        return s

    # Fast path: already string-like or simple
    # If no padding and already object/string, check if we need to do anything
    if pad_len == 0 and pd.api.types.is_string_dtype(series):
        return series

    if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
        try:
            # If large series, check duplication? 
            # Optim: if no padding required, simple astr
            if pad_len == 0:
                return series.astype(str)
            
            sample = series.dropna().head(32)
            if sample.empty or sample.map(lambda v: isinstance(v, str)).all():
                coerced = series.astype('string').str.strip()
                coerced = coerced.mask(coerced == '', None)
                coerced = coerced.astype('object')
                return _finalize(coerced)
        except Exception:
            pass

    # Integer dtypes (including pandas nullable ints)
    if pd.api.types.is_integer_dtype(series):
        try:
            coerced = series.astype('Int64').astype(str)
            coerced = coerced.where(series.notna(), None)
            return _finalize(coerced.astype('object'))
        except Exception:
            pass

    # Floating point: preserve integral values when possible
    if pd.api.types.is_float_dtype(series):
        try:
            arr = series.to_numpy(copy=False)
            mask = np.isfinite(arr)
            int_mask = mask & np.isclose(arr, np.round(arr))
            str_series = series.astype(str)
            if int_mask.any():
                ints = pd.Series(np.round(arr[int_mask]).astype(np.int64), index=series.index[int_mask])
                str_series.loc[int_mask] = ints.astype(str)
            coerced = str_series.where(series.notna(), None)
            return _finalize(coerced.astype('object'))
        except Exception:
            pass

    # Fallback: row-wise conversion
    def _convert(val):
        if pd.isna(val):
            return None
        try:
            if isinstance(val, (int, np.integer)):
                s_val = str(int(val))
            elif isinstance(val, (float, np.floating)):
                if np.isfinite(val) and float(val).is_integer():
                    s_val = str(int(val))
                else:
                    s_val = str(float(val))
            else:
                s_val = str(val).strip()
                if not s_val:
                    return None
        except Exception:
            s_val = str(val).strip()
            if not s_val:
                return None
        if pad_len:
            try:
                num = int(float(s_val))
                return f"{num:0{pad_len}d}"
            except Exception:
                pass
        return s_val

    try:
        return original.apply(_convert)
    except Exception:
        return original




def safe_sector_slug(sector: Optional[str]) -> str:
    """Generate a filename-safe slug from a sector name."""
    text = str(sector or 'default')
    slug = ''.join(ch if (ch.isalnum() or ch in {'-', '_'}) else '_' for ch in text)
    slug = slug.strip('_') or 'default'
    return slug


def serialize_attrs(attrs) -> dict:
    """Safely serialize a dictionary of attributes to JSON-compatible dict."""
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


def is_netcdf_file(filepath: str) -> bool:
    """Detect if a file is a NetCDF file by checking its magic number signature.
    Supported: Classic, 64-bit Offset, 64-bit Data, and NetCDF-4 (HDF5).
    """
    if not os.path.isfile(filepath):
        return False
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
            # NetCDF Classic (CDF\x01), 64-bit Offset (CDF\x02), 64-bit Data (CDF\x05)
            if header.startswith(b'CDF\x01') or header.startswith(b'CDF\x02') or header.startswith(b'CDF\x05'):
                return True
            # NetCDF-4 (HDF5 format signature: \x89HDF\r\n\x1a\n)
            if header.startswith(b'\x89HDF\r\n\x1a\n'):
                return True
    except Exception:
        pass
    return False




def _prune_incompatible_bundled_libs() -> None:
    """Drop PyInstaller-bundled glibc toolchain libs that require newer glibc.

    Ubuntu 20.04 ships glibc 2.31. When we build on a newer distro, PyInstaller
    may bundle host versions of libgcc_s/libstdc++ that require glibc >= 2.34,
    causing ImportError when NumPy loads. Removing those copies lets the runtime
    fall back to the system-provided libraries (which are compatible with 2.31).

    The extraction directory is unique per run, so deleting the files is safe.
    """

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return

    for candidate in ("libgcc_s.so.1", "libstdc++.so.6"):
        path = os.path.join(meipass, candidate)
        if os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass


def _import_numpy_with_diagnostics():
    try:
        import numpy as _np  # type: ignore
        return _np
    except Exception as exc:  # pragma: no cover - diagnostic branch
        def _describe(err: Optional[BaseException]) -> str:
            if err is None:
                return "<none>"
            return f"{err.__class__.__name__}: {err}"

        sys.stderr.write("\n[NumPy diagnostics] import numpy failed.\n")
        sys.stderr.write(f"  sys.executable: {sys.executable}\n")
        sys.stderr.write(f"  sys.argv[0]: {sys.argv[0]}\n")
        sys.stderr.write(f"  cwd: {os.getcwd()}\n")
        sys.stderr.write(f"  sys.path: {sys.path}\n")
        sys.stderr.write(f"  exception: {_describe(exc)}\n")
        sys.stderr.write(f"  __cause__: {_describe(getattr(exc, '__cause__', None))}\n")
        sys.stderr.write(f"  __context__: {_describe(getattr(exc, '__context__', None))}\n")
        sys.stderr.write("  traceback follows:\n")
        traceback.print_exc()
        raise

_display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
try:
    matplotlib.use('TkAgg' if _display else 'Agg')
except Exception:
    matplotlib.use('Agg')

try:
    import matplotlib.pyplot  # noqa: F401
except Exception:
    pass

USING_TK = matplotlib.get_backend().lower().startswith('tk')
if USING_TK:
    try:
        import tkinter as tk  # type: ignore
        from tkinter import filedialog, ttk  # type: ignore
    except Exception:
        USING_TK = False
        matplotlib.use('Agg')
        try:
            import matplotlib.pyplot  # noqa: F401
        except Exception:
            pass
        tk = None  # type: ignore
        ttk = None  # type: ignore
        filedialog = None  # type: ignore
else:
    tk = None  # type: ignore
    ttk = None  # type: ignore
    filedialog = None  # type: ignore

