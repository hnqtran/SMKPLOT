"""Utility functions for SMKPLOT GUI."""

import os
import sys
import traceback
from typing import Optional
import matplotlib


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

def _config_file() -> str:
    """Return the path to the configuration file."""
    cfg_dir = os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('./'), '.config')
    try:
        os.makedirs(cfg_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(cfg_dir, 'smkgui_settings.json')

def load_settings() -> dict:
    """Load the entire settings dictionary from the JSON config file."""
    import json
    try:
        cfg = _config_file()
        if not os.path.exists(cfg):
            return {}
        with open(cfg, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_settings(settings: dict) -> None:
    """Collect and save current GUI settings to the JSON config file."""
    import json
    try:
        cfg = _config_file()
        with open(cfg, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception:
        # best-effort; ignore failures
        pass