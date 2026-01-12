"""Data processing functions for SMKPLOT GUI."""

import io
import os, sys
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence, Dict, List, Any, Union, Tuple, Set

import logging
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from joblib import Memory  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    Memory = None

import pyproj

# Set PROJ data directory for pyproj
proj_lib_path = os.path.join(os.path.dirname(os.path.dirname(pyproj.__file__)), 'proj_dir', 'share', 'proj')
if os.path.exists(proj_lib_path):
    pyproj.datadir.set_data_dir(proj_lib_path)
    db_path = os.path.join(proj_lib_path, 'proj.db')
    if os.path.exists(db_path):
        pyproj.database.set_context_database_path(db_path)

import csv
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from config import USE_SPHERICAL_EARTH


_CLEAN_NAME_QUOTE_RE = re.compile(r"['`,]")
_CLEAN_NAME_WHITESPACE_RE = re.compile(r"\s+")
_GRIDDESC_SPLIT_RE = re.compile(r',\s*|\s+')
_UNNAMED_COLUMN_RE = re.compile(r'^Unnamed:')

# Define constant column name lists for remapping
country_cols = ['country_cd','country','country_id']
tribal_cols = ['tribal_code','tribal_name','tribe_id']
region_cols = ['region_cd', 'regioncd','region', 'county_id','fips']
facility_cols = ['facility_id', 'facility']
unit_cols = ['unit_id']
rel_cols = ['rel_point_id','release_point_id','point_id']
emis_cols = ['ann_value', 'emission']
scc_cols = ['scc']
pol_cols = ['poll','pollutant']
lat_cols = ['latitude', 'lat']
lon_cols = ['longitude', 'lon']

# Define country code mappings following  ISO 3166-1 alpha-2 codes for Northern Hemisphere countries
country_code_mappings = {
    'US': '0',    # United States
    'CA': '1',    # Canada
    'MX': '2',    # Mexico
    'RU': '3',    # Russia
    'CN': '4',    # China
    'JP': '5',    # Japan
    'GB': '6',    # United Kingdom
    'FR': '7',    # France
    'DE': '8',    # Germany
    'IT': '9',    # Italy
    'ES': '10',   # Spain
    'KR': '11',   # South Korea
    'TR': '12',   # Turkey
    'IR': '13',   # Iran
    'SA': '14',   # Saudi Arabia
    'UA': '15',   # Ukraine
    'PL': '16',   # Poland
    'IQ': '17',   # Iraq
    'AF': '18',   # Afghanistan
    'PK': '19',   # Pakistan
    'ID': '20',   # Indonesia (partly in Northern Hemisphere)
    'EG': '21',   # Egypt
    'NG': '22',   # Nigeria
    'ET': '23',   # Ethiopia
    'DZ': '24',   # Algeria
    'MA': '25',   # Morocco
    'VE': '26',   # Venezuela
    'TH': '27',   # Thailand
    'VN': '28',   # Vietnam
    'PH': '29',   # Philippines
    'SD': '30',   # Sudan
    # ... Add all other Northern Hemisphere countries as needed
}

def _env_flag(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


_CACHE_DISABLED = _env_flag('SMKGUI_DISABLE_CACHE')
_memory: Optional[Memory] = None  # type: ignore[assignment]


def _resolve_default_cache_dir() -> Optional[str]:
    try:
        here = Path(__file__).resolve()
    except Exception:
        here = Path.cwd()
    search_roots = [here.parent, here.parent.parent, here.parent.parent.parent]
    for root in search_roots:
        try:
            candidate = root / 'smkplot.py'
        except Exception:
            continue
        if candidate.exists():
            target = root / 'joblib_cache'
            try:
                target.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue
            return str(target)
    fallback = Path(tempfile.gettempdir()) / 'smkgui_modular_cache'
    try:
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)
    except Exception:
        return None


if not _CACHE_DISABLED and Memory is not None:
    cache_dir = _resolve_default_cache_dir()
    if cache_dir:
        try:
            _memory = Memory(cache_dir, verbose=0)
        except Exception:
            _memory = None


def _memoize(maxsize: int = 4):
    def decorator(fn):
        if _memory is not None:
            return _memory.cache(fn)
        if _CACHE_DISABLED:
            return fn
        return lru_cache(maxsize=maxsize)(fn)

    return decorator


def _coerce_merge_key(series: pd.Series, pad: Optional[int] = None) -> pd.Series:
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

    # Fast path: already string-like
    if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
        try:
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


def merge_emissions_with_geometry(
    emis_df: pd.DataFrame,
    base_geom: gpd.GeoDataFrame,
    merge_on: str,
    *,
    sort: bool = False,
    copy_geometry: bool = False,
    pad_fips: bool = True,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Prepare emissions data and merge with geometry using shared rules.

    Returns a tuple of (merged GeoDataFrame, prepared emissions DataFrame). When duplicate
    keys exist on the emissions side, numeric columns are summed via a straightforward
    pandas ``groupby``. Both sides have their merge key coerced to comparable string forms so
    GUI and batch share a single code path.
    """

    if not isinstance(emis_df, pd.DataFrame):
        raise TypeError("emis_df must be a pandas DataFrame")
    if not isinstance(base_geom, gpd.GeoDataFrame):
        raise TypeError("base_geom must be a GeoDataFrame")
    if merge_on not in emis_df.columns:
        raise KeyError(f"'{merge_on}' not found in emissions DataFrame")
    if merge_on not in base_geom.columns:
        raise KeyError(f"'{merge_on}' not found in geometry DataFrame")

    pad_len = 6 if pad_fips and str(merge_on).upper() == 'FIPS' else None

    emis_prepped = emis_df
    if merge_on in emis_df.columns and not emis_df[merge_on].is_unique:
        attrs = dict(getattr(emis_df, 'attrs', {}))
        try:
            grouped = emis_df.groupby(merge_on, dropna=False, sort=sort)
        except TypeError:
            grouped = emis_df.groupby(merge_on, sort=sort)
        emis_prepped = grouped.sum(numeric_only=True).reset_index()
        try:
            emis_prepped.attrs = attrs
        except Exception:
            pass
    elif emis_df is not emis_prepped:
        try:
            emis_prepped.attrs = dict(getattr(emis_df, 'attrs', {}))
        except Exception:
            pass

    if isinstance(emis_prepped, pd.DataFrame):
        try:
            coerced = _coerce_merge_key(emis_prepped[merge_on], pad_len)
        except Exception:
            coerced = emis_prepped[merge_on]
        needs_update = False
        try:
            needs_update = not emis_prepped[merge_on].equals(coerced)
        except Exception:
            needs_update = True
        if needs_update:
            emis_prepped = emis_prepped.copy()
            try:
                emis_prepped[merge_on] = coerced
            except Exception:
                pass

    geom_prepped = base_geom
    try:
        geom_series = base_geom[merge_on]
    except Exception:
        geom_series = None
    if geom_series is not None:
        try:
            coerced_geom = _coerce_merge_key(geom_series, pad_len)
        except Exception:
            coerced_geom = geom_series
        needs_geom_update = False
        try:
            needs_geom_update = not geom_series.equals(coerced_geom)
        except Exception:
            needs_geom_update = True
        if copy_geometry or needs_geom_update:
            try:
                geom_prepped = base_geom.copy()
            except Exception:
                geom_prepped = base_geom
        if needs_geom_update:
            try:
                geom_prepped[merge_on] = coerced_geom
            except Exception:
                pass

    if not sort:
        merged = geom_prepped.merge(emis_prepped, on=merge_on, how='left', sort=False)
    else:
        merged = geom_prepped.merge(emis_prepped, on=merge_on, how='left')

    try:  # attach helper attrs for downstream consumers (stats, caches)
        merged.attrs['__prepared_emis'] = emis_prepped
        merged.attrs['__merge_key'] = merge_on
    except Exception:
        pass

    return merged, emis_prepped


def _file_signature(path: str) -> Tuple[str, Optional[int]]:
    if not isinstance(path, str):
        return (str(path), None)
    abs_path = os.path.abspath(path)
    try:
        return (abs_path, int(os.path.getmtime(abs_path)))
    except Exception:
        return (abs_path, None)

def _emit_user_message(notify: Optional[Callable[[str, str], None]], level: str, message: str) -> None:
    """Send a status message via callback when available, else print as fallback."""
    msg = str(message).strip()
    if not msg:
        return
    lvl = level.upper() if level else 'INFO'
    if callable(notify):
        try:
            notify(lvl, msg)
            return
        except Exception:
            pass
    print(msg)

def _check_column_in_df(df: pd.DataFrame, col_list: List[str], warn: bool = True) -> Optional[str]:
    # lower case all columns for case-insensitive matching, also trim whitespace
    col_list = [col.strip().lower() for col in col_list]
    for col in df.columns:
        if col.strip().lower() in col_list:
            return col
    #raise ValueError(f"None of {col_list} found in DataFrame.")
    if warn:
        print(f"Warning: None of {col_list} found in DataFrame.")
    return None

def remap_columns(df: pd.DataFrame, src_type: str):
    # Check for region, scc, pol, emis
    country_col = _check_column_in_df(df, country_cols, warn=False)
    tribal_col = _check_column_in_df(df, tribal_cols, warn=False)
    region_col = _check_column_in_df(df, region_cols)
    scc_col = _check_column_in_df(df, scc_cols)
    pol_col = _check_column_in_df(df, pol_cols)
    emis_col = _check_column_in_df(df, emis_cols)
    # Base mapping for columns common to both point and nonpoint
    col_maps = {
        country_col: 'country_cd',
        region_col: 'region_cd',
        tribal_col: 'tribal_code',
        scc_col: 'scc',
        pol_col: 'poll',
        emis_col: 'ann_value',
    }

    if src_type == 'ff10_nonpoint':
        col_lst = [country_col, region_col, tribal_col, scc_col, pol_col, emis_col]
        df = df[col_lst].rename(columns={icol: col_maps[icol] for icol in col_lst})
        return df, col_lst

    if src_type == 'ff10_point':
        facility_col = _check_column_in_df(df, facility_cols)
        unit_col = _check_column_in_df(df, unit_cols)
        rel_col = _check_column_in_df(df, rel_cols)
        lat_col = _check_column_in_df(df, lat_cols)
        lon_col = _check_column_in_df(df, lon_cols)
        col_maps.update({
            facility_col: 'facility_id',
            unit_col: 'unit_id',
            rel_col: 'rel_point_id',
            lat_col: 'latitude',
            lon_col: 'longitude'
        })
        col_lst = [country_col, region_col, tribal_col, facility_col, unit_col, rel_col, scc_col, pol_col, emis_col, lat_col, lon_col]
        df = df[col_lst].rename(columns={icol: col_maps[icol] for icol in col_lst})
        return df, col_lst

def _safe_pivot(df: pd.DataFrame, index_cols: List[str], pol_col: str, emis_col: str) -> pd.DataFrame:
    """
    Safely pivot a long DataFrame to wide format, avoiding OverflowError in pandas groupby
    when dealing with many grouping columns (high cardinality cartesian product).
    """
    # Create a single tuple key for the source columns to bypass multi-key groupby overflow
    source_key_col = '__source_key__'
    # Use MultiIndex to create tuples efficiently (vectorized C implementation)
    try:
        df[source_key_col] = pd.MultiIndex.from_frame(df[index_cols]).to_flat_index()
    except Exception:
        # Fallback to generator if MultiIndex fails
        df[source_key_col] = list(zip(*[df[c] for c in index_cols]))
    
    # Group by the single source key and pollutant
    # This reduces the grouping dimensionality to 2 (Source, Pollutant)
    df_grouped = df.groupby([source_key_col, pol_col], sort=False, as_index=False)[emis_col].sum()
    
    # Pivot to wide format
    df_wide = df_grouped.pivot(index=source_key_col, columns=pol_col, values=emis_col).fillna(0)
    
    # Restore the original columns from the index tuples
    # The index of df_wide is now the tuples.
    index_df = pd.DataFrame(df_wide.index.tolist(), columns=index_cols, index=df_wide.index)
    
    # Concatenate the index columns with the wide pollutant data
    result = pd.concat([index_df, df_wide], axis=1).reset_index(drop=True)
    result.columns.name = None
    
    return result

def _categorize_columns(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        try:
            if pd.api.types.is_categorical_dtype(series):
                continue
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                df[col] = series.astype('category')
        except Exception:
            continue


def _normalize_delim(d: Optional[str]) -> Optional[str]:
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
    }
    return mapping.get(low, d)

# Robust coordinate parsing: try common column names and coerce to float when possible
def _parse_coord(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == '' or s.upper() in ('NA', 'N/A', 'NONE'):
        return None
    # remove commas used as thousands separators
    s = s.replace(',', '')
    try:
        return float(s)
    except Exception:
        return None


def filter_dataframe_by_range(
    df: pd.DataFrame,
    column: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    """Return subset of ``df`` where ``column`` lies within [start, end] using digit-aware matching."""

    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("filter_dataframe_by_range expects a pandas DataFrame")
    if not (column and start is not None and end is not None):
        return df

    target = str(column).strip().lower()
    col_name = None
    for cand in df.columns:
        if str(cand).strip().lower() == target:
            col_name = cand
            break
    if col_name is None:
        logging.warning("Filter column %s not found in DataFrame; skipping range filter", column)
        return df

    start_str = str(start)
    end_str = str(end)
    width = max(len(start_str), len(end_str))
    if width <= 0:
        return df

    pattern = re.compile(rf"(\d{{{width}}})")

    def _extract(token) -> str:
        if token is None:
            return ''
        match = pattern.search(str(token).strip())
        return match.group(1) if match else ''

    extracted = df[col_name].apply(_extract)
    mask = extracted.apply(lambda val: start_str <= val <= end_str if val else False)
    if mask.all():
        return df
    filtered = df.loc[mask].copy()
    try:
        filtered.attrs = dict(getattr(df, 'attrs', {}) or {})
    except Exception:
        pass
    return filtered


def filter_dataframe_by_values(
    df: pd.DataFrame,
    column: Optional[str],
    values: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Return subset of ``df`` where ``column`` matches any value in ``values``."""

    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("filter_dataframe_by_values expects a pandas DataFrame")
    if not column or not values:
        return df

    normalized_tokens: List[str] = []
    digits_only = True
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        digits = re.sub(r"\D", "", text)
        if digits:
            normalized_tokens.append(digits)
        else:
            digits_only = False
            normalized_tokens.append(text.lower())
    if not normalized_tokens:
        return df

    target = str(column).strip().lower()
    col_name = None
    for cand in df.columns:
        if str(cand).strip().lower() == target:
            col_name = cand
            break
    if col_name is None:
        logging.warning("Filter column %s not found in DataFrame; skipping value filter", column)
        return df

    series = df[col_name]
    if not pd.api.types.is_string_dtype(series):
        series = series.astype('string')
    series = series.fillna('').str.strip()

    if digits_only and normalized_tokens:
        width = max(len(tok) for tok in normalized_tokens)
        normalized = [tok[-width:].zfill(width) for tok in normalized_tokens]
        series_cmp = series.str.replace(r"\D", "", regex=True).str[-width:].str.zfill(width)
    else:
        normalized = normalized_tokens
        series_cmp = series.str.lower()

    mask = series_cmp.isin(normalized)
    if mask.all():
        return df
    filtered = df.loc[mask].copy()
    try:
        filtered.attrs = dict(getattr(df, 'attrs', {}) or {})
    except Exception:
        pass
    return filtered

def read_inputfile(
    fpath: Union[str, Sequence[str]],
    sector: Optional[str] = None,
    delim: Optional[str] = ",",
    skiprows: Optional[int] = None,
    comment: Optional[str] = None,
    encoding: Optional[str] = None,
    header_last: bool = False,
    flter_col: Optional[str] = None,
    flter_start: Optional[str] = None,
    flter_end: Optional[str] = None,
    flter_val: Optional[Sequence[str]] = None,
    notify: Optional[Callable[[str, str], None]] = None,
):
    """Check file format identifier to select appropriate parser.
    Rules:
      * Check if 1st line of file contains "#FORMAT=FF10_NONPOINT" or "#FORMAT=FF10_POINT", if so, treat as FF10 format and call read_ff10().
      * If 1st line has #LIST, treat as list file of multiple FF10 files
      * else, treat as smkreport and call read_smkreport().
    """
    if isinstance(fpath, (list, tuple)):
        # Recursively read each file and concatenate
        dfs = []
        raw_dfs = []
        for p in fpath:
            p = p.strip()
            if not p:
                continue
            _emit_user_message(notify, 'INFO', f"Reading input file: {p} ...")
            try:
                d, r = read_inputfile(
                    p, sector, delim, skiprows, comment, encoding, header_last, 
                    flter_col, flter_start, flter_end, flter_val, notify
                )
                if d is not None:
                    dfs.append(d)
                if r is not None:
                    raw_dfs.append(r)
            except Exception as e:
                _emit_user_message(notify, 'ERROR', f"Failed reading {p}: {e}")
                # We continue with other files or fail? 
                # Let's fail hard if one fails to ensure consistency, 
                # or maybe just log. User usually wants all.
                raise e 
        
        if not dfs:
            return None, None
        
        # Concatenate
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_raw = pd.concat(raw_dfs, ignore_index=True) if raw_dfs else None
        
        # Merge attributes (prefer first file's attributes, or accumulate?)
        # For simple metadata like 'pollutants', we might want to re-detect.
        # But commonly preserved attrs might be source_type, etc.
        if dfs:
            # Copy attrs from the first dataframe
            for k, v in dfs[0].attrs.items():
                combined_df.attrs[k] = v
                
        return _normalize_input_result((combined_df, combined_raw))

    with open(fpath, 'r', errors='ignore') as f:
        first_line = f.readline().strip()
    if "#FORMAT=FF10_NONPOINT" in first_line:
        _emit_user_message(notify, 'INFO', "Detected FF10_NONPOINT format identifier in first line. Readding as FF10 format for nonpoint...")
        result = read_ff10(
            fpath=fpath,
            src_name=sector,
            flter_col=flter_col,
            flter_start=flter_start,
            flter_end=flter_end,
            flter_val=flter_val,
            delim=delim,
            skiprows=skiprows,
            comment=comment,
            encoding=encoding,
            emis_unit="tons/yr",
        )
    elif "#FORMAT=FF10_POINT" in first_line:
        _emit_user_message(notify, 'INFO', "Detected FF10_POINT format identifier in first line. Readding as FF10 format for point...")
        result = read_ff10(
            fpath=fpath,
            src_name=sector,
            flter_col=flter_col,
            flter_start=flter_start,
            flter_end=flter_end,
            flter_val=flter_val,
            delim=delim,
            skiprows=skiprows,
            comment=comment,
            encoding=encoding,
            emis_unit="tons/yr",
        )
    elif first_line.lstrip().startswith('#LIST'):
        _emit_user_message(notify, 'INFO', "Detected #LIST format identifier in first line. Readding as list file...")
        result = read_listfile(
            fpath=fpath,
            src_name=sector,
            flter_col=flter_col,
            flter_start=flter_start,
            flter_end=flter_end,
            flter_val=flter_val,
            delim=delim,
            encoding=encoding,
            notify=notify,
        )
    else:
        if "#Label" in first_line or "# County" in first_line:
             _emit_user_message(notify, 'INFO', "Detected SMKREPORT header in first line. Reading as SMKREPORT format...")
        else:
             _emit_user_message(notify, 'INFO', "Could not determine file format. Reading as SMKREPORT format...")
        result = read_smkreport(
            fpath=fpath,
            delim=delim,
            skiprows=skiprows,
            comment=comment,
            encoding=encoding,
            header_last=header_last,
        )
    return _normalize_input_result(result)


def _normalize_input_result(result):
    """Ensure read_inputfile always returns a tuple of (processed_df, raw_df)."""
    df = None
    raw_df = None
    if isinstance(result, tuple):
        if not result:
            return None, None
        df = result[0]
        if len(result) > 1:
            raw_df = result[1]
    else:
        df = result
    if isinstance(df, pd.DataFrame) and raw_df is None:
        try:
            attrs = getattr(df, 'attrs', {})
            if isinstance(attrs, dict):
                raw_df = attrs.get('raw_df')
        except Exception:
            raw_df = None
    return df, raw_df


def read_ff10(
    fpath: str,
    src_name: Optional[str] = None,
    member_of_list: bool = False,
    delim: Optional[str] = ",",
    skiprows: Optional[int] = None, 
    comment: Optional[str] = None,
    encoding: Optional[str] = None,
    emis_unit: Optional[str] = "tons/yr",
    flter_col: Optional[str] = None,
    flter_start: Optional[str] = None,
    flter_end: Optional[str] = None,
    flter_val: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    
    # Examine 1st row for "#FORMAT=FF10_NONPOINT" or "#FORMAT=FF10_POINT" and dertermine src_type
    with open(fpath, 'r', errors='ignore') as f:
        first_line = f.readline().strip()
        if "#FORMAT=FF10_NONPOINT" in first_line:
            src_type = "ff10_nonpoint"
        elif "#FORMAT=FF10_POINT" in first_line:
            src_type = "ff10_point"
        else:
            raise ValueError(f"File {fpath} does not have valid FF10 format identifier in first line.")

    # Optimization: Scan for header index without reading whole file into memory
    header_idx = None
    sep = _normalize_delim(delim)
    
    # If delimiter is not provided, sniff it from the first few lines
    if not sep:
        try:
            with open(fpath, 'r', encoding=encoding, errors='ignore') as f:
                sample = f.read(8192)
                sep = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|']).delimiter
        except Exception:
            sep = ','

    with open(fpath, 'r', encoding=encoding, errors='ignore') as f:
        for idx, line in enumerate(f):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            # Check for key headers
            lower_line = line.lower()
            if 'region_cd' in lower_line or 'region' in lower_line or 'fips' in lower_line:
                # Verify by splitting
                tokens = [t.strip() for t in line.split(sep)]
                lower_tokens = [t.lower() for t in tokens]
                if any(h in lower_tokens for h in ['country','country_cd','region_cd','region','scc', 'fips']):
                    header_idx = idx
                    break
    
    if header_idx is None:
        raise ValueError("Header row with 'region_cd' not found.")

    # Use pandas C engine for speed
    # Specify dtypes for common columns to save memory and avoid type inference overhead
    dtype_map = {
        'region_cd': 'string',
        'region': 'string',
        'fips': 'string',
        'scc': 'string',
        'poll': 'category',
        'country_cd': 'category',
        'tribal_code': 'category',
        'facility_id': 'string',
        'unit_id': 'string',
        'rel_point_id': 'string',
        'ann_value': 'float32',
        'emission': 'float32'
    }

    try:
        df = pd.read_csv(
            fpath,
            sep=sep,
            skiprows=header_idx,
            encoding=encoding,
            dtype=dtype_map,
            low_memory=False,
            comment='#'
        )
    except Exception as e:
        # Fallback to python engine if C engine fails (e.g. regex sep)
        logging.warning(f"Fast CSV read failed ({e}), falling back to slower python engine...")
        df = pd.read_csv(
            fpath,
            sep=sep,
            skiprows=header_idx,
            encoding=encoding,
            dtype=dtype_map,
            comment='#',
            engine='python'
        )

    # Build DataFrame, remapping columns to standard names
    # df is already a DataFrame
    df, col_lst = remap_columns(df, src_type)

    df = filter_dataframe_by_values(df, flter_col, flter_val)
    df = filter_dataframe_by_range(df, flter_col, flter_start, flter_end)

    # If this file is part of a list, return df for subsequent processing in list handler
    if member_of_list:
        return df
    
    # Find pol_col
    pol_col = _check_column_in_df(df, pol_cols)
    emis_col = _check_column_in_df(df, emis_cols)

    # Pivot to wide format
    # Use safe pivot to avoid OverflowError
    index_cols = [icol for icol in col_lst if icol not in (pol_col, emis_col)]
    df = _safe_pivot(df, index_cols, pol_col, emis_col)

    # Identify pollutant columns
    pollutant_cols = [col for col in df.columns if col not in col_lst]

    # Convert pollutant columns to numeric:
    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Preserve a copy of the parsed raw dataset for QA preview
    raw_df = df.copy()

    # Build FIPS columns
    df = get_emis_fips(df)

    wide = df.copy()

    # Get unit mapping for pollutant columns
    try:
        wide.attrs['units_map'] = {k: emis_unit for k in pollutant_cols}
    except Exception:
        pass
    
    # Get source name
    if src_name is not None:
        try:
            wide.attrs['source_name'] = src_name
        except Exception:
            pass
    # Get source type
    try:
        wide.attrs['source_type'] = src_type
    except Exception:
        pass

    return wide, raw_df

def _read_single_file_wrapper(fp, src_name, flter_col, flter_start, flter_end, flter_val, delim, encoding):
    """Wrapper for reading a single FF10 file in a worker process."""
    # Check type
    with open(fp, 'r', errors='ignore') as f:
        first_line = f.readline().strip()
    
    detected_type = None
    if "#FORMAT=FF10_NONPOINT" in first_line:
        detected_type = "ff10_nonpoint"
    elif "#FORMAT=FF10_POINT" in first_line:
        detected_type = "ff10_point"
    else:
        raise ValueError(f"File {fp} in LIST does not have valid FF10 format identifier.")
        
    df = read_ff10(
        fpath=fp,
        src_name=src_name,
        member_of_list=True,
        flter_col=flter_col,
        flter_start=flter_start,
        flter_end=flter_end,
        flter_val=flter_val,
        delim=delim,
        encoding=encoding,
    )
    return df, detected_type

def read_listfile(
    fpath: str,
    src_name: Optional[str] = None,
    emis_unit: Optional[str] = "tons/yr",
    delim: Optional[str] = ",", 
    encoding: Optional[str] = None,
    flter_col: Optional[str] = None,
    flter_start: Optional[str] = None,
    flter_end: Optional[str] = None,
    flter_val: Optional[Sequence[str]] = None,
    notify: Optional[Callable[[str, str], None]] = None,
) -> pd.DataFrame:

    _emit_user_message(notify, 'INFO', f'Reading LIST file: {fpath}')
    with open(fpath, 'r') as lf:
        filepaths = [line.strip() for line in lf if line.strip() and not line.startswith('#')]
    
    src_type = None
    df_list = []

    # Use parallel processing to read files
    # Determine number of workers (leave one core free, max 32)
    max_workers = min(32, max(1, multiprocessing.cpu_count() - 1))
    
    # If only 1 file or 1 worker, run sequentially to avoid overhead
    if len(filepaths) == 1 or max_workers <= 1:
        for fp in filepaths:
            _emit_user_message(notify, 'INFO', f'Reading file: {fp}')
            df_part, f_type = _read_single_file_wrapper(
                fp, src_name, flter_col, flter_start, flter_end, flter_val, delim, encoding
            )
            if src_type is None:
                src_type = f_type
            elif src_type != f_type:
                raise ValueError(f"File {fp} in LIST has inconsistent source type (expected {src_type}).")
            df_list.append(df_part)
    else:
        _emit_user_message(notify, 'INFO', f'Reading {len(filepaths)} files using {max_workers} parallel workers...')
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _read_single_file_wrapper, 
                    fp, src_name, flter_col, flter_start, flter_end, flter_val, delim, encoding
                ): fp for fp in filepaths
            }
            
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    df_part, f_type = future.result()
                    
                    if src_type is None:
                        src_type = f_type
                    elif src_type != f_type:
                         raise ValueError(f"File {fp} has inconsistent source type {f_type} (expected {src_type})")
                         
                    df_list.append(df_part)
                    # _emit_user_message(notify, 'INFO', f'Finished reading: {fp}') # Reduce spam in parallel mode
                except Exception as exc:
                    _emit_user_message(notify, 'ERROR', f'Error reading {fp}: {exc}')
                    raise exc
    
    # Combine all parts into one DataFrame
    if not df_list:
        raise ValueError("No valid data found in list file.")
        
    df = pd.concat(df_list, ignore_index=True)
    df, col_lst = remap_columns(df, src_type)

    df = filter_dataframe_by_values(df, flter_col, flter_val)
    df = filter_dataframe_by_range(df, flter_col, flter_start, flter_end)
    
    # Find pol_col
    pol_col = _check_column_in_df(df, pol_cols)
    emis_col = _check_column_in_df(df, emis_cols)

    # Pivot to wide format
    # Use safe pivot to avoid OverflowError
    index_cols = [icol for icol in col_lst if icol not in (pol_col, emis_col)]
    df = _safe_pivot(df, index_cols, pol_col, emis_col)

    # Identify pollutant columns
    pollutant_cols = [col for col in df.columns if col not in col_lst]
    # Convert pollutant columns to numeric:
    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Preserve a copy of the parsed raw dataset for QA preview
    raw_df = df.copy()

    # Build FIPS columns
    df = get_emis_fips(df)

    wide = df.copy()

    # Get unit mapping for pollutant columns
    try:
        wide.attrs['units_map'] = {k: emis_unit for k in pollutant_cols}
    except Exception:
        pass
    
    # Get source name
    if src_name is not None:
        try:
            wide.attrs['source_name'] = src_name
        except Exception:
            pass
    
    # Get source type
    try:
        wide.attrs['source_type'] = src_type
    except Exception:
        pass

    return wide, raw_df

def read_smkreport(
    fpath: str,
    delim: Optional[str] = ",",
    skiprows: Optional[int] = None, 
    comment: Optional[str] = None,
    encoding: Optional[str] = None, 
    header_last: bool = False,
) -> pd.DataFrame:
    """Parse an emissions report using a '#Label' or '# County' header line.
    Rules:
      * Check if 1st line of file contains "#FORMAT=FF10_NONPOINT" or "#FORMAT=FF10_POINT", if so, treat as FF10 format and apply special parsing (TBD).
      * Find the first line whose first non-space characters are '#Label' or '# County'.
      * That line (with the leading '#') defines the header; the line immediately above the header (if commented) is treated as a units row.
      * All following non-blank, non-commented lines are treated as data.
        Commented lines after the header are ignored (previously they were included).
        The comment marker defaults to '#', but can be overridden with the `comment` arg.
      * FIPS determination:
          - If a column named (case-insensitive) 'FIPS' exists -> normalize to 5-digit.
          - Else derive from the last 6 digits of a 'Region' column (case-insensitive):
              If length==6 and first char=='0' drop leading 0; else take last 5.
      * Pollutants = numeric columns excluding id-like columns (fips, region, state, county).
    """
    # Optimization: Scan file instead of reading all into memory
    open_kwargs = {'mode': 'r', 'errors': 'ignore'}
    if encoding:
        open_kwargs['encoding'] = encoding

    header_idx = -1
    header_line = ""
    units_line_raw = None
    
    with open(fpath, **open_kwargs) as f:
        prev_line = None
        for i, line in enumerate(f):
            if line.lstrip().startswith('#Label') or line.lstrip().startswith('# County'):
                header_idx = i
                header_line = line.lstrip('#').strip()
                if prev_line and prev_line.lstrip().startswith('#'):
                    units_line_raw = prev_line.lstrip('#').strip()
                if not header_last:
                    break
            prev_line = line
            
    comment_marker = comment if (comment is not None and comment != "") else '#'

    if header_idx == -1:
        # Fallback: Treat the first non-comment line as the header (generic CSV support)
        with open(fpath, **open_kwargs) as f:
            for i, line in enumerate(f):
                sline = line.strip()
                if sline and not sline.startswith(comment_marker):
                    header_idx = i
                    header_line = sline
                    break
        
        if header_idx == -1:
             raise ValueError("No valid header line found in file (neither '#Label'/'# County' nor generic CSV header).")
    
    # Delimiter selection
    sep = delim
    if sep is None:
        # Read sample lines for sniffing
        sample_lines = []
        with open(fpath, **open_kwargs) as f:
            # Skip to header
            for _ in range(header_idx + 1):
                next(f, None)
            for _ in range(50):
                line = next(f, None)
                if line:
                    if not line.strip() or line.lstrip().startswith(comment_marker):
                        continue
                    sample_lines.append(line)
                else:
                    break
        
        candidates = [";", "\t", "|", ",", " "]
        sample = [header_line] + sample_lines
        scores: Dict[str, float] = {}
        for c in candidates:
            counts = [s.count(c) for s in sample if s and not s.isspace()]
            if not counts:
                continue
            nonzero = [n for n in counts if n > 0]
            if not nonzero:
                continue
            coverage = len(nonzero) / max(1, len(counts))
            mean_cnt = sum(nonzero) / len(nonzero)
            variance = sum((x - mean_cnt)**2 for x in nonzero) / len(nonzero)
            stability = 1.0 / (1.0 + variance)
            penalty = 0.6 if c == ' ' else 1.0
            scores[c] = coverage * stability * mean_cnt * penalty
        if scores:
            best = max(scores.items(), key=lambda kv: kv[1])
            if best[1] > 0:
                sep = best[0]

    # Parse header columns
    splitter = sep if sep is not None else (';' if ';' in header_line else ',')
    col_names = [t.strip() for t in header_line.split(splitter)]

    # Choose the faster C engine when feasible
    engine = 'python' if (sep is None or sep == ' ') else 'c'
    
    try:
        df = pd.read_csv(
            fpath, 
            sep=sep, 
            engine=engine, 
            skipinitialspace=True,
            skiprows=header_idx + 1,
            names=col_names,
            comment=comment_marker,
            encoding=encoding,
            low_memory=False
        )
    except Exception:
        # Fallback
        df = pd.read_csv(
            fpath, 
            sep=sep, 
            engine='python', 
            skipinitialspace=True,
            skiprows=header_idx + 1,
            names=col_names,
            comment=comment_marker,
            encoding=encoding
        )

    # Attempt to capture units from the line ABOVE the header (if present)
    units_map: Dict[str, str] = {}
    try:
        if units_line_raw and header_line:
            splitter = sep if sep is not None else (';' if ';' in header_line else ',')
            header_tokens = [t.strip() for t in header_line.split(splitter)]
            unit_tokens = [t.strip() for t in units_line_raw.split(splitter)]
            if len(unit_tokens) < len(header_tokens):
                unit_tokens += [''] * (len(header_tokens) - len(unit_tokens))
            for name, utok in zip(header_tokens, unit_tokens):
                if not utok:
                    continue
                u = utok.strip()
                if u.startswith('[') and u.endswith(']'):
                    u = u[1:-1].strip()
                units_map[name] = u
    except Exception:
        units_map = {}

    # Trim whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    # Drop completely empty Unnamed:* columns (often created by trailing delimiters)
    try:
        df = df.loc[:, ~(df.columns.str.match(_UNNAMED_COLUMN_RE) & df.isna().all())]
    except Exception:
        pass
    # Re-key units map to match trimmed DataFrame column names
    if units_map:
        remapped: Dict[str, str] = {}
        for c in df.columns:
            u = units_map.get(c) or units_map.get(c.strip())
            if u:
                remapped[c] = u
        units_map = remapped

    # Capture the original columns from the input file BEFORE adding any derived fields
    # (e.g., FIPS, ROW/COL, GRID_RC). We'll use this to ensure Preview shows only original data.
    try:
        _original_input_columns: List[str] = list(df.columns)
    except Exception:
        _original_input_columns = []

    # Extract source name BEFORE coercing non-id columns to numeric
    source_name: Optional[str] = None
    try:
        # Standardize column name casing for lookup
        lower_map_pre = {c.lower(): c for c in df.columns}
        cand_names = ['#label', 'label', 'src', 'source', 'sourcename', 'source_name']
        label_col = None
        for nm in cand_names:
            if nm in lower_map_pre:
                label_col = lower_map_pre[nm]
                break
        if label_col is not None and label_col in df.columns:
            s = df[label_col]
            if not pd.api.types.is_string_dtype(s):
                s = s.astype(str)
            s = s.astype(str).str.strip()
            s = s[s.notna() & (s != '')]
            if not s.empty:
                try:
                    # Use the most frequent non-empty label; fallback to first
                    mode_vals = s.mode(dropna=True)
                    source_name = str(mode_vals.iloc[0]) if not mode_vals.empty else str(s.iloc[0])
                except Exception:
                    source_name = str(s.iloc[0])
    except Exception:
        source_name = None

    # Trim whitespace in string cells and coerce potential pollutant columns to numeric
    # Do not coerce common identifier/text columns to numeric (preserve for preview/QA)
    id_like_for_coerce = {
        'fips', 'region', 'state', 'county', 'scc', 'scc description',
        '#label', 'label', 'src', 'source', 'sourcename', 'source_name',
        'speciation prf', 'primary srg', 'fallbk srg','facility id', 'fac name'
    }
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
        if c.lower() not in id_like_for_coerce:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Standardize column name casing for lookup
    lower_map = {c.lower(): c for c in df.columns}

    # Fix for SMOKE reports with composite "County" column (e.g. "2022193 000000001003 Alabama Baldwin Co")
    if 'fips' not in lower_map and 'region_cd' not in lower_map:
        county_col = lower_map.get('county') or lower_map.get('# county') or lower_map.get('#county')
        if county_col:
            # Check if it looks like a composite column (contains spaces)
            sample = df[county_col].astype(str).iloc[0] if not df.empty else ''
            if ' ' in sample:
                try:
                    # Extract the second token (FIPS)
                    # Assuming format: Date FIPS State County...
                    extracted = df[county_col].astype(str).str.split(n=2).str[1]
                    # Clean to 6 digits (consistent with program-wide FIPS handling)
                    # Remove non-digits just in case
                    digits = extracted.str.replace(r'\D', '', regex=True)
                    # Take last 6 digits (or pad to 6)
                    df['FIPS'] = digits.str[-6:].str.zfill(6)
                except Exception:
                    pass

    df = get_emis_fips(df)
    
    if 'x cell' in lower_map and 'y cell' in lower_map:
        df['COL'] = df[lower_map['x cell']].astype(int)
        df['ROW'] = df[lower_map['y cell']].astype(int)
        df['GRID_RC'] = df['ROW'].astype(str) + '_' + df['COL'].astype(str)
    elif 'FIPS' not in df.columns or df['FIPS'].isna().all():
        raise ValueError("Report must contain FIPS (or derivable Region/County) or X/Y columns.")

    # Identify numeric pollutant columns (exclude obvious ID columns)
    id_like = {
        'fips', 'region', 'state', 'county', 'x cell', 'y cell', 'row', 'col', 'grid_rc',
        '#label', 'label', 'src', 'source', 'sourcename', 'source_name', 'scc', 'scc description',
        'facility id', 'fac name'
    }
    pollutant_cols = [
        c for c in df.columns
        if c.lower() not in id_like
        and not c.lower().startswith('unnamed')
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not pollutant_cols:
        raise ValueError('No numeric pollutant columns detected (after excluding id columns).')

    # Optimize memory for categorical columns
    _categorize_columns(df, ['scc', 'poll', 'country_cd', 'tribal_code', 'FIPS', 'region_cd', 'GRID_RC', 'state', 'county'])

    # Preserve a copy of the parsed raw dataset (pre-filter, pre-aggregation) for QA preview
    try:
        # Keep a reference to avoid duplicating memory
        raw_df_for_preview = df
    except Exception:
        raw_df_for_preview = df
    
    if 'FIPS' in df.columns:
        group_key = 'FIPS'
    elif 'GRID_RC' in df.columns:
        group_key = 'GRID_RC'
    else:
        # This case should not be reached due to earlier checks
        raise ValueError("Dataframe has neither FIPS nor GRID_RC for grouping.")

    wide = df[[group_key] + pollutant_cols].groupby(group_key, as_index=False, sort=False).sum()
    if units_map:
        try:
            # Keep only entries for pollutant columns
            wide.attrs['units_map'] = {k: v for k, v in units_map.items() if k in pollutant_cols}
        except Exception:
            pass
    # Attach source name if detected
    if source_name:
        try:
            wide.attrs['source_name'] = source_name
        except Exception:
            pass

    # Attach source type
    try:
        wide.attrs['source_type'] = 'smkreport'
    except Exception:
        pass

    # Attach raw dataframe for preview
    try:
        wide.attrs['raw_df'] = raw_df_for_preview
    except Exception:
        pass
    # Attach original input column names so Preview can hide derived columns
    try:
        wide.attrs['original_columns'] = _original_input_columns
    except Exception:
        pass
    _categorize_columns(wide, ['FIPS', 'region_cd', 'GRID_RC'])
    return wide


@_memoize(maxsize=6)
def _load_shapefile(path: str, get_fips: bool, _signature: Tuple[str, Optional[int]]) -> gpd.GeoDataFrame:
    """Read and normalize a counties shapefile, returning a GeoDataFrame."""
    del _signature  # used only to bust caches when the underlying file changes
    is_url = isinstance(path, str) and path.lower().startswith(("http://", "https://"))
    local_path = path
    tmp_to_cleanup: Optional[str] = None
    if is_url:
        try:
            import urllib.request, tempfile, ssl

            context = ssl._create_unverified_context()
            suffix = os.path.splitext(path)[1] or '.zip'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                with urllib.request.urlopen(path, context=context) as resp:
                    tmp.write(resp.read())
                local_path = tmp.name
                tmp_to_cleanup = local_path
        except Exception as exc:
            raise exc
    try:
        try:
            import pyogrio  # type: ignore  # noqa: F401

            gdf = gpd.read_file(local_path, engine="pyogrio")
        except Exception:
            gdf = gpd.read_file(local_path, engine='fiona')
        
        # Optimization: Simplify geometry to speed up plotting
        # Use a conservative tolerance (e.g. 0.005 degrees ~ 500m)
        if not gdf.empty:
            try:
                gdf['geometry'] = gdf.simplify(0.005, preserve_topology=True)
            except Exception:
                pass

    finally:
        if tmp_to_cleanup:
            try:
                os.unlink(tmp_to_cleanup)
            except Exception:
                pass

    if not get_fips:
        try:
            gdf = gdf.to_crs(4326)
        except Exception:
            pass
        return gdf

    # Check if geoid column exists
    geoid_col = _check_column_in_df(gdf, ['geoid'], warn=False)
    # Check if statefp and countyfp columns exist
    statefp_col = _check_column_in_df(gdf, ['statefp','state_cd','state_code'], warn=False)
    countyfp_col = _check_column_in_df(gdf, ['countyfp','county_cd','county_code'], warn=False)
    # Check if region_cd column exists
    region_col = _check_column_in_df(gdf, ['region_cd','regioncd'], warn=False)
    got_FIPS = False
    if geoid_col:
        lengths = gdf[geoid_col].astype('string').str.strip().str.len()
        #print(lengths.describe())  # min, max, etc.
        geoid_stats = lengths.describe()
        max_len = int(geoid_stats['max'])
        min_len = int(geoid_stats['min'])
        if min_len >= 5 and max_len <= 6:
            gdf['FIPS'] = gdf[geoid_col].astype(str).str.zfill(6)
            got_FIPS = True

    elif statefp_col and countyfp_col and not got_FIPS:
        gdf['FIPS'] = '0' + gdf[statefp_col].astype(str).str.zfill(2) + gdf[countyfp_col].astype(str).str.zfill(3)
        got_FIPS = True
    elif region_col and not got_FIPS:
        gdf['FIPS'] = gdf[region_col].astype(str).str.zfill(6)
        got_FIPS = True

    if not got_FIPS:
        raise ValueError("Could not identify GEOID or STATEFP/COUNTYFP in counties layer.")

    try:
        gdf = gdf.to_crs(4326)
    except Exception:
        pass

    if 'region_cd' not in gdf.columns and 'FIPS' in gdf.columns:
        try:
            gdf['region_cd'] = gdf['FIPS']
        except Exception:
            pass

    cols = ['FIPS', 'region_cd','geometry']
    result = gdf[cols].copy()
    _categorize_columns(result, ['FIPS', 'region_cd'])
    try:
        _ = result.sindex
    except Exception:
        pass
    return result


def read_shpfile(path: str, get_fips: bool = False) -> gpd.GeoDataFrame:
    gdf = _load_shapefile(path, get_fips, _file_signature(path))
    if _memory is None and not _CACHE_DISABLED:
        try:
            return gdf.copy()
        except Exception:
            pass
    return gdf


# ---- GRIDDESC parsing and shapefile creation functions ----
# (Adapted from griddesc2shp.py)

def _clean_name(raw: str) -> str:
    raw = raw.split('!')[0].strip()
    raw = _CLEAN_NAME_QUOTE_RE.sub('', raw)
    raw = _CLEAN_NAME_WHITESPACE_RE.sub(' ', raw)
    return raw.strip()

@_memoize(maxsize=8)
def _load_griddesc(path: str, _signature: Tuple[str, Optional[int]]) -> Tuple[Dict, Dict]:
    del _signature
    with open(path, 'r') as f:
        lines = f.readlines()
    try:
        sep_idx = next(i for i, ln in enumerate(lines) if "' '  !  end coords." in ln)
    except StopIteration as exc:
        raise ValueError("Missing coordinate block terminator (' '  !  end coords.)") from exc

    coords: Dict[str, List[float]] = {}
    i = 0
    while i < sep_idx:
        line = lines[i].strip()
        if line.startswith("'") and not line.startswith("!"):
            name = _clean_name(line)
            i += 1
            if i < sep_idx:
                params_line = lines[i].strip()
                tokens = [tok for tok in _GRIDDESC_SPLIT_RE.split(params_line) if tok]
                coords[name] = [float(tok.replace('D', 'E')) for tok in tokens]
        i += 1

    grids: Dict[str, List[float]] = {}
    i = sep_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("'") and not line.startswith("!") and line != "' '":
            gname = _clean_name(line)
            i += 1
            if i < len(lines):
                params_line = lines[i].strip()
                parts = [p.strip() for p in params_line.split(',') if p.strip()]
                if parts:
                    coord_ref = _clean_name(parts[0])
                    rest = [float(p.replace('D', 'E')) for p in parts[1:]]
                    grids[gname] = [coord_ref] + rest
        i += 1

    if not grids:
        raise ValueError("No grids found in the GRIDDESC file.")

    return coords, grids


def extract_grid(path: str, grid_id: str):
    coords, grids = _load_griddesc(path, _file_signature(path))

    if grid_id is None:
        return list(grids.keys())

    gid_clean = _clean_name(grid_id)
    if gid_clean not in grids:
        raise ValueError(f"Grid '{grid_id}' not found. Available: {', '.join(sorted(grids.keys()))}")

    grid_params = grids[gid_clean]
    coord_name = grid_params[0]
    if coord_name not in coords:
        raise ValueError(f"Projection '{coord_name}' referenced by grid '{gid_clean}' not defined in coords section.")

    return coords[coord_name], grid_params

def _generate_grid_chunk(row_indices, ncols, xorig, yorig, xcell, ycell, proj_str):
    """Worker function to generate a chunk of grid cells."""
    import pyproj
    from shapely.geometry import Polygon
    
    # Re-create transformer in worker
    lcc_proj = pyproj.Proj(proj_str)
    wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(lcc_proj, wgs84_proj, always_xy=True)
    
    features = []
    rows_attr = []
    cols_attr = []
    
    for r in row_indices:
        y0 = yorig + (r - 1) * ycell
        y1 = y0 + ycell
        for c in range(1, int(ncols) + 1):
            x0 = xorig + (c - 1) * xcell
            x1 = x0 + xcell
            pts_proj = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            pts_ll = [transformer.transform(px, py) for (px, py) in pts_proj]
            features.append(Polygon(pts_ll + [pts_ll[0]]))
            rows_attr.append(r)
            cols_attr.append(c)
            
    return features, rows_attr, cols_attr

@_memoize(maxsize=4)
def _create_domain_gdf_cached(
    griddesc_path: str,
    grid_name: str,
    full_grid: bool,
    _signature: Tuple[str, Optional[int]],
) -> gpd.GeoDataFrame:
    del _signature
    coord_params, grid_params = extract_grid(griddesc_path, grid_name)

    _, p_alpha, p_beta, p_gamma, x_cent, y_cent = coord_params
    _, xorig, yorig, xcell, ycell, ncols, nrows, _ = grid_params

    if USE_SPHERICAL_EARTH:
        proj_str = (
            f"+proj=lcc +lat_1={p_alpha} +lat_2={p_beta} +lat_0={y_cent} "
            f"+lon_0={x_cent} +a=6370000.0 +b=6370000.0 +x_0=0 +y_0=0 +units=m +no_defs"
        )
    else:
        proj_str = (
            f"+proj=lcc +lat_1={p_alpha} +lat_2={p_beta} +lat_0={y_cent} "
            f"+lon_0={x_cent} +ellps=WGS84 +datum=WGS84 +x_0=0 +y_0=0 +units=m +no_defs"
        )
    
    # Setup transformer for boundary calculation (still needed for full_grid=False)
    lcc_proj = pyproj.Proj(proj_str)
    wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(lcc_proj, wgs84_proj, always_xy=True)

    if not full_grid:
        ll_x, ll_y = xorig, yorig
        ur_x, ur_y = xorig + (ncols * xcell), yorig + (nrows * ycell)
        corners_proj = {
            'lower_left': (ll_x, ll_y),
            'lower_right': (ur_x, ll_y),
            'upper_right': (ur_x, ur_y),
            'upper_left': (ll_x, ur_y),
        }
        corners_latlon = {name: transformer.transform(x, y) for name, (x, y) in corners_proj.items()}
        polygon_geom = Polygon(
            [
                corners_latlon['lower_left'],
                corners_latlon['lower_right'],
                corners_latlon['upper_right'],
                corners_latlon['upper_left'],
                corners_latlon['lower_left'],
            ]
        )
        return gpd.GeoDataFrame({'name': [grid_name]}, geometry=[polygon_geom], crs='+proj=longlat +datum=WGS84')

    # Parallel grid generation
    features, rows_attr, cols_attr = [], [], []
    
    # Determine number of workers
    max_workers = min(32, max(1, multiprocessing.cpu_count() - 1))
    nrows_int = int(nrows)
    
    # If grid is small, run sequentially
    if nrows_int < 50 or max_workers <= 1:
        f, r, c = _generate_grid_chunk(range(1, nrows_int + 1), ncols, xorig, yorig, xcell, ycell, proj_str)
        features, rows_attr, cols_attr = f, r, c
    else:
        # Split rows into chunks
        chunk_size = max(1, nrows_int // max_workers)
        chunks = []
        for i in range(0, nrows_int, chunk_size):
            start = i + 1
            end = min(i + chunk_size + 1, nrows_int + 1)
            chunks.append(range(start, end))
            
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_generate_grid_chunk, chunk, ncols, xorig, yorig, xcell, ycell, proj_str)
                for chunk in chunks
            ]
            
            for future in as_completed(futures):
                f, r, c = future.result()
                features.extend(f)
                rows_attr.extend(r)
                cols_attr.extend(c)

    gdf = gpd.GeoDataFrame({'ROW': rows_attr, 'COL': cols_attr}, geometry=features, crs='+proj=longlat +datum=WGS84')
    gdf['GRID_RC'] = gdf['ROW'].astype(str) + '_' + gdf['COL'].astype(str)
    _categorize_columns(gdf, ['GRID_RC'])
    try:
        _ = gdf.sindex
    except Exception:
        pass
    return gdf


def create_domain_gdf(
    griddesc_path: str,
    grid_name: str,
    full_grid: bool = True,
) -> gpd.GeoDataFrame:
    gdf = _create_domain_gdf_cached(griddesc_path, grid_name, full_grid, _file_signature(griddesc_path))
    if _memory is None and not _CACHE_DISABLED:
        try:
            return gdf.copy()
        except Exception:
            pass
    return gdf

# ---- End of GRIDDESC functions ----

def _map_chunk(coords_chunk: pd.DataFrame, grid_subset: gpd.GeoDataFrame, lon_col: str, lat_col: str) -> pd.DataFrame:
    """Worker function for parallel spatial join."""
    pts = gpd.GeoDataFrame(
        coords_chunk,
        geometry=gpd.points_from_xy(coords_chunk[lon_col], coords_chunk[lat_col]),
        crs='EPSG:4326'
    )
    
    if grid_subset.crs is None:
        pass 
    
    try:
        if pts.crs != grid_subset.crs:
            pts = pts.to_crs(grid_subset.crs)
    except Exception:
        pass

    # Use sindex query
    try:
        sindex = grid_subset.sindex
        try:
            left_idx, right_idx = sindex.query_bulk(pts.geometry, predicate='within')
        except TypeError:
            left_idx, right_idx = sindex.query_bulk(pts.geometry)
        except AttributeError:
            left_idx = right_idx = None

        if left_idx is not None and len(left_idx) > 0:
            matches = pd.DataFrame({
                'point_ix': left_idx,
                'grid_ix': right_idx,
            }).sort_values(['point_ix', 'grid_ix'], kind='mergesort')
            matches = matches.drop_duplicates('point_ix', keep='first')
            
            assignments = coords_chunk.reset_index(drop=True)
            assignments['GRID_RC'] = pd.NA
            grid_values = grid_subset['GRID_RC'].to_numpy()
            
            # Map back using integer positions
            assignments.loc[matches['point_ix'].to_numpy(), 'GRID_RC'] = grid_values[matches['grid_ix'].to_numpy()]
            return assignments[[lon_col, lat_col, 'GRID_RC']].dropna()
    except Exception:
        pass
        
    # Fallback to sjoin
    try:
        joined = gpd.sjoin(pts, grid_subset, how='left', predicate='within')
    except TypeError:
        joined = gpd.sjoin(pts, grid_subset, how='left', op='within')
        
    return joined[[lon_col, lat_col, 'GRID_RC']].drop_duplicates().dropna()

def map_latlon2grd(emis_df: pd.DataFrame, base_geom: gpd.GeoDataFrame) -> pd.DataFrame:
    """Assign grid cells (GRID_RC) to FF10 point rows using their lon/lat values."""
    if not isinstance(emis_df, pd.DataFrame):
        raise TypeError('FF10 emissions must be provided as a DataFrame.')
    if base_geom is None or not isinstance(base_geom, gpd.GeoDataFrame):
        raise TypeError('Grid geometry must be provided as a GeoDataFrame.')
    if 'GRID_RC' not in base_geom.columns:
        raise ValueError("Grid geometry is missing required 'GRID_RC' column.")

    lon_candidates = ['longitude', 'lon', 'lon_dd']
    lat_candidates = ['latitude', 'lat', 'lat_dd']
    lon_col = next((c for c in emis_df.columns if c in lon_candidates or c.lower() in lon_candidates), None)
    lat_col = next((c for c in emis_df.columns if c in lat_candidates or c.lower() in lat_candidates), None)
    if not lon_col or not lat_col:
        raise ValueError('FF10 point data do not contain recognizable longitude/latitude columns.')

    # Prepare unique lon/lat pairs for the spatial join
    coords = emis_df[[lon_col, lat_col]].dropna().drop_duplicates()
    if coords.empty:
        raise ValueError('No valid longitude/latitude pairs found to map onto the grid.')

    grid_subset = base_geom[['GRID_RC', 'geometry']].dropna(subset=['GRID_RC']).copy()
    if grid_subset.empty:
        raise ValueError('Grid geometry does not contain any GRID_RC values.')
    
    # Ensure CRS alignment
    if grid_subset.crs is None:
        grid_subset.set_crs('EPSG:4326', inplace=True, allow_override=True)

    # Determine if parallel processing is beneficial
    # Threshold: > 50,000 points and > 1 worker available
    max_workers = min(32, max(1, multiprocessing.cpu_count() - 1))
    use_parallel = len(coords) > 50000 and max_workers > 1

    joined = None

    if use_parallel:
        logging.info(f"Mapping {len(coords)} unique locations to grid using {max_workers} workers...")
        # Split coords into chunks
        chunk_size = max(1, len(coords) // max_workers)
        chunks = [coords.iloc[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_map_chunk, chunk, grid_subset, lon_col, lat_col)
                for chunk in chunks
            ]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if not res.empty:
                        results.append(res)
                except Exception as e:
                    logging.warning(f"Parallel mapping chunk failed: {e}")
        
        if results:
            joined = pd.concat(results, ignore_index=True)
    
    # Fallback to sequential if parallel failed or not used
    if joined is None or joined.empty:
        if use_parallel:
            logging.info("Parallel mapping yielded no results or failed, falling back to sequential...")
        joined = _map_chunk(coords, grid_subset, lon_col, lat_col)

    mapped = emis_df.drop(columns=['GRID_RC'], errors='ignore').merge(
        joined,
        on=[lon_col, lat_col],
        how='left'
    )
    try:
        mapped.attrs = dict(getattr(emis_df, 'attrs', {}))
    except Exception:
        pass
    return mapped

def detect_pollutants(df: pd.DataFrame) -> List[str]:
    try:
        cached = df.attrs.get('_detected_pollutants')  # type: ignore[attr-defined]
        if isinstance(cached, (list, tuple)):
            return list(cached)
    except Exception:
        pass
    # Exclude grid identifiers as well
    id_like = {
        'fips', 'region', 'state', 'county', 'x', 'y', 'x cell', 'y cell', 'row', 'col', 'grid_rc',
        '#label', 'label', 'src', 'source', 'sourcename', 'source_name', 'scc', 'scc description',
        'region_cd', 'state_cd', 'county_cd',
        'facility id', 'fac name', 'country_cd', 'tribal_code', 'emis_type',
        'facility_id', 'unit_id', 'rel_point_id','latitude','longitude','lat_dd','lon_dd','lat','lon'
    }
    detected = [
        c for c in df.columns
        if c.lower() not in id_like
        and not c.lower().startswith('unnamed')
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    try:
        df.attrs['_detected_pollutants'] = tuple(detected)
    except Exception:
        pass
    return detected

def get_emis_fips(df: pd.DataFrame):

    # Find region_cd column if df
    region_col = _check_column_in_df(df, region_cols, warn=False)

    # Check if region_col less than 6 characters long, affix with country code
    if region_col:
        lengths = df[region_col].astype('string').str.strip().str.len()
        #print(lengths.describe())  # min, max, etc.
        region_stats = lengths.describe()
        max_len = int(region_stats['max'])
        min_len = int(region_stats['min'])

        if max_len < 6:
            # Locate country_cd column if exists
            country_col = _check_column_in_df(df, country_cols, warn=False)
            #print("Found country_cd column:", country_col)
            if country_col:
                print(f"Building FIPS by affixing country code from column {country_col} to region_cd column {region_col}")
                ## Map country code value from country_code_mappings dict to new column df[country_id]
                country_codes = (
                    df[country_col]
                    .astype('string')
                    .str.strip()
                    .str.upper()
                    .map(country_code_mappings)
                    .fillna('0')
            )
            else:
                country_codes = pd.Series('0', index=df.index)  # default to US
            # Affix region code with country code
            # Ensure robust string conversion (handle potential float/int types)
            s_region = df[region_col].astype(str)
            # If float dtype, convert to int first to avoid ".0" suffix
            if pd.api.types.is_float_dtype(df[region_col]):
                try:
                    s_region = df[region_col].fillna(-1).astype(int).astype(str).replace('-1', '')
                except Exception:
                    pass
            
            fips_suffix = s_region.str.strip().str.zfill(5)
            # Use concat to avoid PerformanceWarning: DataFrame is highly fragmented
            if 'FIPS' in df.columns:
                df = df.drop(columns=['FIPS'])
            df = pd.concat([df, (country_codes + fips_suffix).rename('FIPS')], axis=1)    
        else:
            # Ensure robust string conversion
            s_region = df[region_col].astype(str)
            if pd.api.types.is_float_dtype(df[region_col]):
                try:
                    s_region = df[region_col].fillna(-1).astype(int).astype(str).replace('-1', '')
                except Exception:
                    pass
            
            # Use concat to avoid PerformanceWarning
            if 'FIPS' in df.columns:
                df = df.drop(columns=['FIPS'])
            df = pd.concat([df, s_region.str.zfill(6).rename('FIPS')], axis=1)
        
        return df
    else:
        print("region_cd column not found, checking for statefp and countyfp columns to build FIPS")
        # Find if there is statefp and countyfp columns to build fips
        statefp_col = _check_column_in_df(df, ['statefp','state_cd','state_code'], warn=False)
        countyfp_col = _check_column_in_df(df, ['countyfp','county_cd','county_code'], warn=False)
        if statefp_col and countyfp_col:
            new_fips = '0' + df[statefp_col].astype(str).str.zfill(2) + df[countyfp_col].astype(str).str.zfill(3)
            if 'FIPS' in df.columns:
                df = df.drop(columns=['FIPS'])
            df = pd.concat([df, new_fips.rename('FIPS')], axis=1)

        return df
    
    raise ValueError("No valid FIPS code columns found")

