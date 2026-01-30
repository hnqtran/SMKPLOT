"""Data processing functions for SMKPLOT GUI.

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

import os, sys
import re
import warnings

_CLEAN_NAME_QUOTE_RE = re.compile(r"['`,]")
_CLEAN_NAME_WHITESPACE_RE = re.compile(r"\s+")
_GRIDDESC_SPLIT_RE = re.compile(r',\s*|\s+')
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

from config import (
    USE_SPHERICAL_EARTH,
    COUNTRY_COLS, TRIBAL_COLS, REGION_COLS, FACILITY_COLS, UNIT_COLS, REL_COLS,
    EMIS_COLS, SCC_COLS, POL_COLS, LAT_COLS, LON_COLS, COUNTRY_CODE_MAPPINGS,
    TRIBAL_TO_COUNTY_FIPS
)
from utils import normalize_delim, coerce_merge_key, is_netcdf_file

def _spatial_filter_chunk(gdf_chunk: gpd.GeoDataFrame, overlay: gpd.GeoDataFrame, mode: str) -> gpd.GeoDataFrame:
    """Helper for parallel spatial filtering."""
    if gdf_chunk.empty:
        return gdf_chunk
        
    if mode == 'clipped':
        try:
            return gpd.overlay(gdf_chunk, overlay, how='intersection', keep_geom_type=True)
        except TypeError:
            return gpd.overlay(gdf_chunk, overlay, how='intersection')
    elif mode == 'intersect':
        joined = gpd.sjoin(gdf_chunk, overlay, how='inner', predicate='intersects')
        if joined.empty:
            return gdf_chunk.iloc[:0]
        return gdf_chunk.loc[gdf_chunk.index.isin(joined.index)]
    elif mode == 'within':
        # Use BOTH centroid and representative_point to capture all valid features.
        # Centroid is robust for general shapes; Representative Point handles concave/coastal (zigzag) shapes.
        valid_indices = set()
        
        # Check 1: Centroid
        temp = gdf_chunk[['geometry']].copy()
        if temp.crs and temp.crs.is_geographic:
            try:
                c_proj = temp['geometry'].to_crs('EPSG:3857').centroid
                temp['geometry'] = c_proj.to_crs(temp.crs)
            except Exception:
                temp['geometry'] = temp.geometry.centroid
        else:
             temp['geometry'] = temp.geometry.centroid
             
        joined = gpd.sjoin(temp, overlay, how='inner', predicate='intersects')
        valid_indices.update(joined.index.tolist())
        
        # Check 2: Representative Point
        temp = gdf_chunk[['geometry']].copy() 
        if temp.crs and temp.crs.is_geographic:
            try:
                c_proj = temp['geometry'].to_crs('EPSG:3857').representative_point()
                temp['geometry'] = c_proj.to_crs(temp.crs)
            except Exception:
                temp['geometry'] = temp.geometry.representative_point()
        else:
             temp['geometry'] = temp.geometry.representative_point()
             
        joined = gpd.sjoin(temp, overlay, how='inner', predicate='intersects')
        valid_indices.update(joined.index.tolist())

        if not valid_indices:
            return gdf_chunk.iloc[:0]
        return gdf_chunk.loc[sorted(list(valid_indices))]
    return gdf_chunk


def apply_spatial_filter(gdf: gpd.GeoDataFrame, overlay: gpd.GeoDataFrame, mode: str = 'intersect') -> gpd.GeoDataFrame:
    """
    Apply a spatial filter to a GeoDataFrame using an overlay geometry.
    
    Args:
        gdf: Source GeoDataFrame to filter.
        overlay: Overlay geometry to filter against.
        mode: Filtering mode:
                - 'clipped': Geometrically clip features (intersection).
                - 'intersect': Keep features that intersect the overlay (Spatial Join).
                - 'within': Keep features entirely within the overlay (Spatial Join).
    
    Returns:
        Filtered GeoDataFrame.
    """
    if gdf is None:
        return gdf
    if gdf.empty:
        return gdf

    # Consolidate DataFrame to avoid fragmentation warnings (PerformanceWarning)
    # This is important for pivotted CSVs with many pollutant columns.
    gdf = gdf.copy()

    try:
        if gdf.crs and overlay.crs and (gdf.crs != overlay.crs):
            overlay = overlay.to_crs(gdf.crs)
    except Exception:
        pass  # CRS conversion failed, proceed with original CRS

    # Sequential processing is verified as efficient for standard datasets.
    logging.info(f"Applying spatial filter '{mode}' on {len(gdf)} features...")
    return _spatial_filter_chunk(gdf, overlay, mode)




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
            grouped = emis_df.groupby(merge_on, dropna=False, sort=sort, observed=False)
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
            coerced = coerce_merge_key(emis_prepped[merge_on], pad_len)
        except Exception:
            coerced = emis_prepped[merge_on]
        needs_update = False
        try:
            needs_update = not emis_prepped[merge_on].equals(coerced)
        except Exception:
            needs_update = True
        if needs_update:
            if emis_prepped is emis_df:
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
            coerced_geom = coerce_merge_key(geom_series, pad_len)
        except Exception:
            coerced_geom = geom_series
        needs_geom_update = False
        try:
            needs_geom_update = not geom_series.equals(coerced_geom)
        except Exception:
            needs_geom_update = True
        if copy_geometry:
            geom_prepped = base_geom.copy()
        elif needs_geom_update:
            if geom_prepped is base_geom:
                geom_prepped = base_geom.copy()
        
        if needs_geom_update:
            try:
                geom_prepped[merge_on] = coerced_geom
            except Exception:
                pass

    # Handle geometry conflict BEFORE merge to avoid geometry_x/geometry_y
    # Determine which geometry to keep.
    # Default: Keep base_geom (left) geometry to ensure full map coverage.
    # Exception: If base_geom lacks CRS (e.g. raw indices) and emis_prepped has CRS (e.g. Inline Lazy), prefer emis.
    
    left_has_geom = isinstance(geom_prepped, gpd.GeoDataFrame) and 'geometry' in geom_prepped.columns
    right_has_geom = isinstance(emis_prepped, gpd.GeoDataFrame) and 'geometry' in emis_prepped.columns
    
    if left_has_geom and right_has_geom:
        keep_right = False
         
        # Check optimization flags
        is_lazy = False
        try:
            st = emis_prepped.attrs.get('source_type')
            if st and str(st) in ('gridded_lazy', 'inline_point_lazy'):
                is_lazy = True
        except Exception:
            pass

        # Check CRS
        if geom_prepped.crs is None and emis_prepped.crs is not None:
                keep_right = True
        elif is_lazy:
                keep_right = True
         
        if keep_right:
                # Drop geometry from left (convert to DataFrame)
                geom_prepped = pd.DataFrame(geom_prepped).drop(columns=['geometry'])
        else:
                # Drop geometry from right
                emis_prepped = pd.DataFrame(emis_prepped).drop(columns=['geometry'])

    if not sort:
        merged = geom_prepped.merge(emis_prepped, on=merge_on, how='left', sort=False)
    else:
        merged = geom_prepped.merge(emis_prepped, on=merge_on, how='left')
    
    # If merging a DF and GDF, ensure the final result is a GeoDataFrame if geometry exists.
    if 'geometry' in merged.columns and not isinstance(merged, gpd.GeoDataFrame):
        merged = gpd.GeoDataFrame(merged, geometry='geometry')
        # Restore CRS if needed
        if isinstance(emis_prepped, gpd.GeoDataFrame) and emis_prepped.crs:
            if merged.crs is None: merged.crs = emis_prepped.crs
    
    try:  # attach helper attrs for downstream consumers (stats, caches)
        merged.attrs['__prepared_emis'] = emis_prepped
        merged.attrs['__merge_key'] = merge_on
        
        # Propagate metadata attrs from emissions dataframe
        src_attrs = getattr(emis_df, 'attrs', {})
        for k in ['stack_groups_path', 'proxy_ncf_path', 'original_ncf_path', 'source_type']:
            if k in src_attrs:
                merged.attrs[k] = src_attrs[k]
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
    # Check for region, scc, pol, emis using imported constants
    country_col = _check_column_in_df(df, COUNTRY_COLS, warn=False)
    tribal_col = _check_column_in_df(df, TRIBAL_COLS, warn=False)
    region_col = _check_column_in_df(df, REGION_COLS)
    scc_col = _check_column_in_df(df, SCC_COLS)
    pol_col = _check_column_in_df(df, POL_COLS)
    emis_col = _check_column_in_df(df, EMIS_COLS)
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
        # Rename standard columns but keep others
        rename_map = {icol: col_maps[icol] for icol in col_lst if icol is not None}
        df = df.rename(columns=rename_map)
        
        # Update col_lst to reflect renamed standard columns and include all ORIGINAL extra columns
        # Note: 'col_lst' here effectively becomes the "schema" of the dataframe, 
        # so we need to return the list of columns that SHOULD act as keys + value.
        
        # Helper: get the new name if renamed, else old name
        def _get_new_name(old_name):
            return rename_map.get(old_name, old_name)
        
        # Reconstruct full column list to ensure processing preserves all input columns.
        full_col_lst = list(df.columns)
        return df, full_col_lst

    if src_type == 'ff10_point':
        facility_col = _check_column_in_df(df, FACILITY_COLS)
        unit_col = _check_column_in_df(df, UNIT_COLS)
        rel_col = _check_column_in_df(df, REL_COLS)
        lat_col = _check_column_in_df(df, LAT_COLS)
        lon_col = _check_column_in_df(df, LON_COLS)
        
        col_maps.update({
            facility_col: 'facility_id',
            unit_col: 'unit_id',
            rel_col: 'rel_point_id',
            lat_col: 'latitude',
            lon_col: 'longitude'
        })
        
        target_cols = [country_col, region_col, tribal_col, facility_col, unit_col, rel_col, scc_col, pol_col, emis_col, lat_col, lon_col]
        # Rename standard columns but keep others
        rename_map = {icol: col_maps[icol] for icol in target_cols if icol is not None}
        df = df.rename(columns=rename_map)
        
        full_col_lst = list(df.columns)
        return df, full_col_lst

def _safe_pivot(df: pd.DataFrame, index_cols: List[str], pol_col: str, emis_col: str) -> pd.DataFrame:
    """
    Safely pivot a long DataFrame to wide format, avoiding "Product space too large"
    errors in pandas when dealing with high cardinality cartesian products.
    """
    try:
        # Capture original dtypes to restore them later
        original_dtypes = df[index_cols].dtypes

        # Pre-aggregate to reduce duplicate index combinations before pivoting
        # This handles cases where multiple rows map to the same (index_cols + pol_col)
        # IMPORTANT: dropna=False to preserve rows with NaN in index columns (e.g. tribal_code)
        df_agg = df.groupby(index_cols + [pol_col], observed=True, as_index=False, dropna=False)[emis_col].sum()
        
        # Create a single surrogate key for the index columns
        if len(index_cols) > 1:
            surrogate_col = "_temp_pivot_idx_key_"
            try:
                # Optimized tuple creation
                df_agg[surrogate_col] = pd.MultiIndex.from_frame(df_agg[index_cols]).to_flat_index()
            except Exception:
                # Fallback
                df_agg[surrogate_col] = list(zip(*[df_agg[c] for c in index_cols]))
            
            # Pivot using the surrogate key
            pivot = df_agg.pivot(index=surrogate_col, columns=pol_col, values=emis_col).fillna(0)
            
            # Reconstruct the original index columns from the surrogate key
            index_tuples = pivot.index
            
            if len(index_tuples) > 0:
                # Rebuild key DataFrame
                key_df = pd.DataFrame(index_tuples.tolist(), columns=index_cols, index=pivot.index)
                
                # Restore original dtypes
                for col in index_cols:
                    if original_dtypes[col] != key_df[col].dtype:
                        try:
                            key_df[col] = key_df[col].astype(original_dtypes[col])
                        except Exception:
                            pass # Keep inferred if conversion fails

                # Concatenate with the pivoted data
                result = pd.concat([key_df, pivot], axis=1).reset_index(drop=True)
            else:
                # If result is empty, we must still preserve the schema (columns)
                # pivot.reset_index() would lose index_cols
                # Create empty DF with correct columns
                cols = index_cols + list(pivot.columns)
                result = pd.DataFrame(columns=cols)
                # Restore dtypes
                for col in index_cols:
                    try:
                        result[col] = result[col].astype(original_dtypes[col])
                    except Exception:
                        pass
                
        else:
            # Simple case: only 1 index column
            pivot = df_agg.pivot(index=index_cols[0], columns=pol_col, values=emis_col).fillna(0)
            result = pivot.reset_index()
             
            # If empty, ensure column name is restored from index name
            if result.empty and index_cols[0] not in result.columns:
                result[index_cols[0]] = []
             
            # Restore dtype for single column if needed
            col0 = index_cols[0]
            if col0 in result.columns and original_dtypes[col0] != result[col0].dtype:
                try:
                    result[col0] = result[col0].astype(original_dtypes[col0])
                except Exception:
                    pass

        result.columns.name = None
        return result

    except Exception as e:
        logging.getLogger().info(f"Optimized pivot failed ({e}); falling back to standard pivot_table.")
        try:
            return df.pivot_table(
                index=index_cols,
                columns=pol_col,
                values=emis_col,
                aggfunc='sum',
                fill_value=0,
                dropna=False,
                observed=True
            ).reset_index().rename_axis(None, axis=1)
        except Exception as e2:
            logging.error(f"Fallback pivot also failed: {e2}")
            raise e2


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
    return_raw: bool = True,
    ncf_params: Optional[Dict[str, Any]] = None,
    lazy: bool = False,
):
    """Check file format identifier to select appropriate parser.
    Rules:
        * Check if 1st line of file contains "#FORMAT=FF10_NONPOINT" or "#FORMAT=FF10_POINT", if so, treat as FF10 format and call read_ff10().
        * If 1st line has #LIST, treat as list file of multiple FF10 files
        * else, treat as smkreport and call read_smkreport().
    """
    # NetCDF Support: Improved detection using magic numbers
    is_ncf = False
    if isinstance(fpath, str):
        if is_netcdf_file(fpath):
            is_ncf = True
        elif fpath.lower().endswith(('.ncf', '.nc')):
            # Fallback for empty/unreadable files with ncf extension
            is_ncf = True

    if is_ncf:
        try:
            # Lazy import to avoid circular dependency
            from ncf_processing import read_ncf_emissions
            _emit_user_message(notify, 'INFO', "Detected NetCDF binary format. Reading as IOAPI NetCDF...")
            
            # Prepare optional args
            kwargs = {}
            if ncf_params:
                kwargs.update(ncf_params)
            
            # We don't support filtering inside read_ncf yet, but we can filter after.
            result_df = read_ncf_emissions(fpath, notify=notify, lazy=lazy, **kwargs)
            
            # Apply sector/source_type metadata if possible
            if sector:
                result_df.attrs['source_name'] = sector
            result_df.attrs['source_type'] = 'gridded_netcdf'
            
            # Filtering if requested (post-read)
            # Currently not implemented for NetCDF inputs (typically wide format)
            # filtered_df = result_df

            # For NCF, we treat the result as both processed and raw
            if not return_raw:
                return result_df, None
            return _normalize_input_result((result_df, result_df))
        except Exception as e:
            _emit_user_message(notify, 'ERROR', f"Failed reading NetCDF {fpath}: {e}")
            raise e

    if isinstance(fpath, (list, tuple)):
        # Recursively read each file and concatenate
        dfs = []
        raw_dfs = []
        
        # Parallelize reading if multiple files provided
        # Use simple heuristic: if > 1 file, use parallel workers
        paths = [p.strip() for p in fpath if p.strip()]
        
        if not paths:
            return None, None
            
        max_workers = min(32, max(1, multiprocessing.cpu_count() - 1))
        
        if len(paths) == 1 or max_workers <= 1:
            for p in paths:
                _emit_user_message(notify, 'INFO', f"Reading input file: {p} ...")
                try:
                    d, r = read_inputfile(
                        p, sector, delim, skiprows, comment, encoding, header_last, 
                        flter_col, flter_start, flter_end, flter_val, notify,
                        return_raw=return_raw,
                        lazy=lazy
                    )
                    if d is not None:
                        dfs.append(d)
                    if r is not None and return_raw:
                        raw_dfs.append(r)
                except Exception as e:
                    _emit_user_message(notify, 'ERROR', f"Failed reading {p}: {e}")
                    raise e
        else:
            _emit_user_message(notify, 'INFO', f"Reading {len(paths)} files using {max_workers} parallel workers...")
            # Since read_inputfile is not a simple function (it's recursive), we need a wrapper
            # But passing 'notify' (callback) to a subprocess is tricky/impossible if it's not picklable.
            # We will pass notify=None to workers.
             
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # We need to wrap the arguments
                futures = {}
                for p in paths:
                    fut = executor.submit(
                        read_inputfile,
                        p, sector, delim, skiprows, comment, encoding, header_last,
                        flter_col, flter_start, flter_end, flter_val, None, # notify cannot be passed
                        return_raw,
                        None, # ncf_params
                        lazy
                    )
                    futures[fut] = p
                
                for fut in as_completed(futures):
                    p = futures[fut]
                    try:
                        d, r = fut.result()
                        if d is not None:
                            dfs.append(d)
                        if r is not None and return_raw:
                            raw_dfs.append(r)
                    except Exception as e:
                        _emit_user_message(notify, 'ERROR', f"Failed reading {p}: {e}")
                        raise e

        if not dfs:
            return None, None
        
        # Capture attrs from the first dataframe before clearing them
        # This checks for metadata equality which can fail if attrs contain DataFrames
        saved_attrs = {}
        if dfs:
            saved_attrs = dict(dfs[0].attrs)
            for d in dfs:
                d.attrs = {}
        
        # Concatenate
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Handle raw_dfs similarly if present
        if raw_dfs:
            for r in raw_dfs:
                r.attrs = {}
            combined_raw = pd.concat(raw_dfs, ignore_index=True)
        else:
            combined_raw = None
        
        # Restore attributes to the combined dataframe
        combined_df.attrs = saved_attrs
        
        if not return_raw:
            return combined_df, None
            
        return _normalize_input_result((combined_df, combined_raw))

    # Optimized path for already-processed CSVs (e.g., from export-csv)
    # If the file path indicates it's a pivoted output or contains known columns, skip heavy processing.
    if isinstance(fpath, str) and (fpath.endswith('.csv') or fpath.endswith('.txt')):
        try:
            # Peek at columns
            peek_df = pd.read_csv(fpath, nrows=5, sep=delim or ',', comment=comment or '#')
            cols_upper = [c.upper() for c in peek_df.columns]
             
            # Heuristic: If we have GRID_RC or FIPS, and it looks like a pivoted file, load directly.
            if 'GRID_RC' in cols_upper or 'FIPS' in cols_upper:
                _emit_user_message(notify, 'INFO', "Detected pre-processed CSV with key columns (GRID_RC/FIPS). Loading directly...")
                # Ensure FIPS and other potential identifiers are read as strings to preserve formatting
                # We use the peek_df to identify object-like columns or specific known identifiers
                from config import key_cols
                dtype_map = {'FIPS': object, 'GRID_RC': object}
                  
                # Add all configured key columns to dtype map to ensure they are read as strings
                if key_cols:
                    for kc in key_cols:
                        dtype_map[kc] = object
                  
                # Inspect peek_df to find other object columns and preserve them
                for col in peek_df.columns:
                        if peek_df[col].dtype == 'object':
                            dtype_map[col] = object
                  
                df = pd.read_csv(fpath, sep=delim or ',', comment=comment or '#', low_memory=False, dtype=dtype_map)
                  
                # Flag as pre-processed so batch logic knows to filter pollutants on re-export
                df.attrs['is_preprocessed'] = True
                  
                # Restore attributes if possible (infer from columns)
                if sector:
                        df.attrs['source_name'] = sector
                  
                # Detect pollutants (numeric columns that aren't metadata)
                # We rely on configured metadata cols to exclude them
                from config import (
                        COUNTRY_COLS, TRIBAL_COLS, REGION_COLS, FACILITY_COLS, 
                        UNIT_COLS, REL_COLS, SCC_COLS, POL_COLS, LAT_COLS, LON_COLS
                )
                all_meta = set(COUNTRY_COLS + TRIBAL_COLS + REGION_COLS + FACILITY_COLS + 
                                UNIT_COLS + REL_COLS + SCC_COLS + POL_COLS + LAT_COLS + LON_COLS + 
                                ['GRID_RC', 'FIPS', 'X', 'Y', 'ROW', 'COL'])
                  
                detected_pols = []
                for c in df.columns:
                        if c.upper() not in [x.upper() for x in all_meta] and pd.api.types.is_numeric_dtype(df[c]):
                            detected_pols.append(c)
                  
                if detected_pols:
                        df.attrs['_detected_pollutants'] = tuple(detected_pols)

                logging.info(f"Loaded DataFrame shape: {df.shape}")
                logging.info(f"Filtering request: col={flter_col}, val_count={len(flter_val) if flter_val else 0}")

                # Apply filtering if requested (since we skipped the chunk processor)
                if flter_col:
                        df = filter_dataframe_by_values(df, flter_col, flter_val)
                        logging.info(f"After value filter: {df.shape}")
                        df = filter_dataframe_by_range(df, flter_col, flter_start, flter_end)
                        logging.info(f"After range filter: {df.shape}")

                if not return_raw:
                        return df, None
                return df, df.copy()
        except Exception:
            # Fallthrough to standard logic if peek fails
            pass

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
            flter_col=flter_col,
            flter_start=flter_start,
            flter_end=flter_end,
            flter_val=flter_val,
        )
    
    if not return_raw:
        if isinstance(result, tuple):
            return result[0], None
        return result, None

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


def _read_and_process_chunks(
    read_func: Callable, 
    read_kwargs: Dict[str, Any], 
    process_chunk_func: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """Read CSV in chunks and apply processing per chunk to reduce memory usage."""
    # Use a reasonable chunk size (e.g. 50k rows)
    chunk_size = 50000
    read_kwargs['chunksize'] = chunk_size
    
    chunks = []
    try:
        reader = read_func(**read_kwargs)
        for chunk in reader:
            if chunk.empty: continue
            
            # Apply processing (renaming, filtering)
            try:
                processed = process_chunk_func(chunk)
            except Exception:
                processed = chunk
            
            if not processed.empty:
                chunks.append(processed)
                
    except Exception as e:
        # If chunking fails (sometimes engines struggle), fall back to full read
        if 'chunksize' in read_kwargs:
            del read_kwargs['chunksize']
        full_df = read_func(**read_kwargs)
        return process_chunk_func(full_df)

    if not chunks:
        # Return empty DF with correct columns if possible? 
        # Hard to know columns without reading.
        # Fallback to reading 1 row?
        if 'chunksize' in read_kwargs: del read_kwargs['chunksize']
        read_kwargs['nrows'] = 0
        return read_func(**read_kwargs)

    return pd.concat(chunks, ignore_index=True)

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
    
    # Normalize passed delimiter
    passed_sep = normalize_delim(delim)
    # If passed_sep is just default comma, we might want to allow autodetection if implicit
    # But for now let's respect it if provided.
    
    detected_sep = passed_sep
    header_line = None

    with open(fpath, 'r', encoding=encoding, errors='ignore') as f:
        for idx, line in enumerate(f):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            
            # Check for key headers
            lower_line = line.lower()
            
            # Heuristic keywords for FF10/Reports
            if any(k in lower_line for k in ['region_cd', 'region', 'fips', 'state', 'country_cd']):
                
                # If delimiter not forced, try to deduce from this line
                current_sep = detected_sep
                if not current_sep:
                    # simplistic vote
                    if line.count(',') > line.count(';'):
                        current_sep = ','
                    elif line.count(';') > line.count(','):
                        current_sep = ';'
                    else:
                        current_sep = ','
                
                # Verify split
                # removing quotes for check
                clean_line = line.replace('"', '').replace("'", "")
                tokens = [t.strip().lower() for t in clean_line.split(current_sep)]
                
                if any(h in tokens for h in ['country','country_cd','region_cd','region','scc', 'fips']):
                    header_idx = idx
                    header_line = line
                    if not detected_sep:
                        detected_sep = current_sep
                    break
    
    if header_idx is None:
        # Fallback: maybe just assume row 0 if not identified (unlikely for FF10)
        # raising error is safer
        raise ValueError("Header row with valid columns (region_cd/fips/country) not found.")
        
    sep = detected_sep or ','

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

    # Identify usecols based on key_cols to optimize reading
    from config import (
        COUNTRY_COLS, TRIBAL_COLS, REGION_COLS, FACILITY_COLS, 
        UNIT_COLS, REL_COLS, SCC_COLS, POL_COLS, EMIS_COLS, LAT_COLS, LON_COLS,
        key_cols
    )
    
    
    # Get actual columns from header line (captured during scan)
    # We re-parse to ensure we get exact column names (case-sensitive) for usecols matches
    try:
        clean_header = header_line.replace('"', '').replace("'", "").strip()
        file_cols = [t.strip() for t in clean_header.split(sep)]
    except Exception:
        file_cols = []

    use_cols = []
    if file_cols:
        # Define sets of interesting columns aliases
        # We need to map key_cols (standard names) to potential file aliases
        # Construct lookup: Standard -> Set(Aliases)
        alias_map = {
            'country_cd': set(COUNTRY_COLS),
            'tribal_code': set(TRIBAL_COLS),
            'region_cd': set(REGION_COLS),
            'facility_id': set(FACILITY_COLS),
            'unit_id': set(UNIT_COLS),
            'rel_point_id': set(REL_COLS),
            'scc': set(SCC_COLS),
            'latitude': set(LAT_COLS),
            'longitude': set(LON_COLS)
        }
        
        # Determine which standard columns we need
        # Always need Pollutant and Emissions
        needed_standards = set(['pollutant', 'emission']) # Internal placeholders
        
        if key_cols:
            needed_standards.update(key_cols)
        else:
            # Default if no key_cols provided
            needed_standards.update(['region_cd', 'scc', 'country_cd'])

        # Iterate file columns and decide to keep
        for col in file_cols:
            c_low = col.lower()
            keep = False
            
            # check against pollutant/emission (always keep)
            if c_low in POL_COLS or c_low in EMIS_COLS:
                keep = True
            else:
                # Check against needed keys
                for std_key in needed_standards:
                    aliases = alias_map.get(std_key, set())
                    # Also match exact name if it's already standard
                    if std_key == c_low or c_low in [a.lower() for a in aliases]:
                        keep = True
                        break
            
            if keep:
                use_cols.append(col)

    # Sanity check: If we failed to find any emission or pollutant columns, 
    # the alias mapping might have failed. Fall back to reading all columns.
    has_emis = any(c.lower() in EMIS_COLS for c in use_cols)
    has_pol = any(c.lower() in POL_COLS for c in use_cols)
    
    if not (has_emis and has_pol) and use_cols:
        # _emit_user_message not available here easily (inside worker?)
        # Just reset use_cols to empty to force full read
        use_cols = []

    read_kwargs = {
        'filepath_or_buffer': fpath,
        'sep': sep,
        'skiprows': header_idx,
        'encoding': encoding,
        'dtype': dtype_map,
        'low_memory': False,
        'comment': '#'
    }
    
    if use_cols:
        read_kwargs['usecols'] = use_cols
        # Remove dtypes for columns we are NOT reading to prevent error
        clean_dtypes = {k: v for k, v in dtype_map.items() if k in use_cols}
        read_kwargs['dtype'] = clean_dtypes
    
    # Capture necessary context for the chunk processor
    # We need to extract the processed column list from at least one chunk
    processed_cols_ref = {'cols': []}

    def _process_chunk(chunk):
        # 1. Remap
        mapped, cols = remap_columns(chunk, src_type)
        if not processed_cols_ref['cols']:
            processed_cols_ref['cols'] = cols
        
        # 2. Filter
        if flter_col:
            mapped = filter_dataframe_by_values(mapped, flter_col, flter_val)
            mapped = filter_dataframe_by_range(mapped, flter_col, flter_start, flter_end)
        
        return mapped

    try:
        df = _read_and_process_chunks(pd.read_csv, read_kwargs, _process_chunk)
    except Exception as e:
        # Fallback to python engine if C engine fails (e.g. regex sep)
        logging.warning(f"Fast CSV read failed ({e}), falling back to slower python engine...")
        read_kwargs['engine'] = 'python'
        # low_memory not supported in python engine
        if 'low_memory' in read_kwargs: del read_kwargs['low_memory']
        df = _read_and_process_chunks(pd.read_csv, read_kwargs, _process_chunk)

    col_lst = processed_cols_ref['cols'] if processed_cols_ref['cols'] else list(df.columns)

    # If this file is part of a list, return df for subsequent processing in list handler
    if member_of_list:
        return df
    
    # Find pol_col
    pol_col = _check_column_in_df(df, POL_COLS)
    emis_col = _check_column_in_df(df, EMIS_COLS)

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
                    _emit_user_message(notify, 'INFO', f'Finished reading: {fp}')
                except Exception as exc:
                    _emit_user_message(notify, 'ERROR', f'Error reading {fp}: {exc}')
                    raise exc
    
    # Combine all parts into one DataFrame
    if not df_list:
        raise ValueError("No valid data found in list file.")
    
    # Stratagy to avoid FutureWarning while preserving schemas:
    # 1. Collect ALL columns from all dataframes (preserving order of appearance)
    all_columns = []
    seen_cols = set()
    for d in df_list:
        if d is not None and isinstance(d, pd.DataFrame):
            for c in d.columns:
                if c not in seen_cols:
                    seen_cols.add(c)
                    all_columns.append(c)

    # 2. Filter out empty/NA DataFrames to avoid FutureWarning
    valid_dfs = [d for d in df_list if d is not None and not d.empty and not d.isna().all().all()]
    
    # 3. Concatenate
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*concatenation.*")
        if not valid_dfs:
            # If all were empty, concat the original list to preserve structure
            df = pd.concat(df_list, ignore_index=True)
        else:
            # Concat only valid data
            df = pd.concat(valid_dfs, ignore_index=True)
        
    # Propagate is_preprocessed if ANY input was preprocessed.
    if any(getattr(d, 'attrs', {}).get('is_preprocessed') for d in df_list):
        df.attrs['is_preprocessed'] = True
        
    # 4. Ensure all discovered columns exist in the result
    # Only reindex if we actually missed something (optimization)
    if len(df.columns) < len(all_columns):
        df = df.reindex(columns=all_columns)
        
    df, col_lst = remap_columns(df, src_type)

    df = filter_dataframe_by_values(df, flter_col, flter_val)
    df = filter_dataframe_by_range(df, flter_col, flter_start, flter_end)
    
    # Find pol_col
    # Find pol_col
    pol_col = _check_column_in_df(df, POL_COLS)
    emis_col = _check_column_in_df(df, EMIS_COLS)

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
    flter_col: Optional[str] = None,
    flter_start: Optional[str] = None,
    flter_end: Optional[str] = None,
    flter_val: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Parse an emissions report using a '#Label' or '# County' header line."""
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
            # Higher penalty for space as it often appears in description text
            penalty = 0.4 if c == ' ' else 1.0
            scores[c] = coverage * stability * mean_cnt * penalty
        if scores:
            best = max(scores.items(), key=lambda kv: kv[1])
            if best[1] > 0:
                sep = best[0]

    # Parse header columns
    splitter = sep if sep is not None else (';' if ';' in header_line else ',')
    col_names = [t.strip() for t in header_line.split(splitter)]

    # Improve read performance by maximizing C engine usage
    read_kwargs = {
        'filepath_or_buffer': fpath,
        'skipinitialspace': True,
        'skiprows': header_idx + 1,
        'names': col_names,
        'comment': comment_marker,
        'encoding': encoding,
    }

    if sep == ' ':
        # Use delim_whitespace=True to allow C engine for space separated files
        read_kwargs['delim_whitespace'] = True
        read_kwargs['engine'] = 'c'
        read_kwargs['low_memory'] = False
    elif sep:
        read_kwargs['sep'] = sep
        read_kwargs['engine'] = 'c'
        read_kwargs['low_memory'] = False
    else:
        read_kwargs['sep'] = None
        read_kwargs['engine'] = 'python'
    
    def _process_rpt_chunk(chunk):
        if flter_col:
            chunk = filter_dataframe_by_values(chunk, flter_col, flter_val)
            chunk = filter_dataframe_by_range(chunk, flter_col, flter_start, flter_end)
        return chunk

    try:
        logging.info("Starting CSV read via Pandas...")
        df = _read_and_process_chunks(pd.read_csv, read_kwargs, _process_rpt_chunk)
        logging.info("Finished CSV read.")
    except Exception as e:
        logging.info(f"C-engine read failed ({e}), attempting Python engine fallback...")
        # Fallback: force python engine
        # Do not pass low_memory here as it is not supported by python engine
        fallback_kwargs = {
            'filepath_or_buffer': fpath,
            'sep': sep,
            'engine': 'python',
            'skipinitialspace': True,
            'skiprows': header_idx + 1,
            'names': col_names,
            'comment': comment_marker,
            'encoding': encoding
        }
        df = _read_and_process_chunks(pd.read_csv, fallback_kwargs, _process_rpt_chunk)

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
            # Check if it looks like a composite column (e.g. "2022001 01001 Alabama Autauga Co")
            # Heuristic: must have at least 4 tokens and one should look like a 5-7 digit FIPS
            sample = df[county_col].astype(str).iloc[0] if not df.empty else ''
            tokens = sample.split()
            if len(tokens) >= 4:
                try:
                    # Look for 5-7 digit numeric token
                    f_token = None
                    for t in tokens:
                        t_clean = re.sub(r'\D', '', t)
                        if 5 <= len(t_clean) <= 7:
                            f_token = t_clean
                            break
                    
                    if f_token:
                        df['FIPS'] = f_token.zfill(6)
                    else:
                        logging.debug(f"Skipping composite County fix: no FIPS-like token found in '{sample}'")
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

    wide = df[[group_key] + pollutant_cols].groupby(group_key, as_index=False, sort=False, observed=False).sum()
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
    if path is None:
        return gpd.GeoDataFrame()
        
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
        # Use a conservative tolerance (e.g. 0.0001 degrees ~ 10m)
        if not gdf.empty:
            try:
                gdf['geometry'] = gdf.simplify(0.0001, preserve_topology=True)
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

def generate_grid_polygons_vectorized(xorig, yorig, xcell, ycell, ncols, nrows, transformer):
    """
    Vectorized grid generation.
    Returns: (features, rows_attr, cols_attr)
    """
    nrows_int = int(nrows)
    ncols_int = int(ncols)
    
    # Generate all edge coordinates (N+1, M+1)
    x_edges = xorig + np.arange(ncols_int + 1) * xcell
    y_edges = yorig + np.arange(nrows_int + 1) * ycell
    xx, yy = np.meshgrid(x_edges, y_edges) # Shape (nrows+1, ncols+1)
    
    # Flatten and transform all coordinates to Lat/Lon at once
    # This avoids calling transform() inside a Python loop 140,000 times
    try:
        lon_flat, lat_flat = transformer.transform(xx.flatten(), yy.flatten())
    except Exception:
        # Fallback for older pyproj versions
        lon_flat, lat_flat = transformer.transform(xx.flatten(), yy.flatten())
        
    lons = lon_flat.reshape(nrows_int + 1, ncols_int + 1)
    lats = lat_flat.reshape(nrows_int + 1, ncols_int + 1)

    # Use shapely.polygons (vectorized) if available (Shapely 2.0+), else fast loop
    # Current environment checks
    try:
        from shapely import polygons
        # Construct coordinate array for all cells: (nrows, ncols, 5, 2)
        # Slicing: coords[row, col] -> (lons[row, col], lats[row, col])
        
        # BL: (r, c)     -> lons[:-1, :-1]
        # BR: (r, c+1)   -> lons[:-1, 1:]
        # TR: (r+1, c+1) -> lons[1:, 1:]
        # TL: (r+1, c)   -> lons[1:, :-1]
        
        bl_x = lons[:-1, :-1].flatten()
        bl_y = lats[:-1, :-1].flatten()
        
        br_x = lons[:-1, 1:].flatten()
        br_y = lats[:-1, 1:].flatten()
        
        tr_x = lons[1:, 1:].flatten()
        tr_y = lats[1:, 1:].flatten()
        
        tl_x = lons[1:, :-1].flatten()
        tl_y = lats[1:, :-1].flatten()
        
        n_cells = len(bl_x)
        coords = np.empty((n_cells, 5, 2), dtype=np.float64)
        
        coords[:, 0, 0] = bl_x; coords[:, 0, 1] = bl_y
        coords[:, 1, 0] = br_x; coords[:, 1, 1] = br_y
        coords[:, 2, 0] = tr_x; coords[:, 2, 1] = tr_y
        coords[:, 3, 0] = tl_x; coords[:, 3, 1] = tl_y
        coords[:, 4, 0] = bl_x; coords[:, 4, 1] = bl_y
        
        features = polygons(coords)
        
        # Generate attributes
        # meshgrid for indices
        r_idx, c_idx = np.indices((nrows_int, ncols_int))
        rows_attr = (r_idx + 1).flatten()
        cols_attr = (c_idx + 1).flatten()
        
        return features, rows_attr, cols_attr
        
    except (ImportError, AttributeError):
        # Fallback if Shapely < 2.0
        features, rows_attr, cols_attr = [], [], []
        # Fast manual loop using pre-computed lons/lats
        for r in range(nrows_int):
            row_num = r + 1
            y_base_idx = r
            for c in range(ncols_int):
                col_num = c + 1
                x_base_idx = c
                
                # Retrieve pre-transformed coordinates
                # BL
                x0, y0 = lons[y_base_idx, x_base_idx], lats[y_base_idx, x_base_idx]
                # BR
                x1, y1 = lons[y_base_idx, x_base_idx+1], lats[y_base_idx, x_base_idx+1]
                # TR
                x2, y2 = lons[y_base_idx+1, x_base_idx+1], lats[y_base_idx+1, x_base_idx+1]
                # TL
                x3, y3 = lons[y_base_idx+1, x_base_idx], lats[y_base_idx+1, x_base_idx]
                
                features.append(Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x0, y0)]))
                rows_attr.append(row_num)
                cols_attr.append(col_num)
                
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
        return gpd.GeoDataFrame({'name': [grid_name]}, geometry=[polygon_geom], crs='EPSG:4326')

    # Vectorized grid generation (Consistent with ncf_processing)
    features, rows_attr, cols_attr = generate_grid_polygons_vectorized(
        xorig, yorig, xcell, ycell, ncols, nrows, transformer
    )

    gdf = gpd.GeoDataFrame({'ROW': rows_attr, 'COL': cols_attr}, geometry=features, crs='EPSG:4326')
    try:
        # Fast str generation
        gdf['GRID_RC'] = gdf['ROW'].astype(str) + '_' + gdf['COL'].astype(str)
    except Exception:
        gdf['GRID_RC'] = [f"{r}_{c}" for r, c in zip(rows_attr, cols_attr)]
    
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

def map_latlon2grd(emis_df: pd.DataFrame, base_geom: gpd.GeoDataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Assign grid cells (GRID_RC) to FF10 point rows using optimized spatial join.
    Removes multiprocessing overhead for spatial joins.
    """
    if 'GRID_RC' in emis_df.columns:
        if verbose:
            print("GRID_RC column exists. Skip mapping lat/lon value to grid column and row")
        return emis_df
    
    if not isinstance(emis_df, pd.DataFrame):
        raise TypeError('FF10 emissions must be provided as a DataFrame.')
    if base_geom is None or not isinstance(base_geom, gpd.GeoDataFrame):
        raise TypeError('Grid geometry must be provided as a GeoDataFrame.')
    if 'GRID_RC' not in base_geom.columns:
        raise ValueError("Grid geometry is missing required 'GRID_RC' column.")

    # Identify Coords
    lon_candidates = ['longitude', 'lon', 'lon_dd', 'x']
    lat_candidates = ['latitude', 'lat', 'lat_dd', 'y']
    
    # Case insensitive check
    cols_map = {c.lower(): c for c in emis_df.columns}
    lon_col = next((cols_map[c] for c in lon_candidates if c in cols_map), None)
    lat_col = next((cols_map[c] for c in lat_candidates if c in cols_map), None)
    
    if not lon_col or not lat_col:
        raise ValueError('FF10 point data do not contain recognizable longitude/latitude columns.')

    # Unique locations
    coords = emis_df[[lon_col, lat_col]].dropna().drop_duplicates()
    if coords.empty:
        raise ValueError('No valid longitude/latitude pairs found to map onto the grid.')

    # Prepare Grid
    grid_subset = base_geom[['GRID_RC', 'geometry']]
    if grid_subset.crs is None:
        grid_subset.set_crs('EPSG:4326', inplace=True)
         
    # Prepare Points
    pts = gpd.GeoDataFrame(
        coords,
        geometry=gpd.points_from_xy(coords[lon_col], coords[lat_col]),
        crs='EPSG:4326'
    )
    
    if pts.crs != grid_subset.crs:
        pts = pts.to_crs(grid_subset.crs)

    if verbose:
        logging.info(f"Mapping {len(coords)} locations to grid (single-process)...")

    # Perform Spatial Join
    # sjoin with predicate='within' is efficient with rtree
    try:
        joined = gpd.sjoin(pts, grid_subset, how='left', predicate='within')
    except TypeError:
        joined = gpd.sjoin(pts, grid_subset, how='left', op='within')
        
    res = joined[[lon_col, lat_col, 'GRID_RC']].dropna().drop_duplicates()
    
    # Merge back
    mapped = emis_df.drop(columns=['GRID_RC'], errors='ignore').merge(
        res,
        on=[lon_col, lat_col],
        how='left'
    )
    
    # Copy attrs
    try:
        mapped.attrs = dict(getattr(emis_df, 'attrs', {}))
    except Exception:
        pass
        
    return mapped


def detect_pollutants(df: pd.DataFrame) -> List[str]:
    try:
        # Priority 1: Specifically cached list (internal)
        cached = df.attrs.get('_detected_pollutants')
        if isinstance(cached, (list, tuple)):
            return list(cached)
        
        # Priority 2: Lazy loader pre-identified list
        lazy_pols = df.attrs.get('available_pollutants')
        if isinstance(lazy_pols, (list, tuple)) and lazy_pols:
            return list(lazy_pols)
    except Exception:
        pass
    # Exclude grid identifiers as well
    id_like = {
        'fips', 'region', 'state', 'county', 'x', 'y', 'x cell', 'y cell', 'row', 'col', 'grid_rc',
        '#label', 'label', 'src', 'source', 'sourcename', 'source_name', 'scc', 'scc description',
        'region_cd', 'state_cd', 'county_cd',
        'facility id', 'fac name', 'country_cd', 'tribal_code', 'emis_type',
        'facility_id', 'unit_id', 'rel_point_id','latitude','longitude','lat_dd','lon_dd','lat','lon',
        'stkhgt', 'stkdiam', 'stktemp', 'stkflow', 'stkvel', 'erptype', 'naics', 'census_tract',
        'design_capacity', 'design_capacity_units', 'reg_codes', 'fac_source_type', 'unit_type_code',
        'control_ids', 'control_measures', 'current_cost', 'cumulative_cost', 'projection_factor',
        'submitter_id', 'calc_method', 'data_set_id', 'facil_category_code', 'oris_facility_code',
        'oris_boiler_id', 'ipm_yn', 'calc_year', 'date_updated', 'fug_height', 'fug_width_xdim',
        'fug_length_ydim', 'fug_angle', 'zipcode', 'annual_avg_hours_per_year',
        'jan_value', 'feb_value', 'mar_value', 'apr_value', 'may_value', 'jun_value',
        'jul_value', 'aug_value', 'sep_value', 'oct_value', 'nov_value', 'dec_value',
        'ann_pct_red', 'jan_pctred', 'feb_pctred', 'mar_pctred', 'apr_pctred',
        'may_pctred', 'jun_pctred', 'jul_pctred', 'aug_pctred', 'sep_pctred', 
        'oct_pctred', 'nov_pctred', 'dec_pctred', 'process_id', 'agy_facility_id',
        'agy_unit_id', 'agy_rel_point_id', 'agy_process_id', 'll_datum', 'horiz_coll_mthd'
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

def remap_tribal_fips(df: pd.DataFrame, fips_col: str = 'FIPS') -> pd.DataFrame:
    """Map tribal area codes back to standard US county FIPS using config mapping."""
    if df is None or not TRIBAL_TO_COUNTY_FIPS:
        return df
    
    # Identify relevant columns
    cols_to_check = [fips_col] if fips_col in df.columns else []
    tribal_col = _check_column_in_df(df, TRIBAL_COLS, warn=False)
    if tribal_col and tribal_col not in cols_to_check:
        cols_to_check.append(tribal_col)
        
    if not cols_to_check:
        return df

    def _get_mapped_val(val):
        if pd.isna(val) or val == '': return None
        s_val = str(val).strip()
        # Direct match (e.g. '88750' or 'Navajo')
        if s_val in TRIBAL_TO_COUNTY_FIPS:
            return TRIBAL_TO_COUNTY_FIPS[s_val]
        # Padded matches
        z_val = s_val.zfill(6)
        if z_val in TRIBAL_TO_COUNTY_FIPS:
            return TRIBAL_TO_COUNTY_FIPS[z_val]
        if z_val.startswith('0') and z_val[1:] in TRIBAL_TO_COUNTY_FIPS:
            return TRIBAL_TO_COUNTY_FIPS[z_val[1:]]
        return None

    # We want to fill fips_col. If fips_col doesn't exist yet, we'll create it later in get_emis_fips.
    # But for now, if it exists, try to fix it.
    has_fips = fips_col in df.columns
    
    # Try mapping from each candidate column
    remapping_found = False
    for context_col in cols_to_check:
        mapped = df[context_col].apply(_get_mapped_val)
        mask = mapped.notna()
        if mask.any():
            if not remapping_found:
                df = df.copy()
                remapping_found = True
            
            target_col = fips_col if has_fips else context_col
            # Update values and ensure consistent string dtypes to avoid mixed-type inference
            # that triggers FutureWarning: Index.insert with object-dtype.
            df.loc[mask, target_col] = mapped[mask].astype(str)
            logging.info(f"Remapped {mask.sum()} tribal records from '{context_col}' to county FIPS in '{target_col}'")

    return df

def get_emis_fips(df: pd.DataFrame, verbose: bool = True):
    # Step 1: Remap tribal codes if they exist in FIPS or tribal_code columns
    df = remap_tribal_fips(df, 'FIPS')
    
    # If FIPS column exists, we might still want to normalize it or ensure it's 6 digits.
    # However, if it's already there and seems to be the right length/format, we can skip heavy rebuilds.
    # For now, let's remove the aggressive early return to ensure normalization happens if needed.
    # We will only skip if FIPS is already 6-digit strings on average.
    pass

    # Find region_cd column if df
    region_col = _check_column_in_df(df, REGION_COLS, warn=False)

    # Check if region_col less than 6 characters long, affix with country code
    if region_col:
        lengths = df[region_col].astype('string').str.strip().str.len()
        region_stats = lengths.describe()
        max_len = int(region_stats['max'])

        if max_len < 6:
            # Locate country_cd column if exists
            country_col = _check_column_in_df(df, COUNTRY_COLS, warn=False)
            if country_col:
                if verbose:
                    logging.info(f"Building FIPS by affixing country code from column {country_col} to region_cd column {region_col}")
                ## Map country code value from COUNTRY_CODE_MAPPINGS dict to new column df[country_id]
                country_codes = (
                    df[country_col]
                    .astype('string')
                    .str.strip()
                    .str.upper()
                    .map(COUNTRY_CODE_MAPPINGS)
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
            # Consolidate via copy to avoid Fragmentation PerformanceWarning, 
            # then assign directly to avoids FutureWarning: Index.insert.
            df = df.copy()
            df['FIPS'] = country_codes + fips_suffix
        else:
            # Ensure robust string conversion
            s_region = df[region_col].astype(str)
            if pd.api.types.is_float_dtype(df[region_col]):
                try:
                    s_region = df[region_col].fillna(-1).astype(int).astype(str).replace('-1', '')
                except Exception:
                    pass
            
            # Consolidate via copy to avoid Fragmentation PerformanceWarning, 
            # then assign directly to avoids FutureWarning: Index.insert.
            df = df.copy()
            df['FIPS'] = s_region.str.zfill(6)
        
        return df
    else:
        if verbose:
            print("region_cd column not found, checking for statefp and countyfp columns to build FIPS")
        # Find if there is statefp and countyfp columns to build fips
        statefp_col = _check_column_in_df(df, ['statefp','state_cd','state_code'], warn=False)
        countyfp_col = _check_column_in_df(df, ['countyfp','county_cd','county_code'], warn=False)
        if statefp_col and countyfp_col:
            original_attrs = df.attrs.copy()
            new_fips = '0' + df[statefp_col].astype(str).str.zfill(2) + df[countyfp_col].astype(str).str.zfill(3)
            # Consolidate via copy to avoid Fragmentation PerformanceWarning, 
            # then assign directly to avoids FutureWarning: Index.insert.
            df = df.copy()
            df['FIPS'] = new_fips

    # Final pass: Remap tribal codes in the newly built or existing FIPS column
    if 'FIPS' in df.columns:
        df = remap_tribal_fips(df, 'FIPS')

    return df
    
    raise ValueError("No valid FIPS code columns found")

