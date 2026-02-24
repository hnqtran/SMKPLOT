#!/proj/ie/proj/SMOKE/htran/Emission_Modeling_Platform/utils/smkplot/.venv/bin/python
"""NetCDF and Inline processing for SMKPLOT.

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
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import netCDF4
import xarray as xr
import pyproj
import tempfile
import atexit
import concurrent.futures

from config import USE_SPHERICAL_EARTH
from data_processing import _file_signature, extract_grid

# Global list to track temporary files for cleanup
_TEMP_FILES = []
# Global cache for Inline Mapping: (path_hash, stack_hash, grid_name) -> (valid_indices, v_rows, v_cols, gi)
_INLINE_MAPPING_CACHE = {}

def _cleanup_temp_files():
    for f in _TEMP_FILES:
        try:
            if os.path.exists(f):
                os.unlink(f)
        except Exception:
            pass

atexit.register(_cleanup_temp_files)

def _process_inline_chunk(inln_path, vars_to_process, t_start, t_end, valid_indices, v_rows, v_cols, nrows, ncols, n_lay, src_count):
    """
    Worker function to process a chunk of time steps for inline processing.
    """
    import numpy as np
    import netCDF4
    
    t_indices = range(t_start, t_end)
    chunk_len = len(t_indices)
    res_data = {} # {vname: np.array(chunk_len, LAY, Rows, Cols)}
    
    # Pre-allocate
    for vname in vars_to_process:
        res_data[vname] = np.zeros((chunk_len, n_lay, nrows, ncols), dtype=np.float32)
        
    try:
        with netCDF4.Dataset(inln_path, 'r') as nc_in:
            for i, t in enumerate(t_indices):
                for vname in vars_to_process:
                    data_in = nc_in.variables[vname][t]
                     
                    # Reshape logic
                    if data_in.ndim > 2:
                        data_in = data_in.reshape(data_in.shape[0], -1)
                    elif data_in.ndim == 1:
                        data_in = data_in.reshape(1, -1)
                     
                    if data_in.shape[1] != src_count:
                        if data_in.shape[0] == src_count:
                                data_in = data_in.T
                               
                    # Subset Valid
                    if valid_indices.size > 0:
                        data_valid = data_in[:, valid_indices]
                          
                        # Accumulate
                        for L in range(n_lay):
                                np.add.at(res_data[vname][i, L, :, :], (v_rows, v_cols), data_valid[L, :])
    except Exception as e:
        # Returning Exception for main process to handle
        return e

    return (t_start, t_end, res_data)

def get_proj_object_from_info(gi):
    """Returns pyproj.CRS object based on grid info dictionary."""
    if gi['proj_type'] == 2: # LCC
        proj_str = (f"+proj=lcc +lat_1={gi['p_alp']} +lat_2={gi['p_bet']} "
                    f"+lat_0={gi['ycent']} +lon_0={gi['xcent']} "
                    f"+x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs")
        return pyproj.CRS.from_user_input(proj_str)
    elif gi['proj_type'] == 6: # Polar Stereographic
        proj_str = (f"+proj=stere +lat_ts={gi['p_alp']} +lat_0={gi['ycent']} +lon_0={gi['xcent']} "
                    f"+x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs")
        return pyproj.CRS.from_user_input(proj_str)
    elif gi['proj_type'] == 1: # Lat/Lon
        return pyproj.CRS.from_epsg(4326)
    else:
        raise NotImplementedError(f"Projection type {gi['proj_type']} not yet implemented.")

def _get_inline_mapping_key(inln_path, stack_groups_path, grid_name):
    return (inln_path, stack_groups_path, grid_name)

def _get_inline_mapping(inln_path, stack_groups_path, griddesc_path, grid_name):
    """
    Compute or retrieve valid source mapping for Inline processing.
    Returns: (valid_indices, v_rows, v_cols, gi)
    """
    key = _get_inline_mapping_key(inln_path, stack_groups_path, grid_name)
    if key in _INLINE_MAPPING_CACHE:
        return _INLINE_MAPPING_CACHE[key]

    with netCDF4.Dataset(inln_path) as nc_in:
        # Determine Grid Name logic (duplicated from process_inline_emissions, centralized here soon)
        if not grid_name:
            if hasattr(nc_in, 'GDNAM'):
                grid_name = str(nc_in.GDNAM).strip()
            else:
                grid_name = 'UNKNOWN'
        
        # Grid Definition
        gi = None
        if griddesc_path and os.path.exists(griddesc_path):
            try:
                coord_params, grid_params = extract_grid(griddesc_path, grid_name)
                gi = {
                    'proj_type': int(coord_params[0]),
                    'p_alp': float(coord_params[1]),
                    'p_bet': float(coord_params[2]),
                    'p_gam': float(coord_params[3]),
                    'xcent': float(coord_params[4]),
                    'ycent': float(coord_params[5]),
                    'xorig': float(grid_params[1]),
                    'yorig': float(grid_params[2]),
                    'xcell': float(grid_params[3]),
                    'ycell': float(grid_params[4]),
                    'ncols': int(grid_params[5]),
                    'nrows': int(grid_params[6])
                }
            except Exception:
                pass
        
        if gi is None:
            try:
                gi = {}
                gi['xorig'] = float(getattr(nc_in, 'XORIG') or 0)
                gi['yorig'] = float(getattr(nc_in, 'YORIG') or 0)
                gi['xcell'] = float(getattr(nc_in, 'XCELL') or 0)
                gi['ycell'] = float(getattr(nc_in, 'YCELL') or 0)
                gi['ncols'] = int(getattr(nc_in, 'NCOLS') or 1)
                gi['nrows'] = int(getattr(nc_in, 'NROWS') or 1)
                gi['proj_type'] = int(getattr(nc_in, 'GDTYP') or 2)
                gi['p_alp'] = float(getattr(nc_in, 'P_ALP') or 0)
                gi['p_bet'] = float(getattr(nc_in, 'P_BET') or 0)
                gi['p_gam'] = float(getattr(nc_in, 'P_GAM') or 0)
                gi['xcent'] = float(getattr(nc_in, 'XCENT') or 0)
                gi['ycent'] = float(getattr(nc_in, 'YCENT') or 0)
            except AttributeError:
                raise ValueError("Could not determine grid parameters.")

        # Read STACK_GROUPS
        logging.info(f"DEBUG: Reading STACK_GROUPS: {stack_groups_path}")
        with netCDF4.Dataset(stack_groups_path) as nc_stack:
            lats = np.array(nc_stack.variables['LATITUDE'][:]).flatten()
            lons = np.array(nc_stack.variables['LONGITUDE'][:]).flatten()
            
            # Check for spatial dimensions in STACK_GROUPS FILEDESC attribute
            if gi is not None and (gi['ncols'] == 1 or gi['nrows'] == 1):
                filedesc = str(getattr(nc_stack, 'FILEDESC', ''))
                import re
                c_match = re.search(r'/NCOLS3D/\s*(\d+)', filedesc)
                r_match = re.search(r'/NROWS3D/\s*(\d+)', filedesc)
                if c_match and r_match:
                    gi['ncols'] = int(c_match.group(1))
                    gi['nrows'] = int(r_match.group(1))
                    logging.info(f"Inferred spatial grid dimensions from STACK_GROUPS FILEDESC: {gi['ncols']}x{gi['nrows']}")
                else:
                    logging.warning(f"Inline file has 1D dimensions and STACK_GROUPS FILEDESC does not provide NCOLS3D/NROWS3D. Plotting may be incorrect.")

        logging.info(f"DEBUG: Projecting {len(lats)} sources...")

    # Project
    proj_crs = get_proj_object_from_info(gi)
    proj = pyproj.Proj(proj_crs)
    xx, yy = proj(lons, lats)
    
    if gi['xcell'] == 0 or gi['ycell'] == 0:
        raise ValueError("Grid cell size is zero.")
         
    cols = np.floor((xx - gi['xorig']) / gi['xcell']).astype(int)
    rows = np.floor((yy - gi['yorig']) / gi['ycell']).astype(int)
    
    valid_mask = (cols >= 0) & (cols < gi['ncols']) & (rows >= 0) & (rows < gi['nrows'])
    
    v_cols = cols[valid_mask]
    v_rows = rows[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    res = (valid_indices, v_rows, v_cols, gi)
    _INLINE_MAPPING_CACHE[key] = res
    return res

def process_inline_emissions(inln_path: str, stack_groups_path: str, griddesc_path: str = None, grid_name: str = None) -> str:
    """
    Process INLN file using STACK_GROUPS file mapping and GridDef.
    Converts the 1D inline emissions into a 2D gridded NetCDF file (temporary).
    Returns the path to the temporary NetCDF file.
    """
    if not stack_groups_path or not os.path.exists(stack_groups_path):
        raise ValueError("STACK_GROUPS file is required for processing inline emissions but was not provided or found.")
        
    with netCDF4.Dataset(inln_path) as nc_in:
        # Determine Grid Name
        if not grid_name:
            if hasattr(nc_in, 'GDNAM'):
                grid_name = str(nc_in.GDNAM).strip()
            else:
                # If no GDNAM, we might proceed if we don't strictly need it (e.g. if using heuristics), but usually IOAPI has it.
                # Warn and use placeholder if missing?
                logging.warning("GRID_NAME not set and GDNAM attribute missing in INLN file. Using 'UNKNOWN'.")
                grid_name = 'UNKNOWN'
        
        # Grid Definition
        gi = None
        if griddesc_path and os.path.exists(griddesc_path):
            try:
                # Reuse existing centralized logic
                coord_params, grid_params = extract_grid(griddesc_path, grid_name)
                # Map to dictionary structure used below
                gi = {
                    'proj_type': int(coord_params[0]),
                    'p_alp': float(coord_params[1]),
                    'p_bet': float(coord_params[2]),
                    'p_gam': float(coord_params[3]),
                    'xcent': float(coord_params[4]),
                    'ycent': float(coord_params[5]),
                    'xorig': float(grid_params[1]),
                    'yorig': float(grid_params[2]),
                    'xcell': float(grid_params[3]),
                    'ycell': float(grid_params[4]),
                    'ncols': int(grid_params[5]),
                    'nrows': int(grid_params[6])
                }
            except Exception:
                pass
        
        # If GRIDDESC failed or not provided, try to extract from NC header
        if gi is None:
            # Try to construct 'gi' from NC global attributes if they exist
            try:
                gi = {}
                gi['xorig'] = float(getattr(nc_in, 'XORIG') or 0)
                gi['yorig'] = float(getattr(nc_in, 'YORIG') or 0)
                gi['xcell'] = float(getattr(nc_in, 'XCELL') or 0)
                gi['ycell'] = float(getattr(nc_in, 'YCELL') or 0)
                gi['ncols'] = int(getattr(nc_in, 'NCOLS') or 1)
                gi['nrows'] = int(getattr(nc_in, 'NROWS') or 1)
                 
                gi['proj_type'] = int(getattr(nc_in, 'GDTYP') or 2)
                gi['p_alp'] = float(getattr(nc_in, 'P_ALP') or 0)
                gi['p_bet'] = float(getattr(nc_in, 'P_BET') or 0)
                gi['p_gam'] = float(getattr(nc_in, 'P_GAM') or 0)
                gi['xcent'] = float(getattr(nc_in, 'XCENT') or 0)
                gi['ycent'] = float(getattr(nc_in, 'YCENT') or 0)
                 
                # Fix for Inline files where NCOLS/NROWS describe list size (e.g. 1 x NSRC)
                # Spatial dimensions will be inferred from STACK_GROUPS or GDNAM below.
                pass

            except AttributeError:
                raise ValueError(f"Could not determine grid parameters from INLN header or GRIDDESC for '{grid_name}'.")

        # Read STACK_GROUPS
        with netCDF4.Dataset(stack_groups_path) as nc_stack:
            if 'LATITUDE' not in nc_stack.variables or 'LONGITUDE' not in nc_stack.variables:
                raise ValueError("STACK_GROUPS file must contain LATITUDE and LONGITUDE variables.")
            
            lats = np.array(nc_stack.variables['LATITUDE'][:]).flatten()
            lons = np.array(nc_stack.variables['LONGITUDE'][:]).flatten()
            
            # Check for spatial dimensions in STACK_GROUPS FILEDESC attribute
            if gi is not None and (gi['ncols'] == 1 or gi['nrows'] == 1):
                filedesc = str(getattr(nc_stack, 'FILEDESC', ''))
                import re
                c_match = re.search(r'/NCOLS3D/\s*(\d+)', filedesc)
                r_match = re.search(r'/NROWS3D/\s*(\d+)', filedesc)
                if c_match and r_match:
                    gi['ncols'] = int(c_match.group(1))
                    gi['nrows'] = int(r_match.group(1))
                    logging.info(f"Inferred spatial grid dimensions from STACK_GROUPS FILEDESC: {gi['ncols']}x{gi['nrows']}")
                else:
                    logging.warning(f"Inline file has 1D dimensions ({gi['ncols']}x{gi['nrows']}) and STACK_GROUPS FILEDESC does not provide NCOLS3D/NROWS3D.")

        # Project
        proj_crs = get_proj_object_from_info(gi)
        proj = pyproj.Proj(proj_crs)
        xx, yy = proj(lons, lats)
        
        # Calculate Grid Indices (0-based)
        # Avoid division by zero if xcell/ycell are 0 (bad header)
        if gi['xcell'] == 0 or gi['ycell'] == 0:
            raise ValueError("Grid cell size (XCELL/YCELL) is zero. Invalid grid definition.")
             
        cols = np.floor((xx - gi['xorig']) / gi['xcell']).astype(int)
        rows = np.floor((yy - gi['yorig']) / gi['ycell']).astype(int)
        
        # Determine valid sources
        valid_mask = (cols >= 0) & (cols < gi['ncols']) & (rows >= 0) & (rows < gi['nrows'])
        
        v_cols = cols[valid_mask]
        v_rows = rows[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        logging.info(f"Processing Inline sources: {len(valid_indices)} of {len(lats)} sources are within grid {grid_name}.")
        
        # Create Loop for variables
        vars_to_process = []
        for vname in nc_in.variables:
            if vname not in nc_in.dimensions and vname != 'TFLAG':
                if np.issubdtype(nc_in.variables[vname].dtype, np.floating) or np.issubdtype(nc_in.variables[vname].dtype, np.integer):
                    vars_to_process.append(vname)
        
        # Create Temp File
        fd, temp_path = tempfile.mkstemp(suffix='_inline_2d.nc')
        os.close(fd) # we use netCDF4 to open it
        _TEMP_FILES.append(temp_path)
        
        try:
            with netCDF4.Dataset(temp_path, 'w', format='NETCDF3_CLASSIC') as nc_out:
                # Copy Global Attributes
                for attr in nc_in.ncattrs():
                    nc_out.setncattr(attr, nc_in.getncattr(attr))
                
                # Update Grid Attributes (Force correct 2D dimensions)
                nc_out.NCOLS = gi['ncols']
                nc_out.NROWS = gi['nrows']
                nc_out.XORIG = gi['xorig']
                nc_out.YORIG = gi['yorig']
                nc_out.XCELL = gi['xcell']
                nc_out.YCELL = gi['ycell']
                nc_out.GDNAM = grid_name
                nc_out.GDTYP = gi['proj_type']
                nc_out.P_ALP = gi['p_alp']
                nc_out.P_BET = gi['p_bet']
                nc_out.P_GAM = gi['p_gam']
                nc_out.XCENT = gi['xcent']
                nc_out.YCENT = gi['ycent']
                nc_out.history = (str(getattr(nc_in, 'history', '')) + " ; processed in-memory inline-to-2d").strip()

                # Dimensions
                tstep_dim = nc_in.dimensions.get('TSTEP')
                lay_dim_in = nc_in.dimensions.get('LAY')
                
                n_lay = len(lay_dim_in) if lay_dim_in else 1
                
                nc_out.createDimension('TSTEP', None) # Unlimited
                nc_out.createDimension('LAY', n_lay)
                nc_out.createDimension('ROW', gi['nrows'])
                nc_out.createDimension('COL', gi['ncols'])
                nc_out.createDimension('VAR', len(vars_to_process))
                nc_out.createDimension('DATE-TIME', 2)
                
                # Copy TFLAG
                if 'TFLAG' in nc_in.variables:
                    tf_in = nc_in.variables['TFLAG']
                    tf_out = nc_out.createVariable('TFLAG', 'i4', ('TSTEP', 'VAR', 'DATE-TIME'))
                    for attr in tf_in.ncattrs():
                        tf_out.setncattr(attr, tf_in.getncattr(attr))
                
                # Setup Variables
                out_vars = {}
                for vname in vars_to_process:
                    vin = nc_in.variables[vname]
                    # Use float32 for output to save space, unless input was integer-like? usually emissions are float.
                    vout = nc_out.createVariable(vname, 'f4', ('TSTEP', 'LAY', 'ROW', 'COL'))
                    
                    # Copy attributes
                    for attr in vin.ncattrs():
                        vout.setncattr(attr, vin.getncattr(attr))
                    out_vars[vname] = vout
                
                # Processing Loop (Time Steps)
                # TSTEP dimension size
                n_tsteps = len(tstep_dim) if tstep_dim and not tstep_dim.isunlimited() else (
                    nc_in.variables[vars_to_process[0]].shape[0] if vars_to_process else 1
                )
                if tstep_dim and tstep_dim.isunlimited():
                    # Check actual size from a variable
                    if vars_to_process:
                        n_tsteps = nc_in.variables[vars_to_process[0]].shape[0]

                # Process
                cpu_count = min(8, os.cpu_count() or 1)
                should_parallel = (n_tsteps >= 24) and (cpu_count > 1) 
                
                if should_parallel:
                    logging.info(f"Processing {n_tsteps} time steps in parallel using {cpu_count} workers.")
                    chunk_size = max(12, n_tsteps // (cpu_count * 2))
                     
                    # Create batches of (start, end)
                    batches = []
                    for i in range(0, n_tsteps, chunk_size):
                        batches.append((i, min(i + chunk_size, n_tsteps)))
                     
                    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                        futures = {
                            executor.submit(
                                _process_inline_chunk,
                                inln_path, 
                                vars_to_process, 
                                b[0], b[1], # start, end
                                valid_indices, 
                                v_rows, 
                                v_cols, 
                                gi['nrows'], 
                                gi['ncols'], 
                                n_lay, 
                                len(lats)
                            ): b for b in batches
                        }
                         
                        for future in concurrent.futures.as_completed(futures):
                            res = future.result()
                            if isinstance(res, Exception):
                                raise res
                            if res is None:
                                raise RuntimeError("Parallel worker failed silently.")
                                  
                            start_t, end_t, chunk_res = res
                             
                            # 1. Copy TFLAG for this chunk
                            if 'TFLAG' in nc_in.variables:
                                tf_slice = nc_in.variables['TFLAG'][start_t:end_t]
                                if tf_out.shape[1] == tf_slice.shape[1]:
                                    tf_out[start_t:end_t, :, :] = tf_slice
                                else:
                                    m = min(tf_out.shape[1], tf_slice.shape[1])
                                    tf_out[start_t:end_t, :m, :] = tf_slice[:, :m, :]
                                     
                            # 2. Write Data
                            for vname, data in chunk_res.items():
                                out_vars[vname][start_t:end_t, ...] = data
                             
                            logging.info(f"Processed batch steps {start_t}-{end_t-1}")

                else:
                    # Sequential Fallback
                    for t in range(n_tsteps):
                        # Copy TFLAG
                        if 'TFLAG' in nc_in.variables:
                            tf_in_data = nc_in.variables['TFLAG'][t, :, :]
                            if tf_out.shape[1] == tf_in_data.shape[0]:
                                tf_out[t, :, :] = tf_in_data
                            else:
                                min_var = min(tf_out.shape[1], tf_in_data.shape[0])
                                tf_out[t, :min_var, :] = tf_in_data[:min_var, :]

                        for vname in vars_to_process:
                            # shape: (T, LAY, NSRC) or (T, LAY, 1, NSRC) etc.
                            data_in = nc_in.variables[vname][t] # (LAY, ...)
                            
                            # Flatten to (LAY, NSRC)
                            if data_in.ndim > 2:
                                data_in = data_in.reshape(data_in.shape[0], -1)
                            elif data_in.ndim == 1:
                                data_in = data_in.reshape(1, -1)
                            
                            # Fix Case: (LAY, 1) if NSRC=1
                            if data_in.shape[1] != len(lats):
                                # Try transpose?
                                if data_in.shape[0] == len(lats):
                                        data_in = data_in.T
                            
                            # Prepare grid: (LAY, R, C)
                            grid_data = np.zeros((n_lay, gi['nrows'], gi['ncols']), dtype=np.float32)
                            
                            # Subset valid sources
                            # data_in: (LAY, ALL_SRC) -> valid -> (LAY, VALID_SRC)
                            if valid_indices.size > 0:
                                data_valid = data_in[:, valid_indices]
                                
                                # Vectorized accumulation
                                # add.at works on flat array or specific dims.
                                # grid_data[lay, r, c] += val
                                
                                for L in range(n_lay):
                                    np.add.at(grid_data[L, :, :], (v_rows, v_cols), data_valid[L, :])
                            
                            out_vars[vname][t, :, :, :] = grid_data

            return temp_path

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            # Remove from global list
            if temp_path in _TEMP_FILES:
                _TEMP_FILES.remove(temp_path)
            raise e
def read_ncf_grid_params(ncf_path: str):
    """
    Extract grid parameters from IOAPI NetCDF global attributes.
    Returns:
        coord_params: (GDTYP, P_ALP, P_BET, P_GAM, XCENT, YCENT)
        grid_params:  (GDNAM, XORIG, YORIG, XCELL, YCELL, NCOLS, NROWS, NTHIK)
    """
    with netCDF4.Dataset(ncf_path, 'r') as ds:
        # Helper to safely get attr (some might be missing or different types)
        def get_attr(name, default=None):
            if hasattr(ds, name):
                return getattr(ds, name)
            return default

        gdtyp = int(get_attr('GDTYP', 0))
        p_alp = float(get_attr('P_ALP', 0.0))
        p_bet = float(get_attr('P_BET', 0.0))
        p_gam = float(get_attr('P_GAM', 0.0))
        xcent = float(get_attr('XCENT', 0.0))
        ycent = float(get_attr('YCENT', 0.0))
        
        coord_params = (gdtyp, p_alp, p_bet, p_gam, xcent, ycent)
        
        gdnam = str(get_attr('GDNAM', 'UNKNOWN')).strip()
        xorig = float(get_attr('XORIG', 0.0))
        yorig = float(get_attr('YORIG', 0.0))
        xcell = float(get_attr('XCELL', 0.0))
        ycell = float(get_attr('YCELL', 0.0))
        ncols = int(get_attr('NCOLS', 0))
        nrows = int(get_attr('NROWS', 0))
        nthik = int(get_attr('NTHIK', 1))

        # Fix for Inline files where NCOLS/NROWS describe list size
        if ncols == 1 or nrows == 1:
            import re
            match = re.search(r'_(\d+)[Xx](\d+)\s*$', gdnam)
            if match:
                    ncols = int(match.group(1))
                    nrows = int(match.group(2)) 
        
        grid_params = (gdnam, xorig, yorig, xcell, ycell, ncols, nrows, nthik)
        
    return coord_params, grid_params

def get_ncf_dims(ncf_path: str):
    """
    Return dictionary with dimension details from the NetCDF.
    """
    info = {
        'n_tsteps': 0,
        'n_layers': 0,
        'tflag_values': []
    }
    with netCDF4.Dataset(ncf_path, 'r') as ds:
        dims = ds.dimensions
        if 'TSTEP' in dims:
            info['n_tsteps'] = dims['TSTEP'].size
        if 'LAY' in dims:
            info['n_layers'] = dims['LAY'].size
             
        # Try to read TFLAG key if available to get dates/times
        if 'TFLAG' in ds.variables:
            try:
                tf = ds.variables['TFLAG']
                # Shape usually (TSTEP, VAR, DATE-TIME). DATE-TIME is dim 2 (indices 0=date, 1=time)
                # We take all time steps, first variable (0), both date and time components
                if tf.ndim >= 2:
                    # Get array. If massive, this might be slow, but TSTEP usually < 8760
                    # Use numpy slicing. tf[:, 0, :] -> Shape (TSTEP, 2) if 3D
                    # If 2D (TSTEP, DATE-TIME), then tf[:]
                    if tf.ndim == 3:
                        tflags = tf[:, 0, :] 
                    else:
                        tflags = tf[:]
                     
                    for i in range(tflags.shape[0]):
                        # TFLAG is usually int32
                        yd = int(tflags[i, 0]) # YYYYDDD
                        hms = int(tflags[i, 1]) # HHMMSS
                        info['tflag_values'].append(f"{yd} {hms:06d}")
            except Exception:
                pass
                 
    return info


def create_ncf_domain_gdf(
    ncf_path: str,
    full_grid: bool = True
) -> gpd.GeoDataFrame:
    """
    Generate a GeoDataFrame of grid cells based on the NetCDF's global attributes.
    This avoids the need for an external GRIDDESC file.
    """
    coord_params, grid_params = read_ncf_grid_params(ncf_path)
    grid_name = grid_params[0]
    
    # Unpack for projection string construction
    gdtyp, p_alp, p_bet, p_gam, xcent, ycent = coord_params
    _, xorig, yorig, xcell, ycell, ncols, nrows, _ = grid_params
    
    if gdtyp != 2:
        # GDTYP=2 is LCC. For others (LatLon=1, Mercator=3, etc.), logic differs.
        # Assuming LCC for now based on typical CMAQ/SMOKE outputs.
        # TODO: Add support for other projections if needed.
        pass

    if USE_SPHERICAL_EARTH:
        proj_str = (
            f"+proj=lcc +lat_1={p_alp} +lat_2={p_bet} +lat_0={ycent} "
            f"+lon_0={xcent} +a=6370000.0 +b=6370000.0 +x_0=0 +y_0=0 +units=m +no_defs"
        )
    else:
        proj_str = (
            f"+proj=lcc +lat_1={p_alp} +lat_2={p_bet} +lat_0={ycent} "
            f"+lon_0={xcent} +ellps=WGS84 +datum=WGS84 +x_0=0 +y_0=0 +units=m +no_defs"
        )
        
    # Setup transformer for corners (if full_grid=False) or re-use chunk logic
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

    # Full grid generation
    # Optimized sequential generation using vectorized coordinates and Shapely array creation.
    # Parallel processing with fork() is avoided due to SegFault risks in GUI.

    # STRICT RULE 1: All grid generation MUST use the same identical function.
    try:
        from data_processing import generate_grid_polygons_vectorized
    except ImportError:
        # If running as script or different path structure
        import sys
        sys.path.append(os.path.dirname(__file__))
        from data_processing import generate_grid_polygons_vectorized

    features, rows_attr, cols_attr = generate_grid_polygons_vectorized(
        xorig, yorig, xcell, ycell, ncols, nrows, transformer
    )

    gdf = gpd.GeoDataFrame(
        {
            'ROW': rows_attr,
            'COL': cols_attr,
            # 'GRID_RC': [f"{r}_{c}" for r, c in zip(rows_attr, cols_attr)] # Slow
        },
        geometry=features,
        crs='EPSG:4326'
    )

    # Optimized GRID_RC creation
    try:
        gdf['GRID_RC'] = gdf['ROW'].astype(str) + '_' + gdf['COL'].astype(str)
    except:
        gdf['GRID_RC'] = [f"{r}_{c}" for r, c in zip(rows_attr, cols_attr)]

    # Attach grid info for optimized QuadMesh plotting
    try:
        gdf.attrs['_smk_grid_info'] = {
            'xorig': xorig,
            'yorig': yorig,
            'xcell': xcell,
            'ycell': ycell,
            'ncols': ncols,
            'nrows': nrows,
            'proj_str': proj_str
        }
    except Exception:
        pass

    return gdf


def read_ncf_emissions(
    ncf_path: str,
    pollutants: list = None,
    layer_idx: int = 0,
    tstep_idx: int = None,
    layer_op: str = 'select',
    tstep_op: str = 'mean',
    notify = None,
    xr_ds = None,
    lazy: bool = False
) -> pd.DataFrame:
    """
    Read emissions from NetCDF.
    layer_idx: 0-based index of layer to read. If None, uses layer_op.
    tstep_idx: 0-based index of time step to read. If None, uses tstep_op.
    layer_op: 'select' (default), 'sum', 'mean', 'max', or 'min'.
    tstep_op: 'select' (default if idx provided), 'sum', 'mean', 'max', or 'min' (default mean if idx None).
    notify: Optional callback function(level, message) for GUI notifications.
    """
    # Helper for notifications
    def _notify(level, msg):
        if notify:
            notify(level, msg)
        getattr(logging, level.lower(), logging.info)(msg)

    # Check for Inline Point Source processing request from env vars
    stack_groups = os.environ.get('STACK_GROUPS')
    griddesc = os.environ.get('GRIDDESC')

    # Also detect based on filenames if not explicit in env
    base_name = os.path.basename(ncf_path).lower()
    
    # If standard inln pattern detected but STACK_GROUPS missing, try to auto-locate
    if 'inln' in base_name:
        import re
        parent_dir = os.path.dirname(os.path.abspath(ncf_path))
        found_specific = False
         
        # Strategy 1: Look for date-specific match (YYYYMMDD or YYYYJJJ)
        # Using regex (20\d{6}|20\d{5}) to catch 8-digit Gregorian or 7-digit Julian dates
        match_date = re.search(r'(20\d{6}|20\d{5})', base_name)
        if match_date:
            date_str = match_date.group(1)
            try:
                candidates = [f for f in os.listdir(parent_dir) 
                                if 'stack_groups' in f.lower() 
                                and date_str in f 
                                and (f.endswith('.nc') or f.endswith('.ncf'))]
                if candidates:
                    stack_groups = os.path.join(parent_dir, candidates[0])
                    _notify('INFO', f"Auto-detected date-matched STACK_GROUPS file: {stack_groups}")
                    found_specific = True
            except Exception:
                pass
         
        # Strategy 2: Fallback to any stack_groups file if specific not found AND no global provided
        if not found_specific and not stack_groups:
            try:
                candidates = [f for f in os.listdir(parent_dir) if 'stack_groups' in f.lower() and (f.endswith('.nc') or f.endswith('.ncf'))]
                if candidates:
                    stack_groups = os.path.join(parent_dir, candidates[0])
                    _notify('INFO', f"Auto-detected STACK_GROUPS file (fallback): {stack_groups}")
            except Exception:
                pass

    # We can also detect inline by shape in the dataset
    is_inline_candidate = False
    try:
        with netCDF4.Dataset(ncf_path, 'r') as ds:
            dims = ds.dimensions
            # Inline uses ROW/COL dim names but one of them is 1.
            nrows = dims['ROW'].size if 'ROW' in dims else 0
            ncols = dims['COL'].size if 'COL' in dims else 0
            if (nrows == 1 and ncols > 1) or (ncols == 1 and nrows > 1):
                is_inline_candidate = True
    except Exception:
        pass
             
    if is_inline_candidate and (not stack_groups or not os.path.exists(stack_groups)):
        msg = "Inline file detected (1D structure) but STACK_GROUPS file could not be located. Processing aborted."
        _notify('ERROR', msg)
        raise FileNotFoundError(msg)

    if is_inline_candidate and stack_groups and os.path.exists(stack_groups):
        _notify('INFO', "Detected inline point sources. Converting to grid in-memory (Lazy Mode)...")
        try:
            # Lazy In-Memory Reading - For NetCDF input, ignore GRIDDESC to strictly use headers
            # User requirement: GRIDDESC is not used for netcdf/inline.
            logging.info(f"DEBUG: Calling read_inline_emissions_lazy for {ncf_path}")
            data_dict, nrows, ncols = read_inline_emissions_lazy(
                ncf_path, stack_groups, pollutants, 
                tstep_idx, layer_idx, tstep_op, layer_op, 
                griddesc_path=None
            )
            
            # Construct DataFrame manually
            
            # 1-based indices for compatibility
            row_idx, col_idx = np.indices((nrows, ncols))
            row_idx = row_idx + 1 
            col_idx = col_idx + 1 
            
            full_data = {
                'ROW': row_idx.flatten(),
                'COL': col_idx.flatten(),
            }
            # GRID_RC creation delayed to save memory/time or done via vectorized op if needed
            # But GUI expects it. 
            # Optimization: only create for subset? No, we filter later.
            # Let's add it only if needed?
            # full_data['GRID_RC'] = ...
            
            full_data.update(data_dict)
            
            df = pd.DataFrame(full_data)
            
            # Filter non-zero
            # Identify pollutant cols
            pol_cols = list(data_dict.keys())
            
            if pol_cols:
                # Vectorized numeric check (any non-zero)
                # mask = (df[pol_cols] != 0).any(axis=1)
                
                # Log pruning (DISABLED to fix full-grid animation alignment)
                # n_total = len(df)
                # df = df[mask].copy()
                # n_kept = len(df)
                # if n_total > n_kept:
                #      _notify('INFO', f"Inline Sparse: Pruned {n_total - n_kept} empty records. Keeping {n_kept}.")
                pass
            
            if df.empty:
                _notify('WARNING', "Inline processing resulted in empty dataframe (no non-zero emissions).")
                return gpd.GeoDataFrame({'geometry': []}, crs='EPSG:4326')

            # Re-get 'gi' from cache or re-call
            _, _, _, gi = _get_inline_mapping(ncf_path, stack_groups, griddesc, None)
            
            if gi['xcell'] == 0:
                _notify('ERROR', "Grid XCELL is 0. Map coordinates will be invalid.")
                 
            # Create Grid Polygons
            from shapely.geometry import box
            
            # df has ROW, COL (1-based).
            # Convert to 0-based for geometry gen
            r_0 = df['ROW'].values - 1
            c_0 = df['COL'].values - 1
            
            x0 = gi['xorig'] + c_0 * gi['xcell']
            y0 = gi['yorig'] + r_0 * gi['ycell']
            x1 = x0 + gi['xcell']
            y1 = y0 + gi['ycell']
            
            # Vectorized geometry (Shapely 2.0+) with timing
            import time
            _t_geom = time.time()
            try:
                import shapely
                geoms = shapely.box(x0, y0, x1, y1) if hasattr(shapely, 'box') else None
                if geoms is None: raise AttributeError()
            except:
                from shapely.geometry import box
                geoms = [box(x0[i], y0[i], x1[i], y1[i]) for i in range(len(df))]
            _notify('INFO', f"Inline geometry: {time.time()-_t_geom:.2f}s for {len(df)} cells")
            
            logging.info("DEBUG: Creating GeoDataFrame...")
            gdf = gpd.GeoDataFrame(df, geometry=geoms)
            
            # Add GRID_RC after filtering to be faster
            logging.info("DEBUG: Adding GRID_RC...")
            try:
                gdf['GRID_RC'] = gdf['ROW'].astype(str) + "_" + gdf['COL'].astype(str)
            except Exception:
                gdf['GRID_RC'] = [f"{r:d}_{c:d}" for r, c in zip(gdf['ROW'], gdf['COL'])]
            
            # Set CRS
            proj_obj = get_proj_object_from_info(gi)
            gdf.crs = proj_obj.srs
            
            # Attach grid info
            gdf.attrs['_smk_grid_info'] = {
                'xorig': gi['xorig'],
                'yorig': gi['yorig'],
                'xcell': gi['xcell'],
                'ycell': gi['ycell'],
                'ncols': gi['ncols'],
                'nrows': gi['nrows'],
                'proj_str': proj_obj.srs
            }
            gdf.attrs['_smk_is_native'] = True

            # Set Attributes
            gdf.attrs['source_type'] = 'inline_point_lazy'
            gdf.attrs['stack_groups_path'] = stack_groups
            gdf.attrs['original_ncf_path'] = ncf_path

            # Extract variable metadata (units/long_name)
            var_metadata = {}
            try:
                with netCDF4.Dataset(ncf_path, 'r') as ds:
                    for pol in data_dict.keys():
                        if pol in ds.variables:
                            var_obj = ds.variables[pol]
                            lname = getattr(var_obj, 'long_name', '')
                            units = getattr(var_obj, 'units', '')
                            # Clean up string
                            if isinstance(lname, bytes): lname = lname.decode('utf-8', 'ignore')
                            if isinstance(units, bytes): units = units.decode('utf-8', 'ignore')
                            var_metadata[pol] = {
                                'long_name': str(lname).strip(),
                                'units': str(units).strip()
                            }
            except Exception as e:
                _notify('WARNING', f"Could not read metadata for inline file: {e}")

            gdf.attrs['variable_metadata'] = var_metadata
            
            _notify('INFO', f"Inline Lazy Load Complete: {len(gdf)} cells with emissions.")
            
            return gdf

        except Exception as e:
            import traceback
            traceback.print_exc()
            _notify('ERROR', f"Failed to process as inline emissions (Lazy): {e}. Falling back to standard grid reading.")
            # Fallback to standard reading (which will likely result in strip maps, but at least not crash)
    # --- Optimized xarray/dask Path ---
    try:
        import xarray as xr
        import dask
        
        # Open dataset lazily or use provided handle
        ds = xr_ds if xr_ds is not None else xr.open_dataset(ncf_path, chunks={})
        
        # 1. Identify pollutants (variables with ROW and COL dims)
        available_vars = []
        for vname in ds.data_vars:
            if vname == 'TFLAG': continue
            if 'ROW' in ds[vname].dims and 'COL' in ds[vname].dims:
                available_vars.append(vname)
        
        # 2. Extract metadata
        var_metadata = {}
        for pol in available_vars:
            v_attr = ds[pol].attrs
            var_metadata[pol] = {
                'long_name': str(v_attr.get('long_name', '')).strip(),
                'units': str(v_attr.get('units', '')).strip()
            }
            
        nrows = ds.sizes['ROW']
        ncols = ds.sizes['COL']
        
        # 3. Handle Initial Load (Skeleton Case or Detect All)
        if pollutants is None:
            if lazy:
                _notify('INFO', f"Lazy-loading NetCDF skeleton with {len(available_vars)} variables...")
                
                # Build Row/Col structure
                row_idx, col_idx = np.indices((nrows, ncols), dtype=np.int32)
                row_idx = row_idx.flatten() + 1
                col_idx = col_idx.flatten() + 1
                
                df = pd.DataFrame({'ROW': row_idx, 'COL': col_idx})
                df['GRID_RC'] = df['ROW'].astype(str) + '_' + df['COL'].astype(str)
                
                # Attach metadata
                df.attrs['variable_metadata'] = var_metadata
                df.attrs['available_pollutants'] = available_vars
                df.attrs['pollutants'] = available_vars # compat
                df.attrs['source_type'] = 'gridded_netcdf'
                df.attrs['_smk_is_native'] = True
                df.attrs['_smk_xr_ds'] = ds
                
                try:
                    coord_params, grid_params = read_ncf_grid_params(ncf_path)
                    proj_obj = get_proj_object_from_info({
                        'proj_type': coord_params[0], 'p_alp': coord_params[1],
                        'p_bet': coord_params[2], 'p_gam': coord_params[3],
                        'xcent': coord_params[4], 'ycent': coord_params[5]
                    })
                    df.attrs['_smk_grid_info'] = {
                        'xorig': grid_params[1], 'yorig': grid_params[2],
                        'xcell': grid_params[3], 'ycell': grid_params[4],
                        'ncols': grid_params[5], 'nrows': grid_params[6],
                        'proj_str': proj_obj.srs
                    }
                except: pass
                
                return df
            else:
                # Eager load all detected pollutants (standard for batch mode)
                pollutants = available_vars

        # 4. Handle Specific Pollutant Extraction
        _notify('INFO', f"Extracting {len(pollutants)} pollutants using xarray...")
        
        # Slice Time
        if 'TSTEP' in ds.sizes:
            if tstep_idx is not None:
                # Ensure valid index
                t_idx = min(max(0, tstep_idx), ds.sizes['TSTEP']-1)
                ds_sliced = ds.isel(TSTEP=t_idx)
            else:
                if tstep_op == 'mean': ds_sliced = ds.mean(dim='TSTEP')
                elif tstep_op == 'sum': ds_sliced = ds.sum(dim='TSTEP')
                elif tstep_op == 'max': ds_sliced = ds.max(dim='TSTEP')
                elif tstep_op == 'min': ds_sliced = ds.min(dim='TSTEP')
                else: ds_sliced = ds.mean(dim='TSTEP')
        else:
            ds_sliced = ds

        # Slice Layers
        if 'LAY' in ds_sliced.sizes:
            if layer_idx is not None:
                l_idx = min(max(0, layer_idx), ds_sliced.sizes['LAY']-1)
                ds_sliced = ds_sliced.isel(LAY=l_idx)
            else:
                if layer_op == 'mean': ds_sliced = ds_sliced.mean(dim='LAY')
                elif layer_op == 'sum': ds_sliced = ds_sliced.sum(dim='LAY')
                elif layer_op == 'max': ds_sliced = ds_sliced.max(dim='LAY')
                elif layer_op == 'min': ds_sliced = ds_sliced.min(dim='LAY')
                else: ds_sliced = ds_sliced.isel(LAY=0) # Default to surface
        
        # Prepare data dictionary
        data_dict = {}
        for pol in pollutants:
            if pol in ds_sliced:
                # Compute and convert to 1D
                data_dict[pol] = ds_sliced[pol].values.astype(np.float32).flatten()
        
        # Build DataFrame
        row_idx, col_idx = np.indices((nrows, ncols), dtype=np.int32)
        df = pd.DataFrame({
            'ROW': row_idx.flatten() + 1,
            'COL': col_idx.flatten() + 1
        })
        df['GRID_RC'] = df['ROW'].astype(str) + '_' + df['COL'].astype(str)
        
        # Performance: Use assign or concat to avoid fragmenting the dataframe with many columns
        if data_dict:
            # We can convert the dict of arrays directly to a dataframe and join
            pol_df = pd.DataFrame(data_dict)
            df = pd.concat([df, pol_df], axis=1)
                
        df.attrs['variable_metadata'] = var_metadata
        df.attrs['_smk_is_native'] = True
        
        try:
            coord_params, grid_params = read_ncf_grid_params(ncf_path)
            proj_obj = get_proj_object_from_info({
                'proj_type': coord_params[0], 'p_alp': coord_params[1],
                'p_bet': coord_params[2], 'p_gam': coord_params[3],
                'xcent': coord_params[4], 'ycent': coord_params[5]
            })
            df.attrs['_smk_grid_info'] = {
                'xorig': grid_params[1], 'yorig': grid_params[2],
                'xcell': grid_params[3], 'ycell': grid_params[4],
                'ncols': grid_params[5], 'nrows': grid_params[6],
                'proj_str': proj_obj.srs
            }
        except: pass
        
        return df

    except Exception as xr_err:
        _notify('WARNING', f"Xarray lazy load failed or unavailable: {xr_err}. Falling back to netCDF4.")

    data_dict = {}
    var_metadata = {}
    
    with netCDF4.Dataset(ncf_path, 'r') as ds:
        # Determine pollutants to read
        available_vars = set(ds.variables.keys())
        # Exclude TFLAG, TSTEP, etc.
        if pollutants is None:
            # Simple heuristic: floats with (TSTEP, LAY, ROW, COL) or (TSTEP, VAR, DATE-TIME)
            # Actually, standard IOAPI vars are (TSTEP, LAY, ROW, COL).
            # TFLAG is (TSTEP, VAR, DATE-TIME).
            candidates = []
            for vname in available_vars:
                if vname == 'TFLAG': continue
                dims = ds.variables[vname].dimensions
                # Check for standard IOAPI dims
                if 'TSTEP' in dims and 'ROW' in dims and 'COL' in dims:
                    candidates.append(vname)
            pollutants = candidates
        
        # Grid dimensions
        nrows = ds.dimensions['ROW'].size
        ncols = ds.dimensions['COL'].size
        
        # Prepare ROW and COL arrays for dataframe
        # Create meshgrid of indices. Note: 1-based indexing for GRID_RC usually.
        # rows 1..NROWS, cols 1..NCOLS
        # But wait, create_ncf_domain_gdf creates ROW/COL as proper lists.
        # We need a dataframe with ROW, COL, GRID_RC, POLL1, POLL2...
        
        # Doing this efficiently without blowing up memory?
        # A single variable is T*L*R*C.
        # If we sum over time and select layer 0 -> R*C matrix.
        
        for pol in pollutants:
            if pol not in available_vars:
                continue
            
            var_obj = ds.variables[pol]
            
            # Capture metadata
            try:
                lname = getattr(var_obj, 'long_name', '')
                units = getattr(var_obj, 'units', '')
                # Clean up string
                if isinstance(lname, bytes): lname = lname.decode('utf-8', 'ignore')
                if isinstance(units, bytes): units = units.decode('utf-8', 'ignore')
                
                var_metadata[pol] = {
                    'long_name': str(lname).strip(),
                    'units': str(units).strip()
                }
            except Exception:
                pass

            # Assuming shape (TSTEP, LAY, ROW, COL)
            # Read and aggregate
            # Optimization: Use slicing to avoid reading full variable into memory
            dims = var_obj.dimensions
            slices = [slice(None)] * len(dims)
            
            # Handle Time Slice
            if 'TSTEP' in dims:
                axis_idx = dims.index('TSTEP')
                if tstep_idx is not None:
                    ts_target = tstep_idx
                    if ts_target >= var_obj.shape[axis_idx]:
                        ts_target = 0
                    slices[axis_idx] = ts_target

            # Handle Layer Slice
            if 'LAY' in dims:
                axis_idx = dims.index('LAY')
                if layer_idx is not None:
                    # If layer_op is select, we slice. If sum/mean, we might slice if logic implies single layer
                    # But current API implies layer_idx is used unless None.
                    # If layer_idx is provided, we use it. Aggregation happens if layer_idx is None.
                    if var_obj.shape[axis_idx] <= layer_idx:
                        layer_idx_act = 0
                    else:
                        layer_idx_act = layer_idx
                    slices[axis_idx] = layer_idx_act

            # Read data subset
            try:
                data = var_obj[tuple(slices)]
            except Exception:
                # Fallback to full read if slicing fails (unlikely)
                data = var_obj[:]

            # Now reduce remaining dimensions if needed
            # Re-evaluate current dimensions after slicing
            # Slicing removes the dimension if integer index used.
            current_dims = []
            for i, d in enumerate(dims):
                if isinstance(slices[i], slice):
                    current_dims.append(d)
                # else: dimension removed

            # Reduce Time (if not sliced)
            if 'TSTEP' in current_dims:
                axis_idx = current_dims.index('TSTEP')
                if tstep_op == 'mean':
                    data = np.mean(data, axis=axis_idx)
                elif tstep_op == 'max':
                    data = np.max(data, axis=axis_idx)
                elif tstep_op == 'min':
                    data = np.min(data, axis=axis_idx)
                else:
                    data = np.sum(data, axis=axis_idx)
                # Shift dims
                current_dims.pop(axis_idx)
            
            # Reduce Layer (if not sliced)
            if 'LAY' in current_dims:
                try:
                    axis_idx = current_dims.index('LAY')
                    if layer_op == 'mean':
                        data = np.mean(data, axis=axis_idx)
                    elif layer_op == 'sum':
                        data = np.sum(data, axis=axis_idx)
                    elif layer_op == 'max':
                        data = np.max(data, axis=axis_idx)
                    elif layer_op == 'min':
                        data = np.min(data, axis=axis_idx)
                    else: 
                        # Default to layer 0 if op is confusing
                        data = np.take(data, 0, axis=axis_idx)
                except ValueError:
                    pass

            # Now data should be (ROW, COL) - or (ROW, COL) depending on dim order
            # IOAPI is usually (TSTEP, LAY, ROW, COL).
            # after TSTEP removal: (LAY, ROW, COL)
            # after LAY removal: (ROW, COL)
            # Check shape
            if data.shape != (nrows, ncols):
                # Try transpose if shape is (ncols, nrows) which shouldn't happen for standard IOAPI
                if data.shape == (ncols, nrows):
                    data = data.T
                else:
                    logging.warning(f"Variable {pol} shape mismatch: {data.shape} vs ({nrows}, {ncols})")
                    continue
            
            # Use float32 to save memory
            data_dict[pol] = data.astype(np.float32).flatten()

    # Construct DataFrame
    # 1-based indices for compatibility
    row_idx, col_idx = np.indices((nrows, ncols))
    row_idx = row_idx + 1 # 1-based
    col_idx = col_idx + 1 # 1-based
    
    # Try Lazy/Sparse Grid Generation for Standard NetCDF (Speeds up initial loading)
    try:
        if data_dict:
            coord_params_lz, grid_params_lz = read_ncf_grid_params(ncf_path)
            # grid_params: (gdnam, xorig, yorig, xcell, ycell, ncols, nrows, nthik)
            xorig_lz, yorig_lz, xcell_lz, ycell_lz = grid_params_lz[1], grid_params_lz[2], grid_params_lz[3], grid_params_lz[4]
             
            if xcell_lz > 0 and ycell_lz > 0:
                flat_rows_lz = row_idx.flatten()
                flat_cols_lz = col_idx.flatten()
                 
                full_data_lz = {'ROW': flat_rows_lz, 'COL': flat_cols_lz}
                full_data_lz.update(data_dict)
                df_lz = pd.DataFrame(full_data_lz)
                 
                # Filter Non-Zero (Sparse optimization)
                # DISABLED: User requested to keep zero-value cells for correct domain average calculations.
                # Previously we pruned cells where all pollutants were 0.
                _notify('INFO', f"Lazy Grid Load: keeping all {len(df_lz)} cells (including zeros).")
                 
                # pol_cols_lz = list(data_dict.keys())
                # if pol_cols_lz:
                #      # Use ANY non-zero check instead of SUM > 0 to preserve negative emissions and avoid cancellation
                #      mask_lz = (df_lz[pol_cols_lz] != 0).any(axis=1)
                #      
                #      # Debug Logging for specific cells if needed (e.g. 50_175)
                #      # row_target, col_target = 50, 175
                #      # if 'ROW' in df_lz.columns:
                #      #     chk = df_lz[(df_lz['ROW']==row_target) & (df_lz['COL']==col_target)]
                #      #     if not chk.empty:
                #      #          chk_vals = chk[pol_cols_lz].values
                #      #          logging.info(f"DEBUG: Cell {row_target}_{col_target} Values: {chk_vals} Mask: {mask_lz.loc[chk.index].values}")
                #
                #      df_lz = df_lz[mask_lz].copy()

                # n_pruned = len(flat_rows_lz) - len(df_lz)
                # if n_pruned > 0:
                #     _notify('INFO', f"Sparse Grid: Pruned {n_pruned} empty cells (zeros). Keeping {len(df_lz)} cells.")
                 
                if df_lz.empty:
                        return gpd.GeoDataFrame(df_lz, geometry=[], crs=None)
                 
                # Generate Geometry
                r_0 = df_lz['ROW'].values - 1
                c_0 = df_lz['COL'].values - 1
                 
                x0 = xorig_lz + c_0 * xcell_lz
                y0 = yorig_lz + r_0 * ycell_lz
                x1 = x0 + xcell_lz
                y1 = y0 + ycell_lz
                 
                from shapely.geometry import box
                # Vectorized geometry (Shapely 2.0+) with timing
                import time
                _t_geom = time.time()
                try:
                    import shapely
                    geoms = shapely.box(x0, y0, x1, y1) if hasattr(shapely, 'box') else None
                    if geoms is None: raise AttributeError()
                except:
                    from shapely.geometry import box
                    geoms = [box(x0[i], y0[i], x1[i], y1[i]) for i in range(len(df_lz))]
                _notify('INFO', f"Grid geometry: {time.time()-_t_geom:.2f}s for {len(df_lz)} cells")
                 
                gi_lz = {
                    'proj_type': coord_params_lz[0],
                    'p_alp': coord_params_lz[1],
                    'p_bet': coord_params_lz[2],
                    'p_gam': coord_params_lz[3],
                    'xcent': coord_params_lz[4],
                    'ycent': coord_params_lz[5]
                }
                try:
                    proj_obj_lz = get_proj_object_from_info(gi_lz)
                    crs_lz = proj_obj_lz.srs
                except:
                    crs_lz = None

                gdf_lz = gpd.GeoDataFrame(df_lz, geometry=geoms, crs=crs_lz)
                 
                # Standard GRID_RC creation
                try:
                    gdf_lz['GRID_RC'] = gdf_lz['ROW'].astype(str) + '_' + gdf_lz['COL'].astype(str)
                except:
                    gdf_lz['GRID_RC'] = [f"{r:d}_{c:d}" for r, c in zip(gdf_lz['ROW'], gdf_lz['COL'])]
                 
                gdf_lz.attrs['variable_metadata'] = var_metadata
                gdf_lz.attrs['source_type'] = 'gridded_lazy'
                gdf_lz.attrs['_smk_is_native'] = True
                
                # Attach grid info
                try:
                    gdf_lz.attrs['_smk_grid_info'] = {
                        'xorig': xorig_lz,
                        'yorig': yorig_lz,
                        'xcell': xcell_lz,
                        'ycell': ycell_lz,
                        'ncols': grid_params_lz[5],
                        'nrows': grid_params_lz[6],
                        'proj_str': crs_lz
                    }
                except: pass
                 
                _notify('INFO', f"Lazy Grid Load (Standard): {len(gdf_lz)} active cells.")
                return gdf_lz

    except Exception as e:
        pass
        # Fallback to original logic

    # Optimize DataFrame creation
    flat_rows = row_idx.flatten()
    flat_cols = col_idx.flatten()
    
    # Prepare all data in a single dictionary to avoid fragmentation warnings
    full_data = {
        'ROW': flat_rows,
        'COL': flat_cols,
    }
    full_data.update(data_dict)
    
    df = pd.DataFrame(full_data)
    
    # Vectorized string creation (approx 10x faster than apply)
    try:
        df['GRID_RC'] = df['ROW'].astype(str) + '_' + df['COL'].astype(str)
    except Exception:
        df['GRID_RC'] = [f"{r}_{c}" for r, c in zip(flat_rows, flat_cols)]
    
    df.attrs['variable_metadata'] = var_metadata
    
    # Attach grid info
    df.attrs['_smk_is_native'] = True
    try:
        coord_params, grid_params = read_ncf_grid_params(ncf_path)
        # grid_params: (gdnam, xorig, yorig, xcell, ycell, ncols, nrows, nthik)
        proj_obj = get_proj_object_from_info({
            'proj_type': coord_params[0],
            'p_alp': coord_params[1],
            'p_bet': coord_params[2],
            'p_gam': coord_params[3],
            'xcent': coord_params[4],
            'ycent': coord_params[5]
        })
        df.attrs['_smk_grid_info'] = {
            'xorig': grid_params[1],
            'yorig': grid_params[2],
            'xcell': grid_params[3],
            'ycell': grid_params[4],
            'ncols': grid_params[5],
            'nrows': grid_params[6],
            'proj_str': proj_obj.srs
        }
    except Exception:
        pass
                 
    return df

def get_ncf_timeseries(
    ncf_path: str,
    pollutant: str,
    row_indices: list,
    col_indices: list,
    layer_idx: int = 0,
    layer_op: str = 'select',
    op: str = 'sum',
    stack_groups_path: str = None
):
    """
    Extract time series for specific grid cells.
    row_indices/col_indices: 0-based index pairs.
    op: 'sum' or 'mean' (spatial aggregation).
    Returns dict with 'times', 'values', 'units'.
    """

    # --- INLINE Support ---
    if stack_groups_path and os.path.exists(stack_groups_path):
        # Use cached mapping directly
        try:
            # Need griddesc? Or use ncf header?
            # _get_inline_mapping takes griddesc_path. Usually passed as None here if unknown.
            # We rely on previous caching or header inference.
            valid_indices, v_rows, v_cols, gi = _get_inline_mapping(ncf_path, stack_groups_path, None, None)
             
            # Calculate source mask for requested cells
            # v_rows/v_cols align with valid_indices.
            # We want sources where (row, col) in requested list.
             
            # Convert request to set of (r, c)
            req_cells = set(zip(row_indices, col_indices))
             
            # Find matching indices in the valid_sources arrays
            # Iterate raw
            match_mask = []
            for r, c in zip(v_rows, v_cols):
                if (r, c) in req_cells:
                        match_mask.append(True)
                else:
                        match_mask.append(False)
            match_mask = np.array(match_mask, dtype=bool)
             
            target_src_indices = valid_indices[match_mask]
             
            if len(target_src_indices) == 0:
                return None
                  
            # Read from INLN
            with netCDF4.Dataset(ncf_path, 'r') as ds:
                # Get Time
                tflag_vals = []
                if 'TFLAG' in ds.variables:
                        tf = ds.variables['TFLAG']
                        if tf.ndim == 3: raw_t = tf[:, 0, :]
                        else: raw_t = tf[:]
                        for i in range(raw_t.shape[0]):
                            tflag_vals.append(f"{int(raw_t[i,0])}_{int(raw_t[i, 1]):06d}")
                  
                if pollutant not in ds.variables: return None
                var = ds.variables[pollutant]
                units = getattr(var, 'units', '')
                  
                # Read only target sources across all time
                # Shape (T, LAY, NSRC) or similar
                # Optimization: Read only columns?
                # NetCDF4 supports boolean indexing? No, but integer indexing yes.
                  
                # We need to handle layers too.
                # Slicing for layers:
                slices = [slice(None)] * var.ndim
                l_axis = var.dimensions.index('LAY') if 'LAY' in var.dimensions else -1
                  
                if l_axis >= 0:
                        if layer_idx is not None:
                            slices[l_axis] = layer_idx
                        elif layer_op == 'select':
                            slices[l_axis] = 0
                        # else: keep all layers, reduce later.
                  
                # Slicing columns (Source Dimension)
                # Identify the source dimension (ROW or COL usually)
                src_dim_idx = -1
                dims_list = var.dimensions
                  
                # Debugging
                logging.info(f"TS Extraction. Pollutant: {pollutant}, Dims: {dims_list}")
                for d in dims_list:
                        logging.info(f"Dim {d}: {ds.dimensions[d].size}")

                # Strategy: Find dimension that is not TSTEP/LAY and has size > 1
                # Or strict check for ROW/COL
                for i, dname in enumerate(dims_list):
                        if dname in ['TSTEP', 'LAY', 'TFLAG', 'DATE-TIME']:
                            continue
                        # If known source dim candidates
                        if dname in ['ROW', 'COL', 'SRC', 'NSRC']:
                            if ds.dimensions[dname].isunlimited() or ds.dimensions[dname].size >= 1:
                                # Prefer the one that is > 1 if multiple match? 
                                # If COL=1 and ROW=N, we want ROW.
                                if ds.dimensions[dname].size > 1:
                                        src_dim_idx = i
                                        break
                                # If both are 1 (1 source), picking either is fine if index is 0. 
                                # But valid_indices will be [0].
                                if src_dim_idx == -1: src_dim_idx = i
                  
                # Fallback: Last dimension
                if src_dim_idx == -1:
                        src_dim_idx = var.ndim - 1
                  
                logging.info(f"Selected Source Dim Index: {src_dim_idx} ({dims_list[src_dim_idx]})")
                slices[src_dim_idx] = target_src_indices
                  
                # Read
                data = var[tuple(slices)]
                  
                # Reduce Layer if needed
                if l_axis >= 0 and layer_idx is None and layer_op != 'select':
                        # data has layer dim. Where is it now?
                        # It wasn't removed.
                        # Index in data might have shifted if dims removed? No.
                        l_axis_now = l_axis
                        if layer_op == 'mean': data = np.mean(data, axis=l_axis_now)
                        elif layer_op == 'max': data = np.max(data, axis=l_axis_now)
                        elif layer_op == 'min': data = np.min(data, axis=l_axis_now)
                        else: data = np.sum(data, axis=l_axis_now)

                # Now data is (T, N_Selected_Sources)
                # Handle singleton dimensions (e.g. COL=1)
                logging.debug(f"[TS Extraction] Raw Data Shape: {data.shape}")
                
                if data.ndim > 2:
                        # Flatten trailing singletons
                        if data.shape[-1] == 1:
                            data = data.reshape(data.shape[0], -1)
                  
                # If op is 'mean', we likely want mean of total? Or mean of individual?
                # Standard logic was just to return aggregated line.
                # New logic: Return Total PLUS Individual components if feasible.
                  
                series_total = np.sum(data, axis=-1)
                logging.debug(f"[TS Extraction] Series Total Shape: {series_total.shape}")
                  
                # Create result structure
                # To avoid breaking existing consumers, we can return 'values' as the total, 
                # and add a new key 'components' or just return a dict in 'values' 
                # (consumer must handle)
                  
                # The user specifically requested multiple lines.
                # Let's read metadata for labels
                components = {}
                try:
                        with netCDF4.Dataset(stack_groups_path, 'r') as sg:
                            # Slicing: [0, 0, indices, 0] typically for (T, L, R, C)
                            # Or check TSTEP size
                            t_idx = 0
                            l_idx = 0
                            c_idx = 0
                            
                            # Read ISTACK
                            if 'ISTACK' in sg.variables:
                                istack_var = sg.variables['ISTACK']
                                # Assuming (T, L, R, C)
                                # We need to slice orthogonal? target_src_indices is list of ints.
                                # NetCDF4 supports v[0, 0, [list], 0]
                                 
                                # Build slice
                                sg_slices = [0] * istack_var.ndim
                                # Find ROW dim
                                row_dim_idx = -2 # default heuristic
                                for i, d in enumerate(istack_var.dimensions):
                                        if d == 'ROW': row_dim_idx = i; break
                                 
                                sg_slices[row_dim_idx] = target_src_indices
                                 
                                # Ensure others are 0 or slice(None) if singleton?
                                # Better to read carefully.
                                # If we just assume standard structure:
                                vals_istack = istack_var[0, 0, target_src_indices, 0]
                            else:
                                vals_istack = target_src_indices # fallback
                            
                            # Read IFIP if available
                            vals_ifip = None
                            if 'IFIP' in sg.variables:
                                vals_ifip = sg.variables['IFIP'][0, 0, target_src_indices, 0]
                                 
                            # Construct labels
                            # data shape (T, N_srcs)
                            if data.ndim == 2 and data.shape[1] == len(target_src_indices):
                                for i in range(len(target_src_indices)):
                                        stk = vals_istack[i]
                                        fip = vals_ifip[i] if vals_ifip is not None else '?'
                                        label = f"Stk {stk} (FIPS {fip})"
                                        components[label] = data[:, i].tolist()
                                      
                except Exception as e:
                        logging.warning(f"Could not read stack details: {e}")
                        # Fallback: labeled by index
                        for i in range(data.shape[1]):
                            components[f"Source {i+1}"] = data[:, i].tolist()

                # Combine into 'values' as a dict, including Total
                values_out = {'Total': series_total.tolist()}
                values_out.update(components)
                       
                return {'times': tflag_vals, 'values': values_out, 'units': units}

        except Exception as e:
            logging.error(f"Inline TS failed: {e}")
            return None

    # --- Standard NetCDF ---
    with netCDF4.Dataset(ncf_path, 'r') as ds:
        # Get TFLAG
        tflag_vals = []
        if 'TFLAG' in ds.variables:
            tf = ds.variables['TFLAG']
            if tf.ndim == 3:
                # (TSTEP, VAR, DATE-TIME)
                raw_t = tf[:, 0, :]
            else:
                raw_t = tf[:]
            for i in range(raw_t.shape[0]):
                yd = int(raw_t[i, 0])
                hms = int(raw_t[i, 1])
                tflag_vals.append(f"{yd}_{hms:06d}")
        
        if pollutant not in ds.variables:
            return None
        var = ds.variables[pollutant]
        dims = var.dimensions
        try:
            units = getattr(var, 'units', '').strip()
            if isinstance(units, bytes): units = units.decode('utf-8', 'ignore')
        except:
            units = ''
        
        # Read full variable - Optimized
        # Identify Layer axis and slice if possible to avoid reading all layers
        dims = var.dimensions
        slices = [slice(None)] * len(dims)
        l_axis = dims.index('LAY') if 'LAY' in dims else None
        
        reduce_layer_later = False
        if l_axis is not None:
            # If specific layer requested (and not aggregating over layers)
            if layer_idx is not None:
                target = layer_idx
                if target >= var.shape[l_axis]: target = 0
                slices[l_axis] = target
            elif layer_op == 'select':
                # Default layer 0
                slices[l_axis] = 0
            else:
                # Need to sum/mean over all layers -> must read all (or loop)
                reduce_layer_later = True
        
        full_data = var[tuple(slices)]
        
        # Reduce Layer first if we read all layers
        if reduce_layer_later:
            # The data still HAS the layer dimension which is now possibly shifted if TSTEP was sliced (it wasn't here)
            # Slicing preserved rank only if slice(None) used. Integer indexing removed rank.
            # We need to find which axis corresponds to LAY in 'full_data'
            # Since we didn't index TSTEP or ROW/COL, the relative order is preserved.
            # We need to find the index of 'LAY' in 'dims'
            axis_idx = dims.index('LAY') # This assumes other dims were not integer-indexed.
            # But wait, did we integer-index anything? NO, slices was all slice(None).
             
            if layer_op == 'sum':
                full_data = np.sum(full_data, axis=axis_idx)
            elif layer_op == 'mean':
                full_data = np.mean(full_data, axis=axis_idx)
            else:
                full_data = np.take(full_data, 0, axis=axis_idx)
        
        # Now we usually have (TSTEP, ROW, COL) order. TSTEP might not exist if data is 2D.
        # But if we want time series, we need TSTEP.
        # If no TSTEP, returns single value?
        
        # Squeeze out dropped layer axis implicitly handled by numpy/pandas reshape? 
        # Actually np.take reduces dimension.
        # We need to find new axes indices for ROW/COL after layer removal.
        # But fancy indexing requires us to construct the full indexer tuple.
        
        # BETTER STRATEGY: Use fancy indexing on the *reduced* array.
        # Or fancy indexing on the original array directly if possible? No, we needed to reduce layer potentially.
        
        # Re-detect axes on full_data (which is now reduced by layer, so rank is N-1)
        # We assume order was (T, L, R, C) -> (T, R, C)
        
        if full_data.ndim == 3:
            # Assume (T, R, C)
            try:
                # Fancy index: data[:, rows, cols]
                r_idx = np.array(row_indices, dtype=int)
                c_idx = np.array(col_indices, dtype=int)
                
                # Check bounds
                if len(r_idx) == 0: return None
                
                # Assume dim 0 is Time, 1 is Row, 2 is Col
                selected = full_data[:, r_idx, c_idx] # Shape (T, N_Cells)
                
                if op == 'sum':
                    series = np.sum(selected, axis=1)
                else:
                    series = np.mean(selected, axis=1)
                
                return {'times': tflag_vals, 'values': series.tolist(), 'units': units}
            except Exception as e:
                logging.error(f"Time series extraction failed: {e}")
                return None
        else:
            return None

def get_ncf_animation_data(
    ncf_path: str,
    pollutant: str,
    row_indices: list,
    col_indices: list,
    layer_idx: int = 0,
    layer_op: str = 'select',
    stack_groups_path: str = None
):
    """
    Extract data for animation: Full time series for each specified cell.
    Returns: {
        'times': [str], 
        'values': np.array shape (n_times, n_cells),
        'units': str
    }
    """
    # --- INLINE Support ---
    if stack_groups_path and os.path.exists(stack_groups_path):
        try:
            valid_indices, v_rows, v_cols, gi = _get_inline_mapping(ncf_path, stack_groups_path, None, None)
             
            # Vectorization Strategy:
            # 1. Build a map of Requested Cells -> Output Index (0..N-1)
            #    This allows O(1) lookups to see if a source contributes to our output.
            #    Crucially, this is built ONCE, not iterated per frame.
            cell_out_map = { (int(r), int(c)): i for i, (r, c) in enumerate(zip(row_indices, col_indices)) }
             
            # 2. Identify Sources that fall into ANY requested cell
            #    and map them to their target output indices.
            #    one source -> one cell (usually)
            needed_src_indices = []
            # Map Global_Source_Index -> List of [Output_Cell_Indices]
            # (Allows for sources contributing to multiple cells if logic ever changed, though standard point is 1:1)
            src_to_out_map = {} 
             
            # Iterate all sources in the file (or valid subset)
            # This loop scale is N_Sources (e.g. 5000 to 1M), not N_GridCells (150k).
            # Efficient if N_Sources isn't massive. If Massive, typically blocked by IO anyway.
            for g_idx, r, c in zip(valid_indices, v_rows, v_cols):
                target = cell_out_map.get((int(r), int(c)))
                if target is not None:
                    needed_src_indices.append(g_idx)
                    if g_idx not in src_to_out_map:
                        src_to_out_map[g_idx] = []
                    src_to_out_map[g_idx].append(target)
             
            all_needed_srcs = sorted(list(set(needed_src_indices)))
             
            logging.info(f"Inline Animation: Found {len(all_needed_srcs)} contributing sources for requested cells.")

            if not all_needed_srcs:
                logging.warning("Inline Animation: No matching sources found for requested cells.")
                # Return Zeroes with correct shape
                # Read time dims for shape
                with netCDF4.Dataset(ncf_path, 'r') as ds:
                        tflag_vals = []
                        if 'TFLAG' in ds.variables:
                            tf = ds.variables['TFLAG']
                            if tf.ndim == 3: raw_t = tf[:, 0, :]
                            else: raw_t = tf[:]
                            for i in range(raw_t.shape[0]):
                                tflag_vals.append(f"{int(raw_t[i,0])}_{int(raw_t[i, 1]):06d}")
                        else:
                            dim_t = ds.dimensions.get('TSTEP')
                            if dim_t: tflag_vals = [f"T{i}" for i in range(dim_t.size)]
                      
                        n_times = len(tflag_vals) if tflag_vals else (ds.dimensions['TSTEP'].size if 'TSTEP' in ds.dimensions else 0)
                        if n_times == 0: return None
                 
                res_arr = np.zeros((n_times, len(row_indices)))
                return {'times': tflag_vals, 'values': res_arr, 'units': ''}
             
            # 3. Batch Read Data for identified sources
            with netCDF4.Dataset(ncf_path, 'r') as ds:
                # Time Metadata
                tflag_vals = []
                if 'TFLAG' in ds.variables:
                        tf = ds.variables['TFLAG']
                        if tf.ndim == 3: raw_t = tf[:, 0, :]
                        else: raw_t = tf[:]
                        for i in range(raw_t.shape[0]):
                            tflag_vals.append(f"{int(raw_t[i,0])}_{int(raw_t[i, 1]):06d}")
                else:
                        dim_t = ds.dimensions.get('TSTEP')
                        if dim_t: tflag_vals = [f"T{i}" for i in range(dim_t.size)]
                  
                if pollutant not in ds.variables: return None
                var = ds.variables[pollutant]
                units = getattr(var, 'units', '')
                  
                # Identify Source Dim
                src_dim_idx = -1
                dims_list = var.dimensions
                for i, dname in enumerate(dims_list):
                        if dname in ['TSTEP', 'LAY', 'TFLAG', 'DATE-TIME']: continue
                        if dname in ['ROW', 'COL', 'SRC', 'NSRC']:
                            if ds.dimensions[dname].size > 1:
                                src_dim_idx = i
                                break
                if src_dim_idx == -1: src_dim_idx = var.ndim - 1
                  
                # Slice Construction
                slices = [slice(None)] * var.ndim
                l_axis = var.dimensions.index('LAY') if 'LAY' in var.dimensions else -1
                if l_axis >= 0:
                        if layer_idx is not None:
                            slices[l_axis] = layer_idx
                        elif layer_op == 'select':
                            slices[l_axis] = 0
                  
                # Read ONLY needed columns
                slices[src_dim_idx] = all_needed_srcs
                  
                # data => (T, [LAY?], N_Read_Srcs)
                data = var[tuple(slices)]
                  
                # Handle Layer Reduction (if aggregates)
                data_dims_map = [] 
                for i, sl in enumerate(slices):
                        if isinstance(sl, slice): data_dims_map.append(i)
                        else: 
                            if isinstance(sl, (list, np.ndarray, tuple)): data_dims_map.append(i)

                if l_axis >= 0 and layer_idx is None and layer_op != 'select':
                    try:
                        ax_now = data_dims_map.index(l_axis)
                        if layer_op == 'mean': data = np.mean(data, axis=ax_now)
                        elif layer_op == 'max': data = np.max(data, axis=ax_now)
                        elif layer_op == 'min': data = np.min(data, axis=ax_now)
                        else: data = np.sum(data, axis=ax_now)
                    except ValueError: pass
                  
                # Flatten to (T, N_Read_Srcs)
                if data.ndim != 2:
                        try: data = data.reshape(data.shape[0], -1)
                        except: pass
                  
                # 4. Vectorized Summation (Scatter Add)
                  
                # Use float32 for result to save memory (cuts 5GB -> 2.5GB for 150k cells)
                n_times = data.shape[0]
                n_cells = len(row_indices)
                  
                # Check total size before alloc
                total_elements = n_times * n_cells
                if total_elements > 5e8: # > 500 million floats (~2GB)
                        logging.warning(f"Large animation data request: {total_elements} elements. Allocating ~{total_elements*4/1024/1024:.1f} MB.")
                  
                try:
                        res_arr = np.zeros((n_times, n_cells), dtype=np.float32)
                except MemoryError:
                        logging.error("OOM: Could not allocate animation array. Try reducing time steps or pruned mode.")
                        return None
                  
                # Map Global_ID -> Local_Data_Col_Index
                global_to_local = { gid: i for i, gid in enumerate(all_needed_srcs) }
                  
                # Prepare index arrays for np.add.at
                # usage: np.add.at(target, indices, source_values)
                # Target: res_arr (T, N_Cells)
                # We operate on axis 1 (Cells).
                  
                # Unroll the mappings: [Local_Src_Idx] -> [Target_Cell_Idx]
                # Use numpy arrays for indices to ensure advanced indexing works
                # Pre-allocate lists
                count_ops = 0
                for gid, targets in src_to_out_map.items():
                        count_ops += len(targets)
                  
                op_src_indices = np.empty(count_ops, dtype=int)
                op_tgt_indices = np.empty(count_ops, dtype=int)
                  
                idx_ptr = 0
                for gid in all_needed_srcs:
                        if gid in src_to_out_map:
                            targets = src_to_out_map[gid]
                            lid = global_to_local[gid]
                            for tid in targets:
                                op_src_indices[idx_ptr] = lid
                                op_tgt_indices[idx_ptr] = tid
                                idx_ptr += 1
                  
                if count_ops > 0:
                        # Advanced Indexing Source: (T, N_Ops)
                        # Ensure data is float32 to match res_arr
                        if data.dtype != np.float32:
                            source_vals = data[:, op_src_indices].astype(np.float32)
                        else:
                            source_vals = data[:, op_src_indices]
                       
                        # Add to target
                        # res_arr is (T, N_Cells). indices tuple must address dims.
                        # (slice(None), op_tgt_indices) -> All times, Specific Cells.
                        # source_vals is (T, N_Ops). Matches shape of target selection?
                        # res_arr[:, op_tgt_indices] has shape (T, N_Ops). Yes.
                       
                        np.add.at(res_arr, (slice(None), op_tgt_indices), source_vals)
                       
                # Cleanup heavy arrays
                del data
                del source_vals
                del op_src_indices
                del op_tgt_indices
                       
                return {'times': tflag_vals, 'values': res_arr, 'units': units}
                  
        except Exception as e:
            logging.error(f"Inline Animation data failed (Vectorized): {e}")
            import traceback
            traceback.print_exc()
            return None
                  
        except Exception as e:
            logging.error(f"Inline Animation data failed: {e}")
            return None

    logging.debug(f"Entering get_ncf_animation_data for {pollutant}")
    with netCDF4.Dataset(ncf_path, 'r') as ds:
        logging.debug("Opened dataset")
        tflag_vals = []
        if 'TFLAG' in ds.variables:
            logging.debug("Reading TFLAG")
            tflag = ds.variables['TFLAG'][:]
            # TFLAG is usually (TSTEP, VAR, 2). 
            dates = tflag[:, 0, 0]
            times = tflag[:, 0, 1]
            tflag_vals = [f"{d}_{t:06d}" for d, t in zip(dates, times)]
            logging.debug(f"Found {len(tflag_vals)} time steps")
        else:
            # Fallback
            dim_t = ds.dimensions.get('TSTEP')
            if dim_t:
                tflag_vals = [f"T{i}" for i in range(dim_t.size)]
                
        if pollutant in ds.variables:
            var_obj = ds.variables[pollutant]
            units = getattr(var_obj, 'units', '').strip()
            
            # Read full data: (T, LAY, ROW, COL)
            # Optimize: if layer_idx is fixed, slice it early
            dims = var_obj.dimensions
            logging.debug(f"Extracting {pollutant} with dims {dims}")
            
            # Logic to slice correct dims
            # IOAPI: TSTEP, LAY, ROW, COL
            slices = [slice(None)] * len(dims)
            
            # Handle Layer
            if 'LAY' in dims:
                lay_dim_idx = dims.index('LAY')
                if layer_op == 'select':
                    # Select single layer
                    idx = layer_idx if layer_idx < var_obj.shape[lay_dim_idx] else 0
                    slices[lay_dim_idx] = idx
                # If sum/mean, we must read all layers first (or loop)
                # Reading all layers might be heavy. 
                # For animation, memory is concern. But let's try reading all if aggregating.
            
            try:
                logging.debug(f"Reading data slice for {pollutant}")
                data = var_obj[tuple(slices)] # Now (T, [LAY], ROW, COL) or (T, ROW, COL)
                logging.debug(f"Data read complete. Shape: {data.shape}")
                
                # If we still have LAY (because of aggregation)
                current_dims = [d for i, d in enumerate(dims) if isinstance(slices[i], slice)]
                if 'LAY' in current_dims:
                    ax = current_dims.index('LAY')
                    if layer_op == 'mean':
                        data = np.mean(data, axis=ax)
                    elif layer_op == 'sum':
                        data = np.sum(data, axis=ax)
                    else:
                        data = np.take(data, 0, axis=ax)
                
                # Now data is (T, ROW, COL)
                if data.ndim == 3:
                    # Select cells
                    r_idx = np.array(row_indices, dtype=int)
                    c_idx = np.array(col_indices, dtype=int)
                    
                    # Check bounds validity
                    if len(r_idx) > 0 and (r_idx.max() >= data.shape[1] or c_idx.max() >= data.shape[2]):
                        return None

                    # Vectorized selection: data[:, r, c]
                    logging.debug(f"Performing vectorized selection for {len(r_idx)} cells")
                    result = data[:, r_idx, c_idx] # (T, N)
                    return {'times': tflag_vals, 'values': result, 'units': units}
                
                return None
            except Exception as e:
                logging.error(f"Animation extract failed: {e}")
                return None
        return None


def read_inline_emissions_lazy(inln_path, stack_groups_path, pollutants, tstep_idx, layer_idx, tstep_op, layer_op, griddesc_path=None):
    valid_indices, v_rows, v_cols, gi = _get_inline_mapping(inln_path, stack_groups_path, griddesc_path, None)
    data_dict = {}
    with netCDF4.Dataset(inln_path) as nc_in:
        if pollutants is None:
            # Only pick float/int vars with appropriate dimensions (contain source dim?)
            # Heuristic: must not be TFLAG, and must have at least one dim that is NOT TSTEP/LAY/DATE-TIME?
            # Or just exclude char.
            pollutants = []
            for v in nc_in.variables:
                if v == 'TFLAG': continue
                if v not in nc_in.dimensions:
                        if np.issubdtype(nc_in.variables[v].dtype, np.number):
                            pollutants.append(v)
        
        for pol in pollutants:
            if pol not in nc_in.variables: continue
            var = nc_in.variables[pol]
            if not np.issubdtype(var.dtype, np.number): continue
            
            slices = [slice(None)] * var.ndim
            t_axis = -1
            if 'TSTEP' in var.dimensions: t_axis = var.dimensions.index('TSTEP')
            l_axis = -1
            if 'LAY' in var.dimensions: l_axis = var.dimensions.index('LAY')
            if t_axis >= 0 and tstep_idx is not None: slices[t_axis] = tstep_idx
            if l_axis >= 0 and layer_idx is not None: slices[l_axis] = layer_idx
            data = var[tuple(slices)]
            # Determine dimensions to reduce
            reductions = []
            
            def get_current_axis(dim_name):
                if dim_name not in var.dimensions: return None
                idx = var.dimensions.index(dim_name)
                # Adjust for initial integer slicing
                axis = 0
                for i in range(idx):
                        dn = var.dimensions[i]
                        # Was this dimension removed by integer slicing?
                        removed = False
                        if dn == 'TSTEP' and tstep_idx is not None: removed = True
                        if dn == 'LAY' and layer_idx is not None: removed = True
                        if not removed: axis += 1
                return axis

            if 'TSTEP' in var.dimensions and tstep_idx is None:
                ax = get_current_axis('TSTEP')
                reductions.append((ax, tstep_op))
            
            if 'LAY' in var.dimensions and layer_idx is None:
                ax = get_current_axis('LAY')
                reductions.append((ax, layer_op))
            
            # Sort by axis descending to avoid shifting
            reductions.sort(key=lambda x: x[0], reverse=True)
            
            for ax, op in reductions:
                logging.debug(f"[InlineLazy] Reducing Axis {ax} with Op '{op}'. Data Shape: {data.shape}")
                if op == 'mean': data = np.mean(data, axis=ax)
                elif op == 'max': data = np.max(data, axis=ax)
                elif op == 'min': data = np.min(data, axis=ax)
                else: data = np.sum(data, axis=ax)
                logging.debug(f"[InlineLazy] Result Shape: {data.shape}, Sum: {np.sum(data)}")
            
            data = data.flatten()
            grid = np.zeros((gi['nrows'], gi['ncols']), dtype=np.float32)
            if valid_indices.size > 0:
                if valid_indices.max() >= data.size:
                    logging.error(f"Lazy Read Mismatch: Data size {data.size} <= Max Index {valid_indices.max()}. Check STACK_GROUPS vs INLN file compatibility.")
                    # Prevent crash, but result will be wrong/partial
                    # subset = data[valid_indices[valid_indices < data.size]]
                    # Actually better to raise or return empty to signal issue
                    # But for now let it crash or filter?
                    pass
                subset = data[valid_indices]
                np.add.at(grid, (v_rows, v_cols), subset)
            data_dict[pol] = grid.flatten()
    return data_dict, gi['nrows'], gi['ncols']
