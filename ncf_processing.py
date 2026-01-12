
import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import netCDF4
import pyproj
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Attempt relative import of sibling module data_processing
try:
    from .data_processing import USE_SPHERICAL_EARTH, _generate_grid_chunk, _file_signature
except ImportError:
    # If running as script or path issues, try direct import
    import sys
    sys.path.append(os.path.dirname(__file__))
    from data_processing import USE_SPHERICAL_EARTH, _generate_grid_chunk, _file_signature


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
    # Current environment has Shapely 2.0.7
    try:
        from shapely import polygons
        # Construct coordinate array for all cells: (nrows, ncols, 5, 2)
        # Each cell has 5 points: BL, BR, TR, TL, BL
        
        # Indices in (nrows+1, ncols+1) array:
        # BL: (r, c)
        # BR: (r, c+1)
        # TR: (r+1, c+1)
        # TL: (r+1, c)
        
        # We need to construct vectors for each corner
        # Using slicing:
        # lons[0:-1, 0:-1] is BL
        # lons[0:-1, 1:]   is BR
        # lons[1:,   1:]   is TR
        # lons[1:,   0:-1] is TL
        
        # Note: Depending on y-origin, row 0 might be bottom. 
        # y_edges starts at yorig. So row index corresponds to Y increasing.
        
        bl_x = lons[:-1, :-1].flatten()
        bl_y = lats[:-1, :-1].flatten()
        
        br_x = lons[:-1, 1:].flatten()
        br_y = lats[:-1, 1:].flatten()
        
        tr_x = lons[1:, 1:].flatten()
        tr_y = lats[1:, 1:].flatten()
        
        tl_x = lons[1:, :-1].flatten()
        tl_y = lats[1:, :-1].flatten()
        
        # Stack into (N, 5, 2)
        # shape (N,)
        zeros = np.zeros_like(bl_x) # placeholder? no, we stack
        
        # Coords shape: (N_cells, 5, 2)
        # axis 0: cell index
        # axis 1: vertex index (0..4)
        # axis 2: x/y (0..1)
        
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
        
    except (ImportError, AttributeError):
        # Fallback if Shapely < 2.0 (unlikely given pip list)
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

    return gdf


def read_ncf_emissions(
    ncf_path: str,
    pollutants: list = None,
    layer_idx: int = 0,
    tstep_idx: int = None,
    layer_op: str = 'select',
    tstep_op: str = 'sum'
) -> pd.DataFrame:
    """
    Read emissions from NetCDF.
    layer_idx: 0-based index of layer to read. If None, uses layer_op.
    tstep_idx: 0-based index of time step to read. If None, uses tstep_op.
    layer_op: 'select' (default), 'sum', or 'mean'.
    tstep_op: 'select' (default if idx provided), 'sum' (default if idx None), or 'mean'.
    """
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
    
    # Optimize DataFrame creation
    flat_rows = row_idx.flatten()
    flat_cols = col_idx.flatten()
    
    # Fast GRID_RC string creation using list comprehension is slow.
    # Use numpy character operations? Or just pandas + map.
    # Fast: f-string list comp is actually usually faster than pandas apply.
    # But for 1M rows, numpy is better.
    # r_s = flat_rows.astype(str) -- slow in Pandas
    # Let's stick to simple DF creation but maybe optimize GRID_RC if needed.
    # Actually, constructing GRID_RC on the fly during merge might be better?
    # But the merge key is GRID_RC.
    
    df = pd.DataFrame({
        'ROW': flat_rows,
        'COL': flat_cols,
    })
    
    # Vectorized string creation (approx 10x faster than apply)
    try:
        df['GRID_RC'] = df['ROW'].astype(str) + '_' + df['COL'].astype(str)
    except Exception:
        df['GRID_RC'] = [f"{r}_{c}" for r, c in zip(flat_rows, flat_cols)]
    
    for pol, values in data_dict.items():
        df[pol] = values
        
    df.attrs['variable_metadata'] = var_metadata
                 
    return df

def get_ncf_timeseries(
    ncf_path: str,
    pollutant: str,
    row_indices: list,
    col_indices: list,
    layer_idx: int = 0,
    layer_op: str = 'select',
    op: str = 'sum'
):
    """
    Extract time series for specific grid cells.
    row_indices/col_indices: 0-based index pairs.
    op: 'sum' or 'mean' (spatial aggregation).
    Returns dict with 'times', 'values', 'units'.
    """
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
    layer_op: str = 'select'
):
    """
    Extract data for animation: Full time series for each specified cell.
    Returns: {
        'times': [str], 
        'values': np.array shape (n_times, n_cells),
        'units': str
    }
    """
    with netCDF4.Dataset(ncf_path, 'r') as ds:
        tflag_vals = []
        if 'TFLAG' in ds.variables:
            tflag = ds.variables['TFLAG'][:]
            # TFLAG is usually (TSTEP, VAR, 2). 
            dates = tflag[:, 0, 0]
            times = tflag[:, 0, 1]
            tflag_vals = [f"{d}_{t:06d}" for d, t in zip(dates, times)]
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
                data = var_obj[tuple(slices)] # Now (T, [LAY], ROW, COL) or (T, ROW, COL)
                
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
                     # Vectorized selection: data[:, r, c]
                     result = data[:, r_idx, c_idx] # (T, N)
                     return {'times': tflag_vals, 'values': result, 'units': units}
                
                return None
            except Exception as e:
                logging.error(f"Animation extract failed: {e}")
                return None
        return None

