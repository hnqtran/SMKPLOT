import netCDF4 as nc
import numpy as np
import datetime
import os

def generate_example_ncf(output_path):
    # 12LISTOS parameters from real file
    # SDATE = 2022001, STIME = 0, TSTEP = 10000, NROWS=25, NCOLS=25
    # We will use a smaller grid 10x10 for the example to keep size small
    nrows, ncols = 10, 10
    nsteps = 2
    
    if os.path.exists(output_path):
        os.remove(output_path)
        
    with nc.Dataset(output_path, 'w', format='NETCDF4_CLASSIC') as ds:
        # Dimensions
        ds.createDimension('TSTEP', None)
        ds.createDimension('DATE-TIME', 2)
        ds.createDimension('LAY', 1)
        ds.createDimension('VAR', 2)
        ds.createDimension('ROW', nrows)
        ds.createDimension('COL', ncols)
        
        # Variables
        tflag = ds.createVariable('TFLAG', 'i4', ('TSTEP', 'VAR', 'DATE-TIME'))
        tflag.units = '<YYYYDDD,HHMMSS>'
        tflag.long_name = 'TFLAG'
        tflag.var_desc = 'Timestep-valid flags'
        
        nox = ds.createVariable('NOX', 'f4', ('TSTEP', 'LAY', 'ROW', 'COL'))
        nox.units = 'moles/s'
        nox.long_name = 'NOX'
        nox.var_desc = 'Model species NOX'
        
        voc = ds.createVariable('VOC', 'f4', ('TSTEP', 'LAY', 'ROW', 'COL'))
        voc.units = 'moles/s'
        voc.long_name = 'VOC'
        voc.var_desc = 'Model species VOC'
        
        # Global Attributes from 12LISTOS
        ds.GDTYP = 2
        ds.P_ALP = 33.0
        ds.P_BET = 45.0
        ds.P_GAM = -97.0
        ds.XCENT = -97.0
        ds.YCENT = 40.0
        # XORIG/YORIG from 12LISTOS: 1812000., 240000.
        ds.XORIG = 1812000.0
        ds.YORIG = 240000.0
        ds.XCELL = 12000.0
        ds.YCELL = 12000.0
        ds.GDNAM = '12LISTOS'
        ds.SDATE = 2022001
        ds.STIME = 0
        ds.TSTEP = 10000
        ds.NVARS = 2
        ds.NCOLS = ncols
        ds.NROWS = nrows
        ds.NLAYS = 1
        ds.FTYPE = 1
        
        # Data
        # Step 0: values from 0 to 99
        # Step 1: values from 0 to 99 * 1.5
        # Total NOX sum for 2 steps = (4950 + 4950*1.5) = 12375. Avg = 6187.5
        # We'll set simpler values for QA/QC consistency
        data_nox = np.arange(nrows * ncols).reshape(nrows, ncols).astype(f'f4')
        data_voc = (data_nox + 1) * 0.8
        
        for t in range(nsteps):
            tflag[t, :, 0] = 2022001
            tflag[t, :, 1] = t * 10000
            
            nox[t, 0, :, :] = data_nox * (1.0 + t * 0.5)
            voc[t, 0, :, :] = data_voc * (1.0 - t * 0.2)
            
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_example_ncf('/proj/ie/proj/SMOKE/htran/Emission_Modeling_Platform/utils/smkplot/example_inputs/example_gridded.nc')
