# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:25:27 2023

@author: veenstra
"""

# interpolation of bathymetry to gtsm grid.
# before may 2023, this was done with uds.ugrid.sel(xslice), but "around the back" cells were selected in for instance slice(-60,-65), the face was computed in between these ranges. This caused the nodes to range from -180 to 150, which resulted in active interpolation of almost the entire gebco grid/dataset.
# now we are selecting the node_x coordinates between slices, the resulting list of xy coordinates is used for the gebco interpolation, effectively requiring only a lon-slice of the gebco dataset at the time.
#TODO: not all values are filled

import xarray as xr
import xugrid as xu
import datetime as dt

ds_gebco = xr.open_dataset('p:\\metocean-data\\open\\GEBCO\\2022\\GEBCO_2022.nc')
#extend gebco to lon=-180 and lat=90 to make interpolation on these nodes possible
ds_gebco = ds_gebco.reset_index(['lat','lon'])
ds_gebco['lon'].values[0] = -180 #replace -179.99791667 with -180
ds_gebco['lat'].values[-1] = 90 #replace 89.99791667 with 90
ds_gebco = ds_gebco.set_index({'lat':'lat','lon':'lon'})

file_net = r'p:\1230882-emodnet_hrsm\global_tide_surge_model\trunk\gtsm4.1\step11_global_1p25eu_withcellinfo_net.nc'
uds = xu.open_dataset(file_net)
nnodes = uds.dims[uds.grid.node_dimension]

stepsize = 5
print(f'interpolating GEBCO to {nnodes} nodes in 360/{stepsize}={360/stepsize} steps:')
dtstart = dt.datetime.now()
for i in range(-180, 180, stepsize):
    xslice = slice(i,i+stepsize)
    
    #def interp_gebco_gtsm(uds,xslice,ds_gebco):
    bool_nodeinslice = (uds.grid.node_coordinates[:,0] >= xslice.start) & (uds.grid.node_coordinates[:,0] < xslice.stop)
    print(xslice, ':', bool_nodeinslice.sum(), 'nodes')
    
    x_sel, y_sel = uds.grid.node_coordinates[bool_nodeinslice].T
    x_sel_ds = xr.DataArray(x_sel,dims=(uds.grid.node_dimension))
    y_sel_ds = xr.DataArray(y_sel,dims=(uds.grid.node_dimension))
    z_sel = ds_gebco.interp(lon=x_sel_ds, lat=y_sel_ds).reset_coords(['lat','lon']) #interpolates lon/lat gebcodata to mesh2d_nNodes dimension #TODO: if these come from xu_grid_uds (without ojb), the mesh2d_node_z var has no ugrid accessor since the dims are lat/lon instead of mesh2d_nNodes
    uds['NetNode_z'][bool_nodeinslice] = z_sel.elevation
    
    # if i==-170:
    #     breakit

print('plot data grid')
print(uds['NetNode_z'].isnull().sum()) #TODO: this must be 0, but is 1733, which is exactly the amount of left nodes ((uds.grid.node_coordinates[:,0] == -180).sum())
#uds.NetNode_z.ugrid.plot()
print(f'{(dt.datetime.now()-dtstart).total_seconds():.2f} sec')





