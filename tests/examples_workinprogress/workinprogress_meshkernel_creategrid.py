# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:54:39 2022

@author: veenstra

"""

import meshkernel
import xarray as xr
import matplotlib.pyplot as plt
plt.close('all')
import numpy as np
import contextily as ctx
import dfm_tools as dfmt

#TODO: maybe use make_basegrid and refine_basegrid functions from dfmt.meshkernel_helpers

#general settings
lon_min,lon_max = -6,2
lat_min,lat_max = 48.5,51.2
lon_res,lat_res = 0.2,0.2
num_x = int(np.ceil((lon_max-lon_min)/lon_res))
num_y = int(np.ceil((lat_max-lat_min)/lat_res))

figsize = (10,4)
crs = 'EPSG:4326'


"""
Make a regular (potentially rotated) rectilinear grid. First generate a curvilinear grid than convert the curvilinear grid into unstructured grid. The steps are the following:
- curvilinear_make_uniform, see the following notebook: https://github.com/Deltares/MeshKernelPy/blob/AddCurvilinearGridSupport/docs/examples/04_curvilineargrid_basics.ipynb
- curvilinear_convert_to_mesh2d: https://github.com/Deltares/MeshKernelPy/blob/118cb4953c4e95d5b18ed283bb37f391134b2bb2/meshkernel/meshkernel.py#L1399 

"""

# Create an instance of MakeGridParameters and set the values
make_grid_parameters = meshkernel.MakeGridParameters(num_columns=num_x,
                                                     num_rows=num_y,
                                                     angle=0.0, #TODO: does non-zero result in an orthogonal spherical grid?
                                                     origin_x=lon_min,
                                                     origin_y=lat_min,
                                                     block_size_x=lon_res,
                                                     block_size_y=lat_res)

grid_in_pol = False
# A polygon must to be provided. If empty it will not be used. If a polygon is provided it will be used in the generation of the curvilinear grid. The polygon must be closed
if grid_in_pol: #can be used instead of origin_x/origin_y and num_x/num_y
    pol_x = np.array([-6,-4,0,-6], dtype=np.double)
    pol_y = np.array([48,51,49.5,48], dtype=np.double)
else:
    pol_x = np.empty(0, dtype=np.double)
    pol_y = np.empty(0, dtype=np.double)
geometry_list = meshkernel.GeometryList(pol_x, pol_y)

mk = meshkernel.MeshKernel() #TODO: is_geographic=True has to be used but it fails: https://github.com/Deltares/MeshKernelPy/issues/39
mk.curvilinear_make_uniform(make_grid_parameters, geometry_list) #TODO: make geometry_list argument optional: https://github.com/Deltares/MeshKernelPy/issues/30
mk.curvilinear_convert_to_mesh2d() #convert to ugrid/mesh2d
mesh2d_basegrid = mk.mesh2d_get() #in case of curvi grid: mk.curvilinear_convert_to_mesh2d()
fig, ax = plt.subplots(figsize=figsize)
mesh2d_basegrid.plot_edges(ax,linewidth=1.2)
ctx.add_basemap(ax=ax, crs=crs, attribution=False)



"""
Mesh refinement in MeshKernelPy with bathymetry samples and plot result
"""
#select and plot bathy
file_nc_bathy = r'p:\metocean-data\open\GEBCO\2021\GEBCO_2021.nc'
data_bathy = xr.open_dataset(file_nc_bathy)
data_bathy_sel = data_bathy.sel(lon=slice(lon_min-1,lon_max+1),lat=slice(lat_min-1,lat_max+1))

fig, ax = plt.subplots(figsize=figsize)
data_bathy_sel.elevation.plot(ax=ax, center=False)
ctx.add_basemap(ax=ax, crs=crs, attribution=False)

#convert bathy data to geomlist
samp_x,samp_y = np.meshgrid(data_bathy_sel.lon.to_numpy(),data_bathy_sel.lat.to_numpy())
samp_z = data_bathy_sel.elevation.to_numpy().astype(float) #TODO: without .astype(float), meshkernelpy generates "TypeError: incompatible types, c_short_Array_27120 instance instead of LP_c_double instance": https://github.com/Deltares/MeshKernelPy/issues/31
samp_x = samp_x.ravel()
samp_y = samp_y.ravel()
samp_z = samp_z.ravel()
geomlist = meshkernel.GeometryList(x_coordinates=samp_x, y_coordinates=samp_y, values=samp_z) #TODO: does not check if lenghts of input array is equal (samp_z[1:]) https://github.com/Deltares/MeshKernelPy/issues/32

#refinement
mesh_refinement_parameters = meshkernel.MeshRefinementParameters(refine_intersected=False, #TODO: provide defaults for several arguments, so less arguments are required: https://github.com/Deltares/MeshKernelPy/issues/40
                                                                 use_mass_center_when_refining=False, #TODO: what does this do?
                                                                 min_face_size=0.01, #TODO: size in meters would be more convenient: https://github.com/Deltares/MeshKernelPy/issues/33 (maybe already works after is_geographic=True?)
                                                                 refinement_type=meshkernel.RefinementType(1), #Wavecourant/1,
                                                                 connect_hanging_nodes=True, #set to False to do multiple refinement steps (e.g. for multiple regions)
                                                                 account_for_samples_outside_face=False, #outsidecell argument for --refine?
                                                                 max_refinement_iterations=5,
                                                                 ) #TODO: missing the arguments dtmax (necessary?), hmin (min_face_size but then in meters instead of degrees), smoothiters (currently refinement is patchy along coastlines, goes good in dflowfm exec after additional implementation of HK), spherical 1/0 (necessary?)

mk.mesh2d_refine_based_on_samples(samples=geomlist,
                                   relative_search_radius=0.5, #TODO: bilin interp is preferred, but this is currently not supported (samples have to be ravelled): https://github.com/Deltares/MeshKernelPy/issues/34
                                   minimum_num_samples=3,
                                   mesh_refinement_params=mesh_refinement_parameters,
                                   )

mesh2d_refinedgrid = mk.mesh2d_get()
fig, ax = plt.subplots(figsize=figsize)
mesh2d_refinedgrid.plot_edges(ax,linewidth=1.2)
ctx.add_basemap(ax=ax, crs=crs, attribution=False)

#TODO: zoomed in plot to focus on patchy coastlines: https://github.com/Deltares/MeshKernelPy/issues/29
fig, ax = plt.subplots(figsize=figsize)
mesh2d_refinedgrid.plot_edges(ax,linewidth=1)
ax.set_xlim(-2.5,-0.5)
ax.set_ylim(49,50)
ctx.add_basemap(ax=ax, crs=crs, attribution=False)


"""
delete (landward) part of grid with polygon and plot result
"""

#line, = ax.plot([], [],'o-') # empty line
# #linebuilder = dfmt.LineBuilder(line) #TODO: this makes it possible to interactively click a line in the bedlevel figure. Use linebuilder.line_array as alternative line_array
# delete_pol = np.array([[ 1.91741935, 49.76580645],
#                         [ 0.20387097, 49.9       ],
#                         [-0.25032258, 48.71290323],
#                         [ 1.92774194, 48.59935484]])
file_ldb = r'p:\1230882-emodnet_hrsm\global_tide_surge_model\trunk\scripts_gtsm5\landboundary\GSHHS_intermediate_min1000km2.ldb'
dfmt.meshkernel_delete_withpol(mk, file_ldb=file_ldb, minpoints=1000) #TODO: update this function to work with GSHHS more efficiently: coastlines_gdb = dfmt.get_coastlines_gdb(bbox=(-10, 35, 10, 60))



mesh2d_noland = mk.mesh2d_get()
fig, ax = plt.subplots(figsize=figsize)
mesh2d_noland.plot_edges(ax,linewidth=1.2)
# xlim,ylim = ax.get_xlim(),ax.get_ylim() #get x/ylims before ldb plotting changes it
# for iP, pol_del in enumerate(pol_ldb_list):
#     ax.plot(pol_del['x'],pol_del['y'],'-r')
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
ctx.add_basemap(ax=ax, crs=crs, attribution=False)



"""
convert meshkernel grid to xugrid, interp bathymetry, plot and save to *_net.nc
"""

xu_grid_uds = dfmt.meshkernel_to_UgridDataset(mk, remove_noncontiguous=True) #TODO: put remove_noncontiguous in meshkernel?: https://github.com/Deltares/MeshKernelPy/issues/44
#TODO: add wgs84 variable with attrs

fig, ax = plt.subplots(figsize=figsize)
xu_grid_uds.grid.plot(ax=ax) #TODO: maybe make uds instead of ds (but then bathy interpolation goes wrong)
ctx.add_basemap(ax=ax, crs=crs, attribution=False)

#interp bathy
data_bathy_interp = data_bathy_sel.interp(lon=xu_grid_uds.obj.mesh2d_node_x, lat=xu_grid_uds.obj.mesh2d_node_y).reset_coords(['lat','lon']) #interpolates lon/lat gebcodata to mesh2d_nNodes dimension #TODO: if these come from xu_grid_uds, the mesh2d_node_z var has no ugrid accessor since the dims are lat/lon instead of mesh2d_nNodes
xu_grid_uds['mesh2d_node_z'] = data_bathy_interp.elevation.clip(max=10)

fig, ax = plt.subplots(figsize=figsize)
xu_grid_uds.mesh2d_node_z.ugrid.plot(ax=ax,center=False)
ctx.add_basemap(ax=ax, crs=crs, attribution=False)

#write xugrid grid to netcdf
xu_grid_uds.ugrid.to_netcdf('test_net.nc')

#TODO: update https://github.com/Deltares/dfm_tools/issues/217

#TODO: network is not orthogonal (when initializing with interacter), probably since the is_geographic keyword does not work yet >> there is a link missing, maybe due to ldb?

