# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:04:56 2023

@author: veenstra
"""

import os
import dfm_tools as dfmt
import xarray as xr
import numpy as np

dir_testinput = r'c:\DATA\dfm_tools_testdata'

file_nc = os.path.join(dir_testinput,'DFM_curvedbend_3D','cb_3d_map.nc') #sigmalayer
file_nc = os.path.join(dir_testinput,'DFM_grevelingen_3D','Grevelingen-FM_0*_map.nc') #zlayer
# file_nc = r'p:\dflowfm\maintenance\JIRA\05000-05999\05477\c103_ws_3d_fourier\DFM_OUTPUT_westerscheldt01_0subst\westerscheldt01_0subst_map.nc' #zsigma model without fullgrid output but with new ocean_sigma_z_coordinate variable
# file_nc = r'p:\archivedprojects\11206813-006-kpp2021_rmm-2d\C_Work\31_RMM_FMmodel\computations\model_setup\run_207\results\RMM_dflowfm_0*_map.nc' #2D model
# file_nc = r'p:\1204257-dcsmzuno\2006-2012\3D-DCSM-FM\A18b_ntsu1\DFM_OUTPUT_DCSM-FM_0_5nm\DCSM-FM_0_5nm_0*_map.nc' #fullgrid
# file_nc = r'p:\archivedprojects\11203379-005-mwra-updated-bem\03_model\02_final\A72_ntsu0_kzlb2\DFM_OUTPUT_MB_02\MB_02_0*_map.nc'

uds = dfmt.open_partitioned_dataset(file_nc.replace('0*','0000')) #.isel(time=0)

uds_edges = dfmt.Dataset_varswithdim(uds, uds.grid.edge_dimension)

#TODO: can also be done for all edge variables, re-add to original dataset as *onfaces variables?
if 'mesh2d_vicwwu' in uds.data_vars:
    varn_onedges = 'mesh2d_vicwwu'
elif 'mesh2d_czu' in uds.data_vars:
    varn_onedges = 'mesh2d_czu'
else:
    varn_onedges = 'mesh2d_edge_type' #if all else fails, interpolate this one

mesh2d_var = uds.grid.to_dataset().mesh2d

print('construct indexer')
varn_fnc = mesh2d_var.attrs['face_node_connectivity']
dimn_maxfn = mesh2d_var.attrs['max_face_nodes_dimension']
dimn_edges = uds.grid.edge_dimension

# > Get face-edge-connectivity variable name + edge-node-connectivity
varn_fnc = uds.grid.to_dataset().mesh2d.attrs['face_node_connectivity']
varn_enc = uds.grid.to_dataset().mesh2d.attrs['edge_node_connectivity']
varn_fec = 'face_edge_connectivity'

# > Get face-edge-connectivity in xarray-format
face_edges = xr.DataArray(data=uds.grid.face_edge_connectivity, dims=['mesh2d_nFaces', dimn_maxfn], coords=dict(
    face_edge_connectivity=([uds.grid.face_dimension, dimn_maxfn], uds.ugrid.grid.face_edge_connectivity)),
                          attrs={'cf_role': 'face_edge_connectivity', 'start_index': 0, '_FillValue': -1})

data_fnc = uds.grid.to_dataset()[varn_fnc]

if hasattr(data_fnc,'_FillValue'):
    data_fnc_validbool = data_fnc!=data_fnc.attrs['_FillValue']
else:
    data_fnc_validbool = None

if hasattr(data_fnc,'start_index'):
    if data_fnc.attrs['start_index'] != 0:
        data_fnc = data_fnc - data_fnc.attrs['start_index'] #TODO: this drops attrs, re-add corrected attrs

interpolation_tstart = dt.datetime.now()
print(f'Starting interpolation from edges to faces of variable {varn_onedges}...')

# > Determine face_edge_connectivity with nan-values where connectivity=-1 (_FillValue)
face_coords = xr.DataArray(data=uds.grid.face_coordinates, dims=[uds.grid.face_dimension, 'Two'], coords=dict(mesh2d_face_x=(uds.grid.face_dimension, uds.ugrid.grid.face_coordinates[:,0]),mesh2d_face_y=(uds.grid.face_dimension, uds.grid.face_coordinates[:,1]))) # uds.grid.face_coordinates
edge_coords = xr.DataArray(data=uds.grid.edge_coordinates, dims=[uds.grid.edge_dimension, 'Two'], coords=dict(mesh2d_edge_x=(uds.grid.edge_dimension, uds.ugrid.grid.edge_coordinates[:,0]), mesh2d_edge_y=(uds.grid.edge_dimension, uds.grid.edge_coordinates[:,1]))) # uds.grid.edge_coordinates
face_edge_x_coords = xr.where(face_edges!=face_edges.attrs['_FillValue'], edge_coords[:,0][face_edges], np.nan)
face_edge_y_coords = xr.where(face_edges!=face_edges.attrs['_FillValue'], edge_coords[:,1][face_edges], np.nan) # change xr --> np if working with numpy arrays

# > Stack the edge coordinates the right way (with nan-values)
face_edge_coords = xr.combine_nested([face_edge_x_coords, face_edge_y_coords], concat_dim='Two').transpose('mesh2d_nFaces',  'mesh2d_nMax_face_nodes', 'Two')

# > Determine weights based on inverse distances  (direct)
def xarray_distance_weights(a, b):
    def weight_func(a, b):
        distance = np.linalg.norm(a[:, np.newaxis, :] - b, axis=-1)
        weights = distance / np.nansum(distance, axis=1)[:, np.newaxis] # remove this if you only want the distance
        return weights
    return xr.apply_ufunc(weight_func, a, b,
    input_core_dims=[list(a.dims), list(b.dims)],
    output_core_dims=[[a.dims[0], next(iter(set(b.dims) - set(a.dims)))]]
    )

weights_fe = xarray_distance_weights(face_coords, face_edge_coords)

#TODO: interpolation is slow for many timesteps, so maybe use .sel() on time dimension first
print('interpolation with indexer: step 1 (for each face, select all corresponding edge values)')
edgevar_tofaces_onint_step1 = uds[varn_onedges].isel({dimn_edges:data_fnc}) #TODO: fails for cb_3d_map.nc, westernscheldt
print('interpolation with indexer: step 2 (replace nonexistent edges with nan)')
edgevar_tofaces_onint_step2 = edgevar_tofaces_onint_step1.where(data_fnc_validbool) #replace all values for fillvalue edges (-1) with nan
print('interpolation with indexer: step 3 (average edge values per face)')
edgevar_tofaces_onint = edgevar_tofaces_onint_step2.mean(dim=dimn_maxfn)
print('interpolation with indexer: done')

if hasattr(mesh2d_var,'interface_dimension'):
    print('average from interfaces to layers (so in z-direction) in case of a 3D model')
    dimn_interface = mesh2d_var.attrs['interface_dimension']
    dimn_layer = mesh2d_var.attrs['layer_dimension']
    #select all top interfaces and all bottom interfaces, sum, divide by two (same as average)
    edgevar_tofaces_topint = edgevar_tofaces_onint.isel({dimn_interface:slice(1,None)})
    edgevar_tofaces_botint = edgevar_tofaces_onint.isel({dimn_interface:slice(None,-1)})
    edgevar_tofaces = (edgevar_tofaces_topint + edgevar_tofaces_botint)/2
    #rename int to lay dimension and re-assign variable attributes
    edgevar_tofaces = edgevar_tofaces.rename({dimn_interface:dimn_layer}).assign_attrs(edgevar_tofaces_onint_step1.attrs)
else:
    edgevar_tofaces = edgevar_tofaces_onint


# > Add to dataset
uds[varn_onedges] = edgevar_tofaces

#TODO: add inverse distance weighing, below example is from faces to edges, so make other way round
"""
edge_coords = grid.edge_coordinates
edge_faces = grid.edge_face_connectivity
boundary = (edge_faces[:, 1] == -1)
edge_faces[boundary, 1] = edge_faces[boundary, 0]
face_coords = grid.face_coordinates[edge_faces]
distance = np.linalg.norm(edge_coords[:, np.newaxis, :] - face_coords, axis=-1)
weights = distance / distance.sum(axis=1)[:, np.newaxis]
values = (face_values[edge_faces] * weights).sum(axis=1)
"""