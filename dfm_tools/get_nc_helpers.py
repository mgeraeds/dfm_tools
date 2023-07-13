# -*- coding: utf-8 -*-
"""
dfm_tools are post-processing tools for Delft3D FM
Copyright (C) 2020 Deltares. All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  if not, see <http://www.gnu.org/licenses/>.

All names, logos, and references to "Deltares" are registered trademarks of
Stichting Deltares and remain full property of Stichting Deltares at all times.
All rights reserved.


INFORMATION
This script is part of dfm_tools: https://github.com/openearth/dfm_tools
Check the README.rst on github for other available functions
Check the tests folder on github for example scripts (this is the dfm_tools pytest testbank)
Check the pptx and example figures in (created by the testbank): N:/Deltabox/Bulletin/veenstra/info dfm_tools

Created on Fri Feb 14 12:43:19 2020

@author: veenstra

helper functions for functions in get_nc.py
"""

import xarray as xr
import xugrid as xu
import pandas as pd
import warnings


def get_ncvarproperties(data_xr):
    if not isinstance(data_xr,(xr.Dataset,xu.UgridDataset)):
        raise TypeError('data_xr should be of type xr.Dataset or xu.UgridDataset')
    
    nc_varkeys = data_xr.variables.mapping.keys()
    
    list_varattrs_pd = []
    for varkey in nc_varkeys:
        varattrs_pd = pd.DataFrame({varkey:data_xr.variables.mapping[varkey].attrs}).T
        varattrs_pd[['shape','dimensions']] = 2*[''] #set dtype as str (float will raise an error when putting tuple in there)
        varattrs_pd.at[varkey,'shape'] = data_xr[varkey].shape
        varattrs_pd.at[varkey,'dimensions'] = data_xr.variables[varkey].dims
        varattrs_pd.loc[varkey,'dtype'] = data_xr.variables[varkey].dtype
        list_varattrs_pd.append(varattrs_pd)
    
    vars_pd = pd.concat(list_varattrs_pd,axis=0)
    vars_pd[vars_pd.isnull()] = '' #avoid nan values
    
    data_xr.close()

    return vars_pd


def rename_waqvars(ds:(xr.Dataset,xu.UgridDataset)):
    """
    Rename all water quality variables in a dataset (like mesh2d_water_quality_output_24) to their long_name attribute (like mesh2d_DOscore)
    
    Parameters
    ----------
    ds : (xr.Dataset,xu.UgridDataset)
        DESCRIPTION.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    """
    #TODO: results also in variable "mesh2d_Water quality mass balance areas" (with spaces), report in FM issue (remove spaces from long_name attr)
    
    if hasattr(ds,'grid'): #append gridname (e.g. mesh2d) in case of mapfile
        varn_prepend = f'{ds.grid.name}_'
    else:
        varn_prepend = ''
    list_waqvars = [i for i in ds.data_vars if 'water_quality_' in i] #water_quality_output and water_quality_stat
    rename_dict = {waqvar:varn_prepend+ds[waqvar].attrs['long_name'] for waqvar in list_waqvars}
    
    if len(rename_dict) == 0: #early return to silence "FutureWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning."
        return ds
    
    #prevent renaming duplicate long_names
    rename_pd = pd.Series(rename_dict)
    if rename_pd.duplicated().sum():
        duplicated_pd = rename_pd.loc[rename_pd.duplicated(keep=False)]
        print(UserWarning(f'duplicate long_name attributes found with dfmt.rename_waqvars(), renaming only first variable:\n{duplicated_pd}'))
        rename_dict = rename_pd.loc[~rename_pd.duplicated()].to_dict()
    
    ds = ds.rename(rename_dict)
    return ds


def rename_fouvars(ds:(xr.Dataset,xu.UgridDataset), drop_tidal_times:bool = True):
    """
    Rename all fourier variables in a dataset (like mesh2d_fourier033_amp) to a unique name containing gridname/quantity/analysistype/tstart/tstop
    
    Parameters
    ----------
    ds : (xr.Dataset,xu.UgridDataset)
        DESCRIPTION.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    """
    
    file_freqs = 'https://raw.githubusercontent.com/Deltares/hatyan/main/hatyan/data/data_foreman_frequencies.txt' #TODO: fix hatyan dependency (MSQM and M1 were also added, but file is not used by hatyan, so might disappear one day)
    freqs_pd = pd.read_csv(file_freqs,names=['freq','dependents'],delim_whitespace=True,comment='#')
    freqs_pd['angfreq'] = freqs_pd['freq'] * 360 #deg/hr
    
    gridname = ds.grid.name
    list_fouvars = [i for i in ds.data_vars if '_fourier' in i] #water_quality_output and water_quality_stat
    
    rename_dict = {}
    for fouvar in list_fouvars:
        fouvar_attrs_lower = {k.lower():v for k,v in ds[fouvar].attrs.items()}
        fouvar_lowerattrs = ds[fouvar].assign_attrs(fouvar_attrs_lower) #to avoid case issues
        
        #quantity
        long_name = fouvar_lowerattrs.attrs['long_name']
        long_name_noprefix = long_name.split(': ')[1]
        quantity_long = long_name_noprefix.split(',')[0]
        quantity_dict = {'water level':'s1', #dict based on https://svn.oss.deltares.nl/repos/delft3d/trunk/src/engines_gpl/dflowfm/packages/dflowfm_kernel/src/dflowfm_kernel/prepost/fourier_analysis.f90
                         #'energy head':'s1', #TODO: duplicate namfun/dictvalue is not convenient
                         'wind speed':'ws',
                         'U-component of cell-centre velocity':'ux',
                         'V-component of cell-centre velocity':'uy',
                         'U-component velocity, column average':'uxa',
                         'V-component velocity, column average':'uya',
                         'velocity magnitude':'uc',
                         #'':'r1', #TODO: unclear which namfun/dictvalue corresponds (trim(namcon(gdfourier%fconno(ifou))))
                         'velocity':'u1',
                         'unit discharge':'qx',
                         'bed stress':'ta',
                         'freeboard':'fb',
                         'waterdepth_on_ground':'wdog',
                         'volume_on_ground':'vog',
                         'discharge through flow link':'q1',
                         'water level at flow link':'su1',
                         'temperature':'tem', #not clear from fourier_analysis.f90, ct in user manual C.13
                         'salt':'sal', #not clear from fourier_analysis.f90, cs in user manual C.13
                         }
        if quantity_long not in quantity_dict.keys():
            raise KeyError(f'quantity_dict does not yet contain quantity for: {quantity_long}')
        quantity = quantity_dict[quantity_long]
        
        #analysistype
        istidal = False
        if hasattr(fouvar_lowerattrs,'frequency_degrees_per_hour'):
            if fouvar_lowerattrs.attrs['frequency_degrees_per_hour'] > 0: #wl mean with numcyc=0 has frequency attribute (wl min with numcyc=0 does not)
                istidal = True #for tidal components with frequency >0
        
        if istidal: #for tidal analysistype
            tidepart = fouvar.split('_')[-1] # amp/phs
            freq = fouvar_lowerattrs.attrs['frequency_degrees_per_hour']
            compidx_closestfreq = (freqs_pd['angfreq'] - freq).abs().argmin()
            compname = freqs_pd.index[compidx_closestfreq] #M2/NU2
            analysistype = tidepart+compname
            warnings.warn(UserWarning('tidal components found in foufile, matching frequency with online list to get component names, which might go wrong. Also, be aware that v0 and knfac columns from fourier inputfile are not available in fourier output, so it is not clear whether to correct for these.'))
        else: #for all other quantities
            fouvar_splitted = fouvar.split('_')
            analysistype = ''.join(fouvar_splitted[2:]) #min/max/mean. min_depth/max_depth etc are converted to mindepth/maxdepth
            
        #tstart/tstop
        refdate = pd.Timestamp(str(fouvar_lowerattrs.attrs['reference_date_in_yyyymmdd']))
        if hasattr(fouvar_lowerattrs,'starttime_fourier_analysis_in_minutes_since_reference_date'):
            tstart_min = fouvar_lowerattrs.attrs['starttime_fourier_analysis_in_minutes_since_reference_date']
            tstop_min = fouvar_lowerattrs.attrs['stoptime_fourier_analysis_in_minutes_since_reference_date']
        elif hasattr(fouvar_lowerattrs,'starttime_min_max_analysis_in_minutes_since_reference_date'):
            tstart_min = fouvar_lowerattrs.attrs['starttime_min_max_analysis_in_minutes_since_reference_date']
            tstop_min = fouvar_lowerattrs.attrs['stoptime_min_max_analysis_in_minutes_since_reference_date']
        else:
            raise AttributeError(f'starttime/stoptime attribute not found in fouvar:\n{fouvar_lowerattrs.attrs}')
        tstart_str = (refdate + pd.Timedelta(minutes=tstart_min)).strftime('%Y%m%d%H%M%S')
        tstop_str = (refdate + pd.Timedelta(minutes=tstop_min)).strftime('%Y%m%d%H%M%S')
        
        rename_dict[fouvar] = f'{gridname}_{quantity}_{analysistype}_{tstart_str}_{tstop_str}'
        if istidal and drop_tidal_times:
            rename_dict[fouvar] = f'{gridname}_{quantity}_{analysistype}' #TODO: might cause conflicting variable names if one component is analysed for multiple periods or if component is not defined in frequency list. Add duplicate check like rename_waqvars() that provides some basic info for debugging.
    
    ds = ds.rename(rename_dict)
    return ds

def inverse_distance_weights(a, b):
    import numpy as np
    """
    Caculate inverse distance weights based and apply using xarray.

    Parameters
    ----------
    a : (xr.DataArray,xu.UgridDataArray)
        Coordinates of faces (dimn_faces, 2)
    
    b : (xr.DataArray,xu.UgridDataArray)
        Coordinates of edges per face (dimn_faces, dimn_face_nodes, 2)

    Returns
    -------
    weights : (xr.DataArray)
        Weights for each of the edge parameters (dimn_faces, dimn_face_nodes)

    """
    def weight_func(a, b):
        distance = np.linalg.norm(a[:, np.newaxis, :] - b, axis=-1)
        weights = distance / np.nansum(distance, axis=1)[:, np.newaxis] # remove this if you only want the distance 
        return weights
    
    weights = xr.apply_ufunc(weight_func, a, b,
                            input_core_dims=[list(a.dims), list(b.dims)],
                            output_core_dims=[[a.dims[0], next(iter(set(b.dims) - set(a.dims)))]])
    return weights 


def interpolate_edge2face(uds:(xr.Dataset,xu.UgridDataset), varn_onedges):
    """
    Interpolate variables on edges to the faces with inverse distance weighting. 

    Parameters
    ----------
    ds : (xr.Dataset,xu.UgridDataset)
        DESCRIPTION.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    """
    # > Imports
    import numpy as np
    import datetime as dt

    # > Determine main dimensions
    dimn_edges = uds.grid.edge_dimension
    dimn_faces = uds.grid.face_dimension
    dimn_maxfn = uds.grid.to_dataset().mesh2d.attrs['max_face_nodes_dimension']
    mesh2d_var = uds.grid.to_dataset().mesh2d

    # > Get face-edge-connectivity variable name + edge-node-connectivity
    varn_fnc = uds.grid.to_dataset().mesh2d.attrs['face_node_connectivity']
    varn_enc = uds.grid.to_dataset().mesh2d.attrs['edge_node_connectivity']
    varn_fec = 'face_edge_connectivity'

    # > Get face-edge-connectivity in xarray-format
    face_edges = xr.DataArray(data=uds.grid.face_edge_connectivity, dims=['mesh2d_nFaces', dimn_maxfn], coords=dict(face_edge_connectivity=([uds.grid.face_dimension, dimn_maxfn], uds.ugrid.grid.face_edge_connectivity)), attrs={'cf_role': 'face_edge_connectivity', 'start_index':0, '_FillValue':-1})

    # > Determine where the fill value is used
    if hasattr(face_edges,'_FillValue'):
        data_fnc_validbool = face_edges!=face_edges.attrs['_FillValue']
    else:
        data_fnc_validbool = None

    interpolation_tstart = dt.datetime.now()
    print(f'Starting interpolation from edges to faces of variable {varn_onedges}...')

    # > Determine face_edge_connectivity with nan-values where connectivity=-1 (_FillValue)
    face_coords = xr.DataArray(data=uds.grid.face_coordinates, dims=[dimn_faces, 'Two'], coords=dict(mesh2d_face_x=(dimn_faces, uds.ugrid.grid.face_coordinates[:,0]),mesh2d_face_y=(dimn_faces, uds.grid.face_coordinates[:,1]))) # uds.grid.face_coordinates
    edge_coords = xr.DataArray(data=uds.grid.edge_coordinates, dims=[dimn_edges, 'Two'], coords=dict(mesh2d_edge_x=(dimn_edges, uds.ugrid.grid.edge_coordinates[:,0]), mesh2d_edge_y=(dimn_edges, uds.grid.edge_coordinates[:,1]))) # uds.grid.edge_coordinates
    face_edge_x_coords = xr.where(face_edges!=face_edges.attrs['_FillValue'], edge_coords[:,0][face_edges], np.nan)
    face_edge_y_coords = xr.where(face_edges!=face_edges.attrs['_FillValue'], edge_coords[:,1][face_edges], np.nan) # change xr --> np if working with numpy arrays

    # > Stack the edge coordinates the right way (with nan-values)
    face_edge_coords = xr.combine_nested([face_edge_x_coords, face_edge_y_coords], concat_dim='Two').transpose('mesh2d_nFaces',  'mesh2d_nMax_face_nodes', 'Two')
    
    # > Determine weights based on inverse distances
    weights_fe = inverse_distance_weights(face_coords, face_edge_coords)

    ## >> Actual interpolation
    #------------------------------------
    # > Step 1: select all edge values corresponding to one face
    print('Working on step 1: selecting all edge values...')
    edgevar_tofaces_onint_step1 = uds[varn_onedges].isel({dimn_edges:face_edges})
    # > Step 2: replace non-existent edges with nan's
    print('Working on step 2: replacing non-existent edges with NaN values...')
    edgevar_tofaces_onint_step2 = edgevar_tofaces_onint_step1.where(data_fnc_validbool)
    print('Working on step 3: using inverse-distance weighting to calculate face values...')
    # > Step 3: use inverse-distance weighting to calculate face values
    edgevar_tofaces_onint = (edgevar_tofaces_onint_step2 * weights_fe).sum(axis=2)

    ## > Interpolate from interfaces to layers
    #------------------------------------------
    if hasattr(mesh2d_var,'interface_dimension'):
        print('Working on step 4: average from interfaces to layers (3D model)...')
        dimn_interface = mesh2d_var.attrs['interface_dimension']
        dimn_layer = mesh2d_var.attrs['layer_dimension']

        # > select all top interfaces and all bottom interfaces, sum, divide by two (same as average)
        edgevar_tofaces_topint = edgevar_tofaces_onint.isel({dimn_interface:slice(1,None)})
        edgevar_tofaces_botint = edgevar_tofaces_onint.isel({dimn_interface:slice(None,-1)})
        edgevar_tofaces = (edgevar_tofaces_topint + edgevar_tofaces_botint)/2

        # > rename int to lay dimension and re-assign variable attributes
        edgevar_tofaces = edgevar_tofaces.rename({dimn_interface:dimn_layer}).assign_attrs(edgevar_tofaces_onint_step1.attrs)
    else:
        edgevar_tofaces = edgevar_tofaces_onint

    ## > Add to dataset
    #------------------------------------
    uds[varn_onedges] = edgevar_tofaces
    print(f'Interpolation from edges to faces finished in {dt.datetime.now() - interpolation_tstart}')

    return uds
