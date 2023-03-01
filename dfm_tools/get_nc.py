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

Created on Fri Feb 14 12:45:11 2020

@author: veenstra
"""

import warnings
import numpy as np
import datetime as dt
import pandas as pd
import xugrid as xu
import xarray as xr
import matplotlib.pyplot as plt
from dfm_tools.xarray_helpers import get_vertical_dimensions


def get_ugrid_verts(data_xr_map): #TODO: remove this deprecated function
    """
    getting ugrid verts from xugrid mapfile.
    """
    raise DeprecationWarning('dfmt.get_ugrid_verts() is deprecated, use uds.grid.face_node_coordinates instead (https://github.com/Deltares/xugrid/issues/48)')
    
    # face_nos = data_xr_map.grid.face_node_connectivity
    
    # face_nnodecoords_x = data_xr_map.grid.node_x[face_nos]
    # face_nnodecoords_x[face_nos==-1] = np.nan
    # face_nnodecoords_y = data_xr_map.grid.node_y[face_nos]
    # face_nnodecoords_y[face_nos==-1] = np.nan
    
    # ugrid_all_verts = np.c_[face_nnodecoords_x[...,np.newaxis],face_nnodecoords_y[...,np.newaxis]]
    # return ugrid_all_verts


def calc_dist_pythagoras(x1,x2,y1,y2): # TODO: only used in dfm_tools.ugrid, remove?
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def calc_dist_haversine(lon1,lon2,lat1,lat2): # TODO: only used in dfm_tools.ugrid, remove?
    """
    calculates distance between lat/lon coordinates in meters
    https://community.esri.com/t5/coordinate-reference-systems-blog/distance-on-a-sphere-the-haversine-formula/ba-p/902128
    """
    # convert to radians
    lon1_rad = np.deg2rad(lon1)
    lon2_rad = np.deg2rad(lon2)
    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    
    # apply formulae
    a = np.sin((lat2_rad-lat1_rad)/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin((lon2_rad-lon1_rad)/2)**2
    c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )
    R = 6371000
    distance = R * c
    if np.isnan(distance).any():
        raise Exception('nan encountered in calc_dist_latlon distance, replaced by 0') #warnings.warn
        #distance[np.isnan(distance)] = 0
    return distance


def polygon_intersect(data_frommap_merged, line_array, calcdist_fromlatlon=None):
    #data_frommap_merged: xugrid dataset (contains ds and grid)
    #TODO: remove hardcoding
    """
    #TODO: maybe move to meshkernel/xugrid functionality?
    Cross section functionality is implemented in MeshKernel (C++) but still needs to be exposed in MeshKernelPy (can be done in dec2022). Here is the function with documentation: 
    https://github.com/Deltares/MeshKernel/blob/067f1493e7f972ba0cdb2a1f4deb48d1c74695d5/include/MeshKernelApi/MeshKernel.hpp#L356
    """
    
    import numpy as np
    #from matplotlib.path import Path
    import shapely #separate import, since sometimes this works, while import shapely.geometry fails
    from shapely.geometry import LineString, Polygon, MultiLineString, Point
    from dfm_tools.get_nc import calc_dist_pythagoras, calc_dist_haversine

    if calcdist_fromlatlon is None:
        #auto determine if cartesian/sperical
        if hasattr(data_frommap_merged.ugrid.obj,'projected_coordinate_system'):
            calcdist_fromlatlon = False
        elif hasattr(data_frommap_merged.ugrid.obj,'wgs84'):
            calcdist_fromlatlon = True
        else:
            raise Exception('To auto determine calcdist_fromlatlon, a variable "projected_coordinate_system" or "wgs84" is required, please provide calcdist_fromlatlon=True/False yourself.')

    dtstart_all = dt.datetime.now()

    #defining celinlinebox
    line_section = LineString(line_array)
    
    ugrid_all_verts = data_frommap_merged.grid.face_node_coordinates
    verts_xmax = np.nanmax(ugrid_all_verts[:,:,0].data,axis=1)
    verts_xmin = np.nanmin(ugrid_all_verts[:,:,0].data,axis=1)
    verts_ymax = np.nanmax(ugrid_all_verts[:,:,1].data,axis=1)
    verts_ymin = np.nanmin(ugrid_all_verts[:,:,1].data,axis=1)
    
    #TODO: replace this with xr.sel() once it works for xugrid (getting verts_inlinebox_nos is than still an issue)
    cellinlinebox_all_bool = (((np.min(line_array[:,0]) <= verts_xmax) &
                               (np.max(line_array[:,0]) >= verts_xmin)) &
                              ((np.min(line_array[:,1]) <= verts_ymax) & 
                               (np.max(line_array[:,1]) >= verts_ymin))
                              )
    #cellinlinebox_all_bool[:] = 1 #to force all cells to be taken into account
    
    intersect_coords = np.empty((0,4))
    intersect_gridnos = np.empty((0),dtype=int) #has to be numbers, since a boolean is differently ordered
    verts_inlinebox = ugrid_all_verts[cellinlinebox_all_bool,:,:]
    verts_inlinebox_nos = np.where(cellinlinebox_all_bool)[0]

    
    print(f'>> finding crossing flow links (can take a while, processing {cellinlinebox_all_bool.sum()} of {len(cellinlinebox_all_bool)} cells): ',end='')
    dtstart = dt.datetime.now()
    for iP, pol_data in enumerate(verts_inlinebox):
        pol_shp = Polygon(pol_data[~np.isnan(pol_data).all(axis=1)])
        intersect_result = pol_shp.intersection(line_section)
        if isinstance(intersect_result,shapely.geometry.multilinestring.MultiLineString): #in the rare case that a cell (pol_shp) is crossed by multiple parts of the line
            intersect_result_multi = intersect_result
        elif isinstance(intersect_result,shapely.geometry.linestring.LineString): #if one linepart trough cell (ex/including node), make multilinestring anyway
            if intersect_result.coords == []: #when the line does not cross this cell, intersect_results.coords is an empty linestring and this cell can be skipped (continue makes forloop continue with next in line without finishing the rest of the steps for this instance)
                continue
            elif len(intersect_result.coords.xy[0]) == 0: #for newer cartopy versions, when line does not cross this cell, intersect_result.coords.xy is (array('d'), array('d')), and both arrays in tuple have len 0.
                continue
            intersect_result_multi = MultiLineString([intersect_result])
        for iLL, intesect_result_one in enumerate(intersect_result_multi.geoms): #loop over multilinestrings, will mostly only contain one linestring. Will be two if the line crosses a cell more than once.
            intersection_line = intesect_result_one.coords
            intline_xyshape = np.array(intersection_line.xy).shape
            #print('len(intersection_line.xy): %s'%([intline_xyshape]))
            for numlinepart_incell in range(1,intline_xyshape[1]): #is mostly 1, but more if there is a linebreakpoint in this cell (then there are two or more lineparts)
                intersect_gridnos = np.append(intersect_gridnos,verts_inlinebox_nos[iP])
                #intersect_coords = np.concatenate([intersect_coords,np.array(intersection_line.xy)[np.newaxis,:,numlinepart_incell-1:numlinepart_incell+1]],axis=0)
                intersect_coords = np.concatenate([intersect_coords,np.array(intersection_line.xy).T[numlinepart_incell-1:numlinepart_incell+1].flatten()[np.newaxis]])
    
    if intersect_coords.shape[0] != len(intersect_gridnos):
        raise Exception('something went wrong, intersect_coords.shape[0] and len(intersect_gridnos) are not equal')
    
    intersect_pd = pd.DataFrame(intersect_coords,index=intersect_gridnos,columns=['x1','y1','x2','y2'])
    intersect_pd.index.name = 'gridnumber'
    print(f'{(dt.datetime.now()-dtstart).total_seconds():.2f} sec')
            
    
    #calculating distance for all crossed cells, from first point of line
    nlinecoords = line_array.shape[0]
    nlinedims = len(line_array.shape)
    ncrosscellparts = len(intersect_pd)
    if nlinecoords<2 or nlinedims != 2:
        raise Exception('ERROR: line_array should at least contain two xy points [[x,y],[x,y]]')
    
    #calculate distance between celledge-linepart crossing (is zero when line iL crosses cell)
    distperline_tostart = np.zeros((ncrosscellparts,nlinecoords-1))
    distperline_tostop = np.zeros((ncrosscellparts,nlinecoords-1))
    linepart_length = np.zeros((nlinecoords))
    for iL in range(nlinecoords-1):
        #calculate length of lineparts
        line_section_part = LineString(line_array[iL:iL+2,:])
        if calcdist_fromlatlon:
            linepart_length[iL+1] = calc_dist_haversine(line_array[iL,0],line_array[iL+1,0],line_array[iL,1],line_array[iL+1,1])
        else:
            linepart_length[iL+1] = line_section_part.length
    
        #get distance between all lineparts and point (later used to calculate distance from beginpoint of closest linepart)
        for iP in range(ncrosscellparts):
            distperline_tostart[iP,iL] = line_section_part.distance(Point(intersect_coords[:,0][iP],intersect_coords[:,1][iP]))
            distperline_tostop[iP,iL] = line_section_part.distance(Point(intersect_coords[:,2][iP],intersect_coords[:,3][iP]))
    linepart_lengthcum = np.cumsum(linepart_length)
    cross_points_closestlineid = np.argmin(np.maximum(distperline_tostart,distperline_tostop),axis=1)
    intersect_pd['closestlineid'] = cross_points_closestlineid
    
    
    if not calcdist_fromlatlon:
        crs_dist_starts = calc_dist_pythagoras(line_array[cross_points_closestlineid,0], intersect_coords[:,0], line_array[cross_points_closestlineid,1], intersect_coords[:,1]) + linepart_lengthcum[cross_points_closestlineid]
        crs_dist_stops = calc_dist_pythagoras(line_array[cross_points_closestlineid,0], intersect_coords[:,2], line_array[cross_points_closestlineid,1], intersect_coords[:,3]) + linepart_lengthcum[cross_points_closestlineid]
    else:
        crs_dist_starts = calc_dist_haversine(line_array[cross_points_closestlineid,0], intersect_coords[:,0], line_array[cross_points_closestlineid,1], intersect_coords[:,1]) + linepart_lengthcum[cross_points_closestlineid]
        crs_dist_stops = calc_dist_haversine(line_array[cross_points_closestlineid,0], intersect_coords[:,2], line_array[cross_points_closestlineid,1], intersect_coords[:,3]) + linepart_lengthcum[cross_points_closestlineid]
    intersect_pd['crs_dist_starts'] = crs_dist_starts
    intersect_pd['crs_dist_stops'] = crs_dist_stops
    intersect_pd['linepartlen'] = crs_dist_stops-crs_dist_starts
    intersect_pd = intersect_pd.sort_values('crs_dist_starts')
    
    print(f'>> polygon_intersect() total, found intersection trough {len(intersect_gridnos)} of {len(cellinlinebox_all_bool)} cells: {(dt.datetime.now()-dtstart_all).total_seconds():.2f} sec')
    return intersect_pd


def get_xzcoords_onintersection(data_frommap_merged, intersect_pd, timestep=None):
    #TODO: remove hardcoding of variable names
    if timestep is None: #TODO: maybe make time dependent grid?
        raise Exception('ERROR: argument timestep not provided, this is necessary to retrieve correct waterlevel or fullgrid output')
    
    dimn_layer, dimn_interfaces = get_vertical_dimensions(data_frommap_merged)
    if dimn_layer is not None:
        nlay = data_frommap_merged.dims[dimn_layer]
    else: #no layers, 2D model
        nlay = 1

    xu_facedim = data_frommap_merged.grid.face_dimension
    xu_edgedim = data_frommap_merged.grid.edge_dimension
    xu_nodedim = data_frommap_merged.grid.node_dimension
        
    #potentially construct fullgrid info (zcc/zw) #TODO: this ifloop is copied from get_mapdata_atdepth(), prevent this duplicate code
    if dimn_layer not in data_frommap_merged.dims: #2D model
        print('depth dimension not found, probably 2D model')
        pass
    elif 'mesh2d_flowelem_zw' in data_frommap_merged.variables: #fullgrid info already available, so continuing
        print('zw/zcc (fullgrid) values already present in Dataset')
        pass
    elif 'mesh2d_layer_sigma' in data_frommap_merged.variables: #reconstruct_zw_zcc_fromsigma and treat as zsigma/fullgrid mapfile from here
        print('sigma-layer model, computing zw/zcc (fullgrid) values and treat as fullgrid model from here')
        data_frommap_merged = reconstruct_zw_zcc_fromsigma(data_frommap_merged)
    elif 'mesh2d_layer_z' in data_frommap_merged.variables:        
        print('z-layer model, computing zw/zcc (fullgrid) values and treat as fullgrid model from here')
        data_frommap_merged = reconstruct_zw_zcc_fromz(data_frommap_merged)
    else:
        raise Exception('layers present, but unknown layertype, expected one of variables: mesh2d_flowelem_zw, mesh2d_layer_sigma, mesh2d_layer_z')
    
    intersect_gridnos = intersect_pd.index
    data_frommap_merged_sel = data_frommap_merged.ugrid.obj.drop_dims([xu_edgedim,xu_nodedim]).isel(time=timestep).isel({xu_facedim:intersect_gridnos}) #TODO: not possible to do isel with non-sorted gridnos on ugridDataset, but it is on xrDataset. Dropping edge/nodedims first for neatness, so there is not mismatch in face/node/edge after using .isel(faces)
    if dimn_layer not in data_frommap_merged_sel.dims:
        data_frommap_wl3_sel = data_frommap_merged_sel['mesh2d_s1'].to_numpy()
        data_frommap_bl_sel = data_frommap_merged_sel['mesh2d_flowelem_bl'].to_numpy()
        zvals_interface = np.linspace(data_frommap_bl_sel,data_frommap_wl3_sel,nlay+1)
    elif 'mesh2d_flowelem_zw' in data_frommap_merged_sel.variables:
        zvals_interface_filled = data_frommap_merged_sel['mesh2d_flowelem_zw'].bfill(dim=dimn_interfaces) #fill nan values (below bed) with equal values
        zvals_interface = zvals_interface_filled.to_numpy().T # transpose to make in line with 2D sigma dataset
    
    #convert to output for plot_netmapdata
    crs_dist_starts_matrix = np.repeat(intersect_pd['crs_dist_starts'].values[np.newaxis],nlay,axis=0)
    crs_dist_stops_matrix = np.repeat(intersect_pd['crs_dist_stops'].values[np.newaxis],nlay,axis=0)
    crs_verts_x_all = np.array([[crs_dist_starts_matrix.ravel(),crs_dist_stops_matrix.ravel(),crs_dist_stops_matrix.ravel(),crs_dist_starts_matrix.ravel()]]).T
    crs_verts_z_all = np.ma.array([zvals_interface[1:,:].ravel(),zvals_interface[1:,:].ravel(),zvals_interface[:-1,:].ravel(),zvals_interface[:-1,:].ravel()]).T[:,:,np.newaxis]
    crs_verts = np.ma.concatenate([crs_verts_x_all, crs_verts_z_all], axis=2)
    
    #define grid
    shape_crs_grid = crs_verts[:,:,0].shape
    shape_crs_flat = crs_verts[:,:,0].ravel().shape
    xr_crs_grid = xu.Ugrid2d(node_x=crs_verts[:,:,0].ravel(),
                             node_y=crs_verts[:,:,1].ravel(),
                             fill_value=-1,
                             face_node_connectivity=np.arange(shape_crs_flat[0]).reshape(shape_crs_grid),
                             )

    #define dataset
    crs_plotdata_clean = data_frommap_merged_sel
    if dimn_layer in data_frommap_merged_sel.dims:
        facedim_tempname = 'facedim_tempname' #temporary new name to avoid duplicate from-to dimension name in .stack()
        crs_plotdata_clean = crs_plotdata_clean.rename({xu_facedim:facedim_tempname})
        crs_plotdata_clean = crs_plotdata_clean.stack({xr_crs_grid.face_dimension:[dimn_layer,facedim_tempname]})
        #reset_index converts face-dimension from multiindex to flat
        crs_plotdata_clean = crs_plotdata_clean.reset_index([xr_crs_grid.face_dimension])
    
    #combine into xugrid
    xr_crs_ugrid = xu.UgridDataset(crs_plotdata_clean, grids=[xr_crs_grid])
    return xr_crs_ugrid


def polyline_mapslice(data_frommap_merged, line_array, timestep, calcdist_fromlatlon=None): #TODO: merge this into one function
    #intersect function, find crossed cell numbers (gridnos) and coordinates of intersection (2 per crossed cell)
    intersect_pd = polygon_intersect(data_frommap_merged, line_array, calcdist_fromlatlon=calcdist_fromlatlon)
    if len(intersect_pd) == 0:
        raise Exception('line_array does not cross mapdata') #TODO: move exception elsewhere?
    #derive vertices from cross section (distance from first point)
    xr_crs_ugrid = get_xzcoords_onintersection(data_frommap_merged, intersect_pd=intersect_pd, timestep=timestep)
    return xr_crs_ugrid


def reconstruct_zw_zcc_fromsigma(data_xr_map):
    """
    reconstruct full grid output (time/face-varying z-values) for sigma model, necessary for slicing sigmamodel on depth value
    """
    data_frommap_wl_sel = data_xr_map['mesh2d_s1']
    data_frommap_bl_sel = data_xr_map['mesh2d_flowelem_bl']
    
    zvals_cen_percentage = data_xr_map['mesh2d_layer_sigma']
    data_xr_map['mesh2d_flowelem_zcc'] = data_frommap_wl_sel+(data_frommap_wl_sel-data_frommap_bl_sel)*zvals_cen_percentage
    
    zvals_interface_percentage = data_xr_map['mesh2d_interface_sigma']
    data_xr_map['mesh2d_flowelem_zw'] = data_frommap_wl_sel+(data_frommap_wl_sel-data_frommap_bl_sel)*zvals_interface_percentage
    
    data_xr_map = data_xr_map.set_coords(['mesh2d_flowelem_zw','mesh2d_flowelem_zcc'])
    return data_xr_map


def reconstruct_zw_zcc_fromz(data_xr_map):
    """
    reconstruct full grid output (time/face-varying z-values) for zvalue model. Necessary when extracting values with zdepth w.r.t. waterlevel/bedlevel
    #TODO: gives spotty result for 0/0.1m w.r.t. bedlevel for Grevelingen zmodel
    #TODO: remove hardcoding of varnames
    """
    
    dimn_layer, dimn_interfaces = get_vertical_dimensions(data_xr_map)
    
    data_frommap_wl_sel = data_xr_map['mesh2d_s1']
    data_frommap_z0_sel = data_frommap_wl_sel*0
    data_frommap_bl_sel = data_xr_map['mesh2d_flowelem_bl']
    
    zvals_cen_zval = data_xr_map['mesh2d_layer_z'] #no clipping for zcenter values, since otherwise interp will fail
    data_xr_map['mesh2d_flowelem_zcc'] = (data_frommap_z0_sel+zvals_cen_zval)

    zvals_interface_zval = data_xr_map['mesh2d_interface_z'] #clipping for zinterface values, to make sure layer interfaces are also at water/bed level
    data_xr_map['mesh2d_flowelem_zw'] = (data_frommap_z0_sel+zvals_interface_zval).clip(min=data_frommap_bl_sel, max=data_frommap_wl_sel)
    bool_notoplayer_int = zvals_interface_zval<zvals_interface_zval.isel({dimn_interfaces:-1})
    bool_int_abovewl = zvals_interface_zval>data_frommap_wl_sel
    data_xr_map['mesh2d_flowelem_zw'] = data_xr_map['mesh2d_flowelem_zw'].where(bool_notoplayer_int | bool_int_abovewl, other=data_frommap_wl_sel) #zvalues of top layer_interfaces that are lower than wl are replaced by wl
    
    data_xr_map = data_xr_map.set_coords(['mesh2d_flowelem_zw','mesh2d_flowelem_zcc'])
    return data_xr_map


def get_Dataset_atdepths(data_xr, depths, reference='z0', zlayer_z0_selnearest=False):
    """
    data_xr_map:
        has to be Dataset (not a DataArray), otherwise mesh2d_flowelem_zw etc are not available (interface z values)
        in case of zsigma/sigma layers (or fullgrid), it is advisable to .sel()/.isel() the time dimension first, because that is less computationally heavy
    depths:
        int/float or list/array of int/float. Depths w.r.t. reference level. If reference=='waterlevel', depth>0 returns only nans. If reference=='bedlevel', depth<0 returns only nans. Depths are sorted and only uniques are kept.
    reference:
        compute depth w.r.t. z0/waterlevel/bed
        default: reference='z0'
    zlayer_z0_interp:
        Use xr.interp() to interpolate zlayer model to z-value. Only possible for reference='z' (not 'waterlevel' or 'bedlevel'). Only used if "mesh2d_layer_z" is present (zlayer model)
        This is faster but results in values interpolated between zcc (z cell centers), so it is different than slicing.
    
    #TODO: zmodel gets depth in figure title, because of .set_index() in open_partitioned_dataset(). Sigmamodel gets percentage/fraction in title. >> set_index was removed there, so check again.
    #TODO: check if attributes should be passed/altered
    #TODO: build in check whether layers/interfaces are coherent (len(int)=len(lay)+1), since one could also supply a uds.isel(layer=range(10)) where there are still 51 interfaces. (if not, raise error to remove layer selection)
    #TODO: also waterlevelvar in 3D model gets depth_fromref coordinate, would be nice to avoid.
    #TODO: clean up unneccesary variables (like pre-interp depth values and interface dim)
    """
    
    depth_varname = 'depth_fromref'
    
    try:
        dimn_layer, dimn_interfaces = get_vertical_dimensions(data_xr)
    except: #TODO: catch nicely
        dimn_layer = dimn_interfaces = None
    
    if dimn_layer is not None: #D-FlowFM mapfile
        gridname = data_xr.grid.name
        varname_zint = f'{gridname}_flowelem_zw'
        dimname_layc = dimn_layer
        dimname_layw = dimn_interfaces
        varname_wl = f'{gridname}_s1'
        varname_bl = f'{gridname}_flowelem_bl'
    elif 'laydim' in data_xr.dims: #D-FlowFM hisfile
        varname_zint = 'zcoordinate_w'
        dimname_layc = 'laydim'
        dimname_layw = 'laydimw'
        varname_wl = 'waterlevel'
        varname_bl = 'bedlevel'
        warnings.warn(UserWarning('get_Dataset_atdepths() is not tested for hisfiles yet, please check your results.'))
    else:
        print('WARNING: depth dimension not found, probably 2D model, returning input Dataset')
        return data_xr #early return
    
    if not isinstance(data_xr,(xr.Dataset,xu.UgridDataset)):
        raise Exception(f'data_xr_map should be of type xr.Dataset, but is {type(data_xr)}')
    
    #create depth xr.DataArray
    if isinstance(depths,(float,int)):
        #depths = depths #float/int
        depth_dims = ()
    else:
        depths = np.unique(depths) #array of unique+sorted floats/ints
        depth_dims = (depth_varname)
    depths_xr = xr.DataArray(depths,dims=depth_dims,attrs={'units':'m',
                                                           'reference':f'model_{reference}',
                                                           'positive':'up'}) #TODO: make more in line with CMEMS etc
    
    #simplified/faster method for zlayer icm z0 reference (mapfiles only) #TODO: maybe remove this part of the code to make it better maintainable.
    if 'mesh2d_layer_z' in data_xr.variables and zlayer_z0_selnearest and reference=='z0': # selects nearest z-center values (instead of slicing), should be faster #TODO: check if this is faster than fullgrid
        print('z-layer model, zlayer_z0_selnearest=True and reference=="z0" so using xr.sel(method="nearest")]')
        warnings.warn(DeprecationWarning('The get_Dataset_atdepths() keyword zlayer_z0_selnearest might be phased out.'))
        data_xr = data_xr.set_index({dimn_layer:'mesh2d_layer_z'})#.rename({'nmesh2d_layer':depth_varname}) #set depth as index on layers, to be able to interp to depths instead of layernumbers
        data_xr[depth_varname] = depths_xr
        data_xr_atdepths = data_xr.sel({dimn_layer:depths_xr},method='nearest')
        data_wl = data_xr[varname_wl]
        data_bl = data_xr[varname_bl]
        data_xr_atdepths = data_xr_atdepths.where((depths_xr>=data_bl) & (depths_xr<=data_wl)) #filter above wl and below bl values
        return data_xr_atdepths #early return
    
    #potentially construct fullgrid info (zcc/zw) #TODO: maybe move to separate function, like open_partitioned_dataset() (although bl/wl are needed anyway)
    if varname_zint in data_xr.variables: #fullgrid info already available, so continuing
        print(f'zw/zcc (fullgrid) values already present in Dataset in variable {varname_zint}')
        pass
    elif 'mesh2d_layer_sigma' in data_xr.variables: #reconstruct_zw_zcc_fromsigma and treat as zsigma/fullgrid mapfile from here
        print('sigma-layer model, computing zw/zcc (fullgrid) values and treat as fullgrid model from here')
        data_xr = reconstruct_zw_zcc_fromsigma(data_xr)
    elif 'mesh2d_layer_z' in data_xr.variables:
        print('z-layer model, computing zw/zcc (fullgrid) values and treat as fullgrid model from here')
        data_xr = reconstruct_zw_zcc_fromz(data_xr)
    else:
        raise Exception('layers present, but unknown layertype/var')
    
    #correct reference level
    if reference=='z0':
        zw_reference = data_xr[varname_zint]
    elif reference=='waterlevel':
        data_wl = data_xr[varname_wl]
        zw_reference = data_xr[varname_zint] - data_wl
    elif reference=='bedlevel':
        data_bl = data_xr[varname_bl]
        zw_reference = data_xr[varname_zint] - data_bl
    else:
        raise Exception(f'unknown reference "{reference}" (possible are z0, waterlevel and bedlevel')
    
    print('>> subsetting data on fixed depth in fullgrid z-data: ',end='')
    dtstart = dt.datetime.now()
        
    if 'time' in data_xr.dims: #TODO: suppress this warning for hisfiles since it does not make sense
        warnings.warn(UserWarning('get_Dataset_atdepths() can be very slow when supplying dataset with time dimension, you could supply ds.isel(time=timestep) instead'))
        
    #get layerno via z-interface value (zw), check which celltop-interfaces are above/on depth and which which cellbottom-interfaces are below/on depth
    bool_topinterface_abovedepth = zw_reference.isel({dimname_layw:slice(1,None)}) >= depths_xr
    bool_botinterface_belowdepth = zw_reference.isel({dimname_layw:slice(None,-1)}) <= depths_xr
    bool_topbotinterface_arounddepth = bool_topinterface_abovedepth & bool_botinterface_belowdepth #this bool also automatically excludes all values below bed and above wl
    bool_topbotinterface_arounddepth = bool_topbotinterface_arounddepth.rename({dimname_layw:dimname_layc}) #correct dimname for interfaces to centers
    data_xr_atdepths = data_xr.where(bool_topbotinterface_arounddepth).max(dim=dimname_layc,keep_attrs=True) #set all layers but one to nan, followed by an arbitrary reduce (max in this case)
    #TODO: suppress/solve warning for DCSM (does not happen when supplying data_xr_map[['mesh2d_sa1','mesh2d_s1','mesh2d_flowelem_bl','mesh2d_flowelem_zw']]): "C:\Users\veenstra\Anaconda3\envs\dfm_tools_env\lib\site-packages\dask\array\core.py:4806: PerformanceWarning: Increasing number of chunks by factor of 20" >> still happens with new xugrid method?
    #TODO: suppress warning (upon plotting/load/etc): "C:\Users\veenstra\Anaconda3\envs\dfm_tools_env\lib\site-packages\dask\array\reductions.py:640: RuntimeWarning: All-NaN slice encountered"
    
    #add depth as coordinate var
    data_xr_atdepths[depth_varname] = depths_xr
    data_xr_atdepths = data_xr_atdepths.set_coords([depth_varname])
    
    print(f'{(dt.datetime.now()-dtstart).total_seconds():.2f} sec')
    
    return data_xr_atdepths


def plot_background(ax=None, projection=None, google_style='satellite', resolution=1, features=None, nticks=6, latlon_format=False, gridlines=False, **kwargs):
    """
    this definition uses cartopy to plot a geoaxis and a satellite basemap and coastlines. A faster alternative for a basemap is contextily:
    import contextily as ctx
    fig, ax = plt.subplots(1,1)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs="EPSG:28992")
    More info at: https://contextily.readthedocs.io/en/latest/reference.html

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot, optional
        DESCRIPTION. The default is None.
    projection : integer, cartopy._crs.CRS or cartopy._epsg._EPSGProjection, optional
        DESCRIPTION. The default is None.
    google_style : Nonetype or string, optional
       The style of the Google Maps tiles. One of None, ‘street’, ‘satellite’, ‘terrain’, and ‘only_streets’. The default is 'satellite'.
    resolution : int, optional
        resolution for the Google Maps tiles. 1 works wel for global images, 12 works well for a scale of Grevelingen lake, using 12 on global scale will give you a server timeout. The default is 1.
    features : string, optional
        Features to plot, options: None, 'ocean', 'rivers', 'land', 'countries', 'countries_highres', 'coastlines', 'coastlines_highres'. The default is None.
    nticks : TYPE, optional
        DESCRIPTION. The default is 6.
    latlon_format : bool, optional
        DESCRIPTION. The default is False.
    gridlines : TYPE, optional
        DESCRIPTION. The default is False.
    **kwargs : TYPE
        additional arguments for ax.add_feature or ax.coastlines(). examples arguments and values are: alpha=0.5, facecolor='none', edgecolor='gray', linewidth=0.5, linestyle=':'

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.img_tiles as cimgt
    import cartopy.feature as cfeature
    import cartopy.mpl.ticker as cticker

    dummy = ccrs.epsg(28992) #to make cartopy realize it has a cartopy._epsg._EPSGProjection class (maybe gets fixed with cartopy updates, see unittest test_cartopy_epsg)
    if ax is None: #provide axis projection on initialisation, cannot be edited later on
        if projection is None:
            projection=ccrs.PlateCarree() #projection of cimgt.GoogleTiles, useful default
        elif isinstance(projection, (cartopy._epsg._EPSGProjection, cartopy.crs.CRS)): #checks if argument is an EPSG projection or CRS projection (like PlateCarree, Mercator etc). Note: it was cartopy._crs.CRS before instead of cartopy.crs.CRS
            pass
        elif type(projection) is int:
            projection = ccrs.epsg(projection)
        else:
            raise Exception('argument projection should be of type integer, cartopy._crs.CRS or cartopy._epsg._EPSGProjection')
        fig, ax = plt.subplots(subplot_kw={'projection': projection})
        #ax = plt.axes(projection=projection)
    elif type(ax) is cartopy.mpl.geoaxes.GeoAxesSubplot:
        if projection is not None:
            print('arguments ax and projection are both provided, the projection from the ax is used so the projection argument is ignored')
    else:
        raise Exception('argument ax should be of type cartopy.mpl.geoaxes.GeoAxesSubplot, leave argument empty or create correct instance with:\nimport cartopy.crs as ccrs\nfig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5), subplot_kw={"projection": ccrs.epsg(28992)})')



    if gridlines:
        ax.gridlines(draw_labels=True)
    elif nticks is not None: #only look at nticks if gridlines are not used
        extent = ax.get_extent()
        ax.set_xticks(np.linspace(extent[0],extent[1],nticks))
        ax.set_yticks(np.linspace(extent[2],extent[3],nticks))


    if google_style is not None:
        request = cimgt.GoogleTiles(style=google_style)
        ax.add_image(request,resolution)


    if features is not None:
        if type(features) is str:
            features = [features]
        elif type(features) is not list:
            raise Exception('argument features should be of type list of str')

        valid_featurelist = ['ocean','rivers','land','countries','countries_highres','coastlines','coastlines_highres']
        invalid_featurelist = [x for x in features if x not in valid_featurelist]
        if invalid_featurelist != []:
            raise Exception('invalid features %s requested, possible are: %s'%(invalid_featurelist, valid_featurelist))

        if 'ocean' in features:
            #feat = cfeature.NaturalEarthFeature(category='physical', name='ocean', facecolor=cfeature.COLORS['water'], scale='10m', edgecolor='face', alpha=alpha)
            #ax.add_feature(feat)
            ax.add_feature(cfeature.OCEAN, **kwargs)
        if 'rivers' in features:
            ax.add_feature(cfeature.RIVERS, **kwargs)
        if 'land' in features:
            #feat = cfeature.NaturalEarthFeature(category='physical', name='land', facecolor=cfeature.COLORS['land'], scale='10m', edgecolor='face', alpha=alpha)
            #ax.add_feature(feat)
            ax.add_feature(cfeature.LAND, **kwargs)
        if 'countries' in features:
            ax.add_feature(cfeature.BORDERS, **kwargs)
        if 'countries_highres' in features:
            feat = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m')
            ax.add_feature(feat, **kwargs)
        if 'coastlines' in features:
            ax.add_feature(cfeature.COASTLINE, **kwargs)
        if 'coastlines_highres' in features:
            ax.coastlines(resolution='10m', **kwargs)

    if latlon_format:
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)


    return ax


def plot_ztdata(data_xr_sel, varname, ax=None, only_contour=False, get_ds=False, **kwargs):
    """
    

    Parameters
    ----------
    data_xr : TYPE
        DESCRIPTION.
    varname : TYPE
        DESCRIPTION.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        the figure axis. The default is None.
    only_contour : bool, optional
        Wheter to plot contour lines of the dataset. The default is False.
    **kwargs : TYPE
        properties to give on to the pcolormesh function.
    
    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    pc : matplotlib.collections.QuadMesh
        DESCRIPTION.
    
    """
    
    if not ax: ax=plt.gca()
    
    if len(data_xr_sel[varname].shape) != 2:
        raise Exception(f'ERROR: unexpected number of dimensions in requested squeezed variable ({data_xr_sel[varname].shape}), first use data_xr.isel(stations=int) to select a single station') #TODO: can also have a different cause, improve message/testing?
    
    #repair zvalues at wl/wl (filling nans and clipping to wl/bl). bfill replaces nan values with last valid value, this is necessary to enable pcolormesh to work. clip forces data to be within bl/wl
    #TODO: put clip in preproces_hisnc to make plotting easier?
    data_xr_sel['zcoordinate_c'] = data_xr_sel['zcoordinate_c'].bfill(dim='laydim').clip(min=data_xr_sel.bedlevel,max=data_xr_sel.waterlevel)
    data_xr_sel['zcoordinate_w'] = data_xr_sel['zcoordinate_w'].bfill(dim='laydimw').clip(min=data_xr_sel.bedlevel,max=data_xr_sel.waterlevel)
    
    # generate 2 2d grids for the x & y bounds (you can also give one 2D array as input in case of eg time varying z coordinates)
    data_fromhis_zcor = data_xr_sel['zcoordinate_w'].to_numpy() 
    data_fromhis_zcor = np.concatenate([data_fromhis_zcor,data_fromhis_zcor[[-1],:]],axis=0)
    time_np = data_xr_sel.time.to_numpy()
    time_cor = np.concatenate([time_np,time_np[[-1]]])
    time_mesh_cor = np.tile(time_cor,(data_fromhis_zcor.shape[-1],1)).T
    if only_contour:
        pc = data_xr_sel[varname].plot.contour(ax=ax, x='time', y='zcoordinate_c', **kwargs)
    else:
        #pc = data_xr_sel[varname].plot.pcolormesh(ax=ax, x='time', y='zcoordinate_w', **kwargs) #is not possible to put center values on interfaces, som more difficult approach needed
        pc = ax.pcolormesh(time_mesh_cor, data_fromhis_zcor, data_xr_sel[varname], **kwargs)
   
    return pc

