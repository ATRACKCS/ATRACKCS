import xarray as xr
import dask
import numpy as np
import pandas as pd
from scipy import ndimage
import geopandas as gpd
from datetime import timedelta
import rioxarray
import rasterio
import glob
import numbers as checknumbers
from shapely.geometry import MultiPolygon, Polygon, shape, Point, MultiPoint, mapping 
from shapely.wkt import loads
from shapely.ops import unary_union
import psutil
from joblib import Parallel, delayed
import time
import warnings
import os
import multiprocessing
import copy
import pickle
from .config import ATRACKCSConfig # Import configuration

# Configure multiprocessing start method once (at module level)
multiprocessing.set_start_method('fork', force=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*CFTimeIndex.*")

# --- Utility Functions ---

def print_memory_usage(tag=""):
    """Prints the current memory usage of the process with an optional tag."""
    process = psutil.Process(os.getpid())
    mem_MB = process.memory_info().rss / 1024**2
    print(f"[{tag}] Memory usage: {mem_MB:.2f} MB")

def preprocess_mask(data_id, variable):
    # (Keep your existing preprocess_mask function here)
    structure = ndimage.generate_binary_structure(2, 2)
    arr = data_id[variable].isel(time=0).values
    arr_filt = ndimage.binary_closing(arr, structure=np.ones((3,3)))
    arr_filt = ndimage.binary_opening(arr_filt, structure=np.ones((3,3)))
    arr_filt = ndimage.binary_fill_holes(arr_filt).astype(int)
    
    blobs, num_blobs = ndimage.label(arr_filt, structure=structure)
    distance = ndimage.distance_transform_edt(blobs)
    blobs[distance != 1] = 0
    blobs = blobs.astype(float)
    blobs_3d = np.expand_dims(blobs, axis=0)
    
    data_id[variable].data[:] = blobs_3d
    filtered_data = data_id.where(data_id != 0, np.nan)
    
    return filtered_data

def polygon_identify(data_id, variable):
    """Identify polygon features in binary masks using convex hulls."""

    data_id = preprocess_mask(data_id, variable)
    df = data_id.to_dataframe().reset_index().dropna()
    
    if df.empty or 'lon' not in df.columns or 'lat' not in df.columns:
        print("[polygon_identify] No data or missing columns.")
        return gpd.GeoDataFrame(columns=['time', variable, 'geometry'], geometry='geometry', crs=4326)
    
    try:
        gdf = gpd.GeoDataFrame(df[["time", variable]], geometry=gpd.points_from_xy(df.lon, df.lat))
    except Exception as e:
        print(f"[polygon_identify] Error creating GeoDataFrame: {e}")
        return gpd.GeoDataFrame(columns=['time', variable, 'geometry'], geometry='geometry', crs=4326)
    
    gdf['geo'] = gdf.geometry.apply(lambda x: x.coords[0])
    df_grouped = gdf.groupby(by=["time", variable])['geo'].apply(lambda x: MultiPoint(sorted(x.tolist()))).reset_index()
    
    #Converting the points-groups to polygons based on the Convex Hull
    df_grouped["geo"] = df_grouped["geo"].apply(lambda x: loads(x.convex_hull.wkt))
    df_grouped = df_grouped.rename(columns={"geo": "geometry"})
    gdf_final = gpd.GeoDataFrame(df_grouped, geometry="geometry", crs=4326)
    gdf_final = gdf_final.loc[gdf_final.geometry.geom_type == 'Polygon']
    
    return gdf_final

def merge_neighbours(gdf_tb, buffer_threshold, UTM_WORLD_ZONE):
    """
    Merge polygons that are spatially close using buffering and unary union,
    and intelligently combine attributes.
    """

    # Ensure all processing is done in projected CRS
    gdf_tb = gdf_tb.to_crs(UTM_WORLD_ZONE)
    
    # Explicitly define gdf_output with correct dtypes to avoid concat warning
    gdf_output = gdf_tb.iloc[0:0].copy()

    # Step 1: Buffer
    gdf_tb['buffered'] = gdf_tb.geometry.buffer(buffer_threshold)
    
    # Step 2: Merge geometries
    merged_geometry = unary_union(gdf_tb['buffered'])
    merged_polygons = [polygon for polygon in merged_geometry.geoms] if merged_geometry.geom_type == 'MultiPolygon' else [merged_geometry]

    # Step 3: For each merged polygon, find which original polygons intersect it
    for poly in merged_polygons:
        # Remove buffer effect
        poly_unbuffered = poly.buffer(-buffer_threshold)

        # Find overlapping original polygons
        overlaps = gdf_tb[gdf_tb['buffered'].intersects(poly)]
        
        if not overlaps.empty:
            # Step 4: Combine attributes
            areas = overlaps['area_tb']
            total_area = overlaps['area_tb'].sum()
            heavy_area = (overlaps['hvy_prec_frac'] * areas).sum()
            hvy_prec_frac_merged = heavy_area / total_area if total_area > 0 else np.nan
            new_row = {
                'geometry': poly_unbuffered,
                'time': overlaps['time'].iloc[0],
                'area_tb': total_area,
                'pp_10rate': overlaps['pp_10rate'].any(),
                'volum_pp': overlaps['volum_pp'].sum(),
                'mean_pp': np.average(overlaps['mean_pp'], weights=areas),
                'mean_pp_above_rate': np.average(overlaps['mean_pp'], weights=areas),
                'hvy_prec_frac': hvy_prec_frac_merged,
                'tb_225': overlaps['tb_225'].any()
            }
            # Append to output GeoDataFrame
            gdf_output = pd.concat(
                [gdf_output, gpd.GeoDataFrame([new_row], crs=gdf_tb.crs)],
                ignore_index=True
            )
    if gdf_output.empty:
        print("[merge_neighbours] No merged polygons were created.")
        return gpd.GeoDataFrame(columns=gdf_tb.columns, crs="EPSG:4326")
                
    # Step 5: Convert to geographic CRS for final output
    gdf_output = gdf_output.to_crs("EPSG:4326").reset_index(drop=True)
    gdf_output['Tb'] = np.arange(1, len(gdf_output) + 1)
    
    return gdf_output

def zero_edges(da):
    """Set boundary pixels of the DataArray to zero."""
    da['Tb'].loc[:, da.lat[0], :] = 0
    da['Tb'].loc[:, da.lat[-1], :] = 0
    da['Tb'].loc[:, :, da.lon[0]] = 0
    da['Tb'].loc[:, :, da.lon[-1]] = 0
    return da

def drop_small_areas(gdf_tb, area_Tb, UTM_WORLD_ZONE):
    gdf = gdf_tb.copy()
    gdf = gdf.to_crs(UTM_WORLD_ZONE)
    gdf_tb["area_tb"] = gdf.geometry.area / 10**6  
    gdf_tb["area_tb"] = gdf_tb["area_tb"].round(1)
    
    gdf_tb = gdf_tb.loc[gdf_tb['area_tb'] >= area_Tb]
    
    return gdf_tb

def clip_tb_pp_merge(sup, ds_tb, tb_overshoot, ds_p, pp_rates, drop_empty_precipitation=True):
    """Extract Tb and precipitation features for each polygon."""

    date = sup['time'].unique().astype(str)

    # Check if 'time' is a dimension in ds_tb and select if present
    if 'time' in ds_tb.dims:
        try:
            ds_tb = ds_tb.sel(time=date[0])
        except KeyError:
            print(f"[clip_tb_pp_merge] Time {date[0]} not found in ds_tb. Available times: {ds_tb['time'].values}")
            return sup

    # If no 'time' dimension, assume data is already for the correct time
    ds_tb = ds_tb.rio.write_crs(4326).rename({"lat": "y", "lon": "x"})

    # Similarly for ds_p
    if 'time' in ds_p.dims:
        try:
            ds_p = ds_p.sel(time=date[0])
        except KeyError:
            print(f"[clip_tb_pp_merge] Time {date[0]} not found in ds_p. Available times: {ds_p['time'].values}")
            return sup
    # If no 'time' dimension, assume data is already for the correct time
    ds_p = ds_p.rio.write_crs(4326).rename({"lat": "y", "lon": "x"})
    
    geometries = sup.geometry
    sup['pp_10rate'] = False
    sup['volum_pp'] = np.nan
    sup['mean_pp'] = np.nan
    sup['mean_pp_above_rate'] = np.nan
    sup['hvy_prec_frac'] = np.nan
    sup['tb_225'] = False
    areapixel = (0.1 * 111.11 * 1000) * (0.1 * 111.11 * 1000)        #area pixel GPM (m2)

    for index_t in sup.index:
        _polygon = geometries.iloc[index_t]
        coords = np.dstack((_polygon.exterior.coords.xy[0], _polygon.exterior.xy[1]))
        geom = [{'type': 'Polygon', 'coordinates': [coords[0]]}]
        
        #Keeping Polygons with at least 5 pixels with temperature <= 225K 
        blob_clipped_tb = ds_tb.rio.clip(geom, geometries.crs, drop=False)
        blob_clipped_tb = blob_clipped_tb.where(blob_clipped_tb <= tb_overshoot)
        #systems with overshooting tops colder than 225K
        if blob_clipped_tb.notnull().sum() >= 5:
            sup.loc[index_t, "tb_225"] = True
        
        blob_clipped_p = ds_p.rio.clip(geom, geometries.crs, drop=False)
        #Calculate the mean precipitation rate
        sup.loc[index_t, "mean_pp"] = round(float(blob_clipped_p.mean(skipna=True).values),4)
        
        # Label polygons with > of 2 mm/h or pp_rates
        blob_clipped = blob_clipped_p.where(blob_clipped_p >= pp_rates)
        #Calculate the mean precipitation rate if its extend is more than 5 pixels
        if blob_clipped.notnull().sum() >= 5:
            sup.loc[index_t, "mean_pp_above_rate"] = round(float(blob_clipped.mean(skipna=True).values), 4)
        
        #Calculate the volumetric precipitation (m3/h)
        areaobj = sup.loc[index_t, 'area_tb'] * 1e6  # from km2 to m2
        sup.loc[index_t, "volum_pp"] = round((blob_clipped_p.mean().item() / 1000) * areaobj, 2)  # m3/h
        
        #Cheking if the system has at least 1 pixel with 10 mm/h 
        if blob_clipped.where(blob_clipped >= 10).notnull().sum() >= 1:
            sup.loc[index_t, 'pp_10rate'] = True
            #Calculating the fraction of heavy precipitation area (>=10 mm/h)
            sup.loc[index_t,'hvy_prec_frac'] = (areapixel * (blob_clipped >= 10).sum().item() / areaobj)
            
    #Drop polygons without precipitation 
    if drop_empty_precipitation:
        sup = sup[sup['mean_pp'].notna()].reset_index(drop=True)
    
    return sup



# --- Core Identification Function ---

def identify_mcs(ds_tb, Tb, area_Tb, buffer_threshold, ds_p, pp_rates, tb_overshoot, UTM_WORLD_ZONE, drop_empty_precipitation):
    """Identify (MCS) using thresholds and morphology."""

    #Aplying threshold (241 is the first threshold)
    mask_tb = (ds_tb <= Tb).astype(int)
    datax = mask_tb.to_dataset(name="Tb").rio.write_crs(4326)
    # Apply zero edges 
    datax = zero_edges(datax)
    #Identification of polygons in the binary image
    gdf_tb = polygon_identify(datax, "Tb")
    #Dropping polygons with area less than specified
    gdf_tb = drop_small_areas(gdf_tb, area_Tb, UTM_WORLD_ZONE).reset_index(drop=True)
    if gdf_tb.empty: 
        return gdf_tb
    #Calculating Brightness Temperatrue and Precipitation Characteristics
    gdf_tb = clip_tb_pp_merge(gdf_tb, ds_tb, tb_overshoot, ds_p, pp_rates, drop_empty_precipitation)
    #Merging polygons that belong to the same systems
    gdf_tb = merge_neighbours(gdf_tb, buffer_threshold, UTM_WORLD_ZONE)
    #Extracting the centroids for tracking
    gdf_tb["centroids"] = gdf_tb.geometry.representative_point()
    gdf_tb.reset_index(inplace = True, drop = True) 
    #gdf_tb.drop(columns = "tb_225", inplace = True)
    return gdf_tb

# --- Parallel Processing Workers ---

def process_one_tb(file_tb, ds_p_hourly, config):
    """Process a single Tb file with preloaded hourly P data."""
        
    ds_t = xr.open_dataset(file_tb)['Tb']
    lat_name = next(dim for dim in ds_t.dims if 'lat' in dim.lower())
    lon_name = next(dim for dim in ds_t.dims if 'lon' in dim.lower())

    #Temporal resampling Tb data
    ds_t = ds_t.rename({lat_name: 'lat', lon_name: 'lon'})
    ds_t2 = ds_t.resample(time="1h").nearest(tolerance="1h")
    
    if ds_t2.isnull().all() and not ds_t.isel(time=1).isnull().all():
        ds_t2 = ds_t.isel(time=slice(1,2)).assign_coords(time=[ds_t['time'].values[0]])

    #Spatial interpolation of Data    
    ds_t = ds_t2.interp(lat=ds_p_hourly.lat, lon=ds_p_hourly.lon)
    ds_t = ds_t.rio.write_crs(4326)
    ds_t.attrs['crs'] = ds_t.rio.crs

    #Setting the time of preference
    if config.UTC_LOCAL_HOUR != 0:
        ds_t['time'] = ds_t['time'] - pd.Timedelta(hours=config.UTC_LOCAL_HOUR)
    
    t0 = ds_t.time.values[0]
    ds_p = ds_p_hourly.sel(time=t0)

    BUFFER_THRESHOLD = 0.1 # Degrees (Buffer for merging neighbor systems)
    UTM_WORLD_ZONE = 3857 
    #Identifying polygons using tb and P
    gdf = identify_mcs(
        ds_t, 
        config.TB_VALUE, 
        config.AREA_VALUE, 
        BUFFER_THRESHOLD, 
        ds_p, 
        config.PP_RATES,
        config.TB_OVERSHOOT,
        UTM_WORLD_ZONE,
        config.DROP_EMPTY_PRECIPITATION
    )
    
    ds_t.close()
    
    return gdf

def read_identify_mcs_parallel(pathTb, pathP, pathResults, config):
    """
    Read and process Tb and P data in parallel, returning a single GeoDataFrame.
    This is the primary data ingestion and identification function.
    """
    #List the files to process
    files_tb = sorted(glob.glob(os.path.join(pathTb, '*.nc*')))
    files_P = sorted(glob.glob(os.path.join(pathP, '*.nc*')))
    print(f"Number of Tb files found: {len(files_tb)}")
    
    if not files_tb or not files_P:
        print("No files found in the specified folders.")
        return None
     
    # Preload and resample all P files
    ds_p_all = xr.open_mfdataset(files_P, combine="by_coords", parallel=True,chunks={"time": 50})["precipitation"]
    
    # using lazily loading here as we don't need entire data at all time
    ds_p_hourly = ds_p_all.resample(time="1h").mean().rename("P")
    ds_p_hourly = ds_p_hourly.rio.write_crs(4326)
    ds_p_hourly.attrs['crs'] = ds_p_hourly.rio.crs

    #The raw P data from some years does not have a valid datetime index 
    #These lines convert CFTimeIndex to DatetimeIndex.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*CFTimeIndex.*")
        ds_p_hourly['time'] = ds_p_hourly.indexes['time'].to_datetimeindex(time_unit='ns')

    if not config.UTC_LOCAL_HOUR == 0: 
        ds_p_hourly['time'] = ds_p_hourly['time'] - pd.Timedelta(hours=config.UTC_LOCAL_HOUR)
     
    # Compute the resampled data in parallel
    with dask.config.set(scheduler='threads'): 
        ds_p_hourly = ds_p_hourly.compute()
     
    print_memory_usage("After loading all precipitation")
        
    # Process Tb files in parallel
    start_time = time.time()
    max_workers=-1
    results = Parallel(n_jobs=max_workers, verbose=0)(
        delayed(process_one_tb)(f, ds_p_hourly, config) for f in files_tb
    )
    end_time = time.time()
    print(f"Storms identification process completed in {end_time - start_time:.1f} seconds")
   
    output_pickle = os.path.join(pathResults, "polygons_cores.pkl")
    with open(output_pickle, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved polygons to {output_pickle}")

    print_memory_usage("All done")
    