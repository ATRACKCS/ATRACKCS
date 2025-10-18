import pandas as pd
import geopandas as gpd
from datetime import timedelta
import numpy as np
import math
import os
import math
import uuid
import tqdm
import time
from geopy.distance import geodesic
from .config import ATRACKCSConfig

"""
Summary of storm features.
@authors:  Vanessa Robledo (vanessa-robledodelgado@uiowa.edu)
"""

def distance_centroids(row):
    """
    Function to estimate the distance between two georreferenced points.
    This function execute inside the function distance_direction_Tracks().
    Inputs:
    * row: GeoDataFrame, containing the shift of the MCS at time t (centroid_)
      and the MCS at time t+1 (centroids_distance)
    
    Outputs:
    * distance: float, between the geometric centroids.
    """
    try:
        d = geodesic((row.centroids.y,row.centroids.x),(row.centroids_distance.y,row.centroids_distance.x)).km
    except:
        d = np.nan
    return d

def direction_points(row, mode = "u"):
    """
    Function to estimate the direction between two georreferenced points.
    
    Inputs:
    * row: GeoDataFrame, containing the shift of the MCS at time t (centroid_)
      and the MCS at time t+1 (centroid_2)
    * mode: str(u, v, deg), select the result of based on the output

        
    Outputs: The output is in function of the mode selected
    * deg: direction in degrees between the two points
    * u: normalized vector component
    * v: normalized vector component
    """
    
    try:
        #distance vector between the centroids of two points - in x: lon and y: lat
        distance = [row.centroids.x - row.centroid_2.x, row.centroids.y - row.centroid_2.y]
        
        #Normal Vector
        norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2) #Based on Pitagoras Teorem
        direction_normalized = [distance[0] / norm, distance[1] / norm] #Divided by normalized vector
        u = direction_normalized[0]; v = direction_normalized[1]
        
        #Conversion of vector components into direction degrees     
        direccion_deg = uv_to_degrees(u,v)
        direccion_deg = direccion_deg.round(1)
    except:
        direccion_deg = np.nan
        u = np.nan; v = np.nan
        
    if mode == "u":
        var = u
    elif mode == "v":
        var = v
    elif mode == "deg":
        var = direccion_deg
    return var

def uv_to_degrees(U,V):
    """
    Calculates the direction in degrees from the u and v components.
    Takes into account the direction coordinates is different than the 
    trig unit circle coordinate. If the direction is 360 then returns zero
    (by %360). 

    Calculates the direction (in degrees) toward which the motion is moving,
    using meteorological orientation (0° = toward north, 90° = toward east).

    This function execute inside the function distance_direction_Tracks().
    
    Inputs:
    * U: float, west/east direction (from the west is positive, from the east is negative)
    * V: float, south/noth direction (from the south is positive, from the north is negative)
      
    Outputs:
    * direction in degrees: float  
    """
    WDIR= (90-np.rad2deg(np.arctan2(V,U)))%360
    return WDIR

def distance_direction_Tracks(sup):
    """
    Function to estimate the distance and direction of the MCS's that compose 
    the tracks. This function execute inside the function feature_Tracks().
     
    Inputs:
    * sup: GeoDataframe, result data generated in the process features_Tb() or features_P().

    Outputs:
    * Geodataframe, with the distances and directions of the MCS's estimated.
    """
    
    #Making new columns
    sup["distance_c"] = None; contador = 0
    sup["direction"] = None; sup["u"] = None; sup["v"] = None

    tracks = sup.belong.unique(); len_track = len(tracks)

    #Estimating distance and direction between two points   
    
    print("Estimating distance and direction between geometric centroids: ")
    for track, progress in zip(tracks, tqdm.tqdm(range(len_track-1))):
        
        #____________________distance___________________________
        track_df = sup.loc[sup["belong"] == track, sup.columns]        
        #Generating centroid to compare (time i + 1) in another column               
        track_df["centroids_distance"] = track_df["centroids"].shift()
        
        #Calculating the distance between centroids (time i and time i + 1)        
        track_df["distance_c"] = track_df.apply(lambda row: distance_centroids(row), axis = 1)
        
        #Obtaining the ids of the centroids (spots) that were obtained.
        index_track_distancia = track_df.index    
        
        #Replacing the obtained distance values in the original Dataframe
        sup.loc[index_track_distancia, "distance_c"] = track_df["distance_c"]
        
        #____________________direction___________________________

        #Generating centroid to compare (time i - 1) in another column               
        track_df["centroid_2"] = track_df["centroids"].shift(-1)
        
        #Calculating the direction between centroids ((time i and time i - 1))
        track_df["direction_fake"] = track_df.apply(lambda row: direction_points(row, mode = "deg"), axis = 1)
        track_df["u_fake"] = track_df.apply(lambda row: direction_points(row, mode = "u"), axis = 1)
        track_df["v_fake"] = track_df.apply(lambda row: direction_points(row, mode = "v"), axis = 1)

        #Aligning the Dataframe so that the calculated direction corresponds to time i + 1
        track_df["direction"] = track_df["direction_fake"].shift(1)
        track_df["u"] = track_df["u_fake"].shift(1)
        track_df["v"] = track_df["v_fake"].shift(1)

        del track_df["direction_fake"]; del track_df["v_fake"]; del track_df["u_fake"]
    
        
        #Replacing the obtained values of direction (degrees) in the original dataframe
        index_track_direccion = track_df.index    
        sup.loc[index_track_direccion, "direction"] = track_df["direction"]   
        sup.loc[index_track_direccion, "u"] = track_df["u"]   
        sup.loc[index_track_direccion, "v"] = track_df["v"]   
                               
        contador +=1
        #porcentaje = round((contador*100./len_track),2)
        
        #Progress bar
        time.sleep(0.01)     

    sup.distance_c = sup.distance_c.astype(float)
    sup.direction = sup.direction.astype(float)
    sup.u = sup.u.astype(float)
    sup.v = sup.v.astype(float)


    return sup

def resume_track(sup, pathResults, config):
    """
    Function for calculating characteristics associated with each tracking
    average speed, average distance, average direction, 
    total duration, average area, total distance traveled, total time traveled
    
    Inputs:
    sup: DataFrame generated in process "finder_msc2"
    initial_time_hour: Default is 0, but could chnage based on in a specific hour duration tracks
    
    Outputs:
    DataFrame containing tracks and features
    """ 
    
    reg_sup = distance_direction_Tracks(sup)

    NUEVODF = sup.copy()
    NUEVODF = NUEVODF.set_index(['belong', 'id_gdf']).sort_index()

    #Replacing old id for spots and tracks based on alphanumeric code 16 and 20
    #characteres respectively    
    new_belong = []
    new_track_id = []
    
    for track_id in NUEVODF.index.levels[1]:
        new_track_id.append(str(uuid.uuid4())[-22:])
    dic_replace_track_id = dict(zip(NUEVODF.index.levels[1].values, new_track_id))
    
    for belong_id in NUEVODF.index.levels[0]:
        new_belong.append(str(uuid.uuid4())[:13])
    dic_replace_belong = dict(zip(NUEVODF.index.levels[0].values, new_belong))
    
    reg_sup_res =  NUEVODF.reset_index()
    
    reg_sup_res.belong = reg_sup_res.belong.replace(dic_replace_belong)
    reg_sup_res.id_gdf = reg_sup_res.id_gdf.replace(dic_replace_track_id)
       
    reg_sup_res = reg_sup_res.drop(labels=['Tb'],axis=1)
       
    reg_sup = reg_sup_res.set_index(["belong" , "id_gdf"]).sort_index()
    #print(reg_sup.head())
    #Calculate direction of movement.
    #reg_sup = distance_direction_Tracks(reg_sup)

    #Attaching to the dataframe total duration and total distance for each spot. Each spot 
    #has the register of the features of his own track
    reg_sup["total_duration"] = None
    reg_sup["total_distance"] = None
    
    count_df = reg_sup_res.id_gdf.groupby(by = [reg_sup_res.belong]).count()
    sum_df = reg_sup_res.distance_c.groupby(by = [reg_sup_res.belong]).sum()
    print("Estimating distance and duration of tracks...")
    for _b in count_df.index: 
    
        count_value = count_df.loc[_b]
        sum_value = sum_df.loc[_b]

        reg_sup.loc[_b, "total_duration"] = count_value
        reg_sup.loc[_b, "total_distance"] = sum_value

    #Filter events based on duration    
    reg_deep_convection = reg_sup[reg_sup['total_duration'] >= config.DURATION]
    
    if reg_deep_convection.empty:
        print(f"No tracks found with duration >= {config.DURATION} hours.")
        return None

    reg_deep_convection['mean_velocity'] = reg_deep_convection["total_distance"]/reg_deep_convection["total_duration"]
    
    #extraigo tambien el original 
    reg_deep_convection['time'] = pd.to_datetime(reg_deep_convection['time'], format="%Y-%m-%d %H:%M:%S")
    groupbelong2 = reg_deep_convection.groupby(level=[0])
    sorttime_deep_convect = groupbelong2.apply(lambda x: x.sort_values(by='time'))
    sorttime_deep_convect.rename(columns={'tb_225': 'tb_overshoot'}, inplace=True)
    sorttime_deep_convect = sorttime_deep_convect[['time', 'geometry', 'area_tb','tb_overshoot','pp_10rate', 'volum_pp', 'mean_pp',
       'mean_pp_above_rate', 'hvy_prec_frac', 'centroids', 'distance_c', 'direction', 'u', 'v',
       'total_duration', 'total_distance','mean_velocity']]
    #print(sorttime_deep_convect.columns)
    sorttime_deep_convect = sorttime_deep_convect.droplevel(0)
    sorttime_deep_convect.index.set_names(['track_id', 'polygon_id'], inplace=True)
    #Saving as .csv 
    sorttime_deep_convect.to_csv(pathResults + "resume_DeepConvection_"+str(reg_sup.time.min())[:-7]+"_"+str(reg_sup.time.max())[:-7]+".csv")

    return sorttime_deep_convect