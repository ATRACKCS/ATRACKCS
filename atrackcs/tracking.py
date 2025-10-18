# stormtracker/tracking.py

import pandas as pd
import geopandas as gpd
from datetime import timedelta
from shapely.ops import unary_union
import numpy as np
import os
import uuid
import tqdm
import time
import numbers as checknumbers
import pickle
from .config import ATRACKCSConfig

#msc_counter = 1

def finder_msc(pathResults, config):
    """
    Tracks convective systems based on polygon overlap percentage.
    """ 
    global msc_counter
    msc_counter = 1 # Reset counter for a fresh run
    
    threshold_overlapping_percentage = config.THRESHOLD_OVERLAPPING_PERCENTAGE
    UTM_WORLD_ZONE = 3857 
    
    # Preparing Dataframe
    pickle_file = os.path.join(pathResults, "polygons_cores.pkl")
    if not os.path.exists(pickle_file):
        print(f"Error: {pickle_file} not found. Run read_identify_mcs_parallel first.")
        return None
        
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    sup = pd.concat(data, ignore_index=True)
    if sup.empty:
        print("No storms identified. Tracking skipped.")
        return None
        
    sup['Tb'] = np.arange(1,len(sup['Tb'])+1)
    sup = sup.to_crs(UTM_WORLD_ZONE)
    
    pd.options.mode.chained_assignment = None
    sup["belong"] = 0
    sup["intersection_percentage"] = np.nan
    sup = (sup.sort_values(['time', 'area_tb'], ascending=[True, False]).reset_index(drop=True))
    
    # Generating tracks
    print("Estimating trajectories: ")
    for isites in tqdm.tqdm(sup.index, desc="Tracking Polygons"):
        # Checking when the record (spot) has not yet been processed or attaching to a track
        if sup.at[isites, "belong"] == 0:
            try: #I dont think this is necessary 
                #Assigning the counter number to identify the track(range 1)
                sup.at[isites, "belong"] = msc_counter 
                
                # Start and end time
                date_initial = sup.loc[isites].time
                date_final = date_initial + timedelta(hours=1)

                #Reference spot at time t
                poli_ti = sup.loc[isites].to_frame().T
                #Establishing EPSG:32718 - UTM zone 18S plane coordinate system
                poli_ti = gpd.GeoDataFrame(poli_ti, geometry='geometry', crs=UTM_WORLD_ZONE)
                
                # Candidates at t+1 
                poli_ti1 = sup.loc[sup.time == date_final]
                #Intersection of reference spot with spots one hour later
                if poli_ti1.empty:
                    sup = sup.drop(index=isites)
                    continue
                    
                intersect = gpd.overlay(poli_ti, poli_ti1, how='intersection',keep_geom_type=False) 
                #Check if instersect is empty, if so, continue with the next isites.
                if intersect.empty:
                    #Drop the reference polygon from sup if it doesn't intersect with anything at t+1
                    sup = sup.drop(index=isites)
                    continue   

                #When there are more than one intersection spot
                if len(intersect) > 1:
                    #sorting the candidates to continue the track by intersected area
                    intersect['area_intersected'] = intersect.area
                    sorted_intersect = intersect.sort_values(by="area_intersected", ascending=False)
                    #Extracting the intersection spot with the maxima area of intersection
                    sorted_intersect['percentage_overlapping'] = (sorted_intersect.area.values)*100/(poli_ti.area.values)
                    #checking if the polygona meets the area overlapping criteria in order
                    if isinstance(threshold_overlapping_percentage, checknumbers.Real):
                        #Check if at least one percentage is higher than thershold
                        sorted_intersect = sorted_intersect[sorted_intersect['percentage_overlapping'] >= 25].reset_index(drop=True)
                        if (sorted_intersect['percentage_overlapping'] >= threshold_overlapping_percentage).sum() >= 1:
                            intersect_percentage = sorted_intersect.iloc[0]["percentage_overlapping"]
                            #If the percentage overlapping meets the criteria for the higher area overlapping, is selected
                            sup.at[isites, "intersection_percentage"] = intersect_percentage
                            spot_tracked = sorted_intersect.iloc[0]
                        #if none, drop the spot due intersection is below threshold and continue with next isites 
                        else:
                            sup.drop(isites, inplace = True)
                            continue
                            #print ("The convective system " + str(isites) + " was dropped to the data due intersections are below that the threshold: "+ str(intersect_percentage)) 
                    #Condition if threshold overlapping percentage is not activated keep the polygon with the higher area intersection.
                    else:
                        spot_tracked = sorted_intersect.iloc[0] #select the higher one
                        intersect_percentage = spot_tracked["percentage_overlapping"]
                        sup.at[isites, "intersection_percentage"] = intersect_percentage 
                        #print ("There was a " + str(intersect_percentage) + "% intersection") 
                else:
                    #Calculate the intersect percentage
                    #TODO: This needs to be revised, for some reason its retrieving area in degrees2 but it is working for the purpose
                    intersect_percentage =((intersect.area.values)*100/(poli_ti.area.values))[0]
                    #Condition if threshold overlapping percentage is activated
                    if isinstance(threshold_overlapping_percentage, checknumbers.Real):
                        if intersect_percentage >= threshold_overlapping_percentage:
                            sup.at[isites, "intersection_percentage"] = intersect_percentage  
                            #print ("There was a " + str(intersect_percentage) + "% intersection")
                            spot_tracked = intersect  
                        else: #If the spot fails with threshold overlapping percentage is dropped
                            sup.drop(isites, inplace = True)
                            continue
                            #print ("The convective system " + str(isites) + " was dropped to the data due intersection is below that the threshold: "+ str(intersect_percentage))
                    #Condition if threshold overlapping percentage is not activated
                    else:
                        sup.at[isites, "intersection_percentage"] = intersect_percentage #Index, col 
                        #print ("There was a " + str(intersect_percentage) + "% intersection")
                        spot_tracked = intersect

                #Attaching found spot to the track
                time_intersect = spot_tracked["time_2"]
                tb_intersect = spot_tracked["Tb_2"]
                if hasattr(tb_intersect, "values"):  # if it's a pd.series or similar
                    tb_intersect = tb_intersect.values[0]
                    time_intersect = time_intersect.values[0]
                #assing the track 
                sup.loc[(sup.time == time_intersect) & (sup.Tb == tb_intersect), "belong"] = msc_counter
                #next track
                msc_counter +=1
                #print ("The convective system: " + str(isites) + " - belongs to the track: " +str(int(msc_counter)))
                continue

            except KeyError as e: #something did not worked for a strage reason
                print(f"Error to process tracking in identified polygon with index {isites}: {e}")
                #In the first step if a polygon fails drop it
                sup.drop(isites, inplace = True)
                msc_counter +=1
                continue

        # Checking when the record (spot) has been processed or attaching to a track ##desde aca           
        elif sup.at[isites, "belong"] in range(1, msc_counter):
            try:
                #Generating start and end time date (time t and time t+1) 
                date_initial = sup.loc[isites].time
                date_final = date_initial + timedelta(hours=1)
            
                #Reference spot
                poli_ti = sup.loc[isites].to_frame().T
                #Establishing EPSG:32718 - UTM zone 18S plane coordinate system                 
                poli_ti = gpd.GeoDataFrame(poli_ti, geometry='geometry', crs=UTM_WORLD_ZONE)
                #Getting the id track previously estimate
                index_track = poli_ti.belong
                #Spots one hour later that do not belong to any track 
                poli_ti1 = sup.loc[(sup.time == date_final) & (sup.belong == 0)]
                if poli_ti1.empty:
                    continue
                #Intersection of reference spot with spots one hour later
                intersect = gpd.overlay(poli_ti, poli_ti1, how='intersection',keep_geom_type=False)
                #If the polygon do not intersect with any others, finish the track #TODO
                if intersect.empty:
                    continue 
                    
                if len(intersect) > 1:
                    intersect['area_intersected'] = intersect.area
                    sorted_intersect = intersect.sort_values(by="area_intersected", ascending=False)
                    sorted_intersect['percentage_overlapping'] = (sorted_intersect.area.values)*100/(poli_ti.area.values)
                    if isinstance(threshold_overlapping_percentage, checknumbers.Real):
                        #Check if at least one percentage is higher than thershold
                        sorted_intersect = sorted_intersect[sorted_intersect['percentage_overlapping'] >= 25].reset_index(drop=True)
                        if (sorted_intersect['percentage_overlapping'] >= threshold_overlapping_percentage).sum() >= 1:
                            intersect_percentage = sorted_intersect.iloc[0]["percentage_overlapping"] 
                            #If the percentage overlapping meets the criteria for the higher area overlapping, is selected
                            sup.at[isites, "intersection_percentage"] = intersect_percentage
                            spot_tracked = sorted_intersect.iloc[0]
                        else:
                            continue
                            #print ("The convective system " + str(isites) + " was dropped to the data due intersection is below that the threshold: "+ str(intersect_percentage)) 
                    #Condition if threshold overlapping percentage is not activated keep the polygon with the higher area intersection.
                    else:
                        spot_tracked = sorted_intersect.iloc[0] #select the higher one
                        intersect_percentage = spot_tracked["percentage_overlapping"]
                        sup.at[isites, "intersection_percentage"] = intersect_percentage 
                #If intersects only with one polygon.
                else: 
                    #TODO: in a potential merge of 2 tracks into one, change the track "id" 
                    # assgining the same belong to both tracks
                    intersect_percentage =((intersect.area.values)*100/(poli_ti.area.values))[0]
                    #Condition if threshold overlapping percentage is activated
                    if isinstance(threshold_overlapping_percentage, checknumbers.Real):
                        if intersect_percentage >= threshold_overlapping_percentage:
                            sup.at[isites, "intersection_percentage"] = intersect_percentage  
                            #print (f"There was a " + str(intersect_percentage) + "% intersection")
                            spot_tracked = intersect  
                        else: #If the spot fails with threshold overlapping percentage is dropped
                            sup.drop(isites, inplace = True)
                            #print (f"The convective system " + str(isites) + " was dropped to the data due intersection is below that the threshold: "+ str(intersect_percentage))
                    #Condition if threshold overlapping percentage is not activated
                    else:
                        sup.at[isites, "intersection_percentage"] = intersect_percentage #Index, col 
                        #print (f"There was a " + str(intersect_percentage) + "% intersection")
                        spot_tracked = intersect             
    
                #Attaching found spot to the track
                time_intersect = spot_tracked["time_2"]
                tb_intersect = spot_tracked["Tb_2"]
                if hasattr(tb_intersect, "values"):  # si es una Serie o similar
                    tb_intersect = tb_intersect.values[0]
                    time_intersect = time_intersect.values[0]
                sup.loc[(sup.time == time_intersect) & (sup.Tb == tb_intersect), "belong"] = index_track.values[0]
                #msc_counter +=1
                    #print ("The convective system: " + str(isites) + " - belongs to the track: " +str(int(msc_counter)))
            except KeyError as e:
                print(f"Error to process tracking in identified polygon with index {isites}: {e}")
                pass
                
        #Progress bar
        time.sleep(0.01)    
       
    # #Transforming plane coordinates to geodesics  coordinates   
    sup["centroids"] = sup["centroids"].to_crs(4326)
    sup = sup.to_crs(4326) 
    
    #Creating an original index
    sup["id_gdf"] = None
    sup["id_gdf"] = sup.index
    return sup
            
 