class ATRACKCSConfig:
    """Default configuration settings for the storm tracking library."""
    
    # Paths (These will be *required* inputs to the main function, but we define them here for clarity)

    # pathTb = '/home/user/data/tb/'
    # pathP = '/home/user/data/p/'
    # pathResults = '/home/user/results/'
    
    # Thresholds
    TB_VALUE = 241            # K (Main cold cloud threshold)
    TB_OVERSHOOT = 225        # K (Coldest cloud top threshold for deep convection)
    AREA_VALUE = 2000         # km^2 (Minimum area for a system)
    PP_RATES = 2              # mm/h (Minimum precipitation rate for classification)
    DROP_EMPTY_PRECIPITATION = True #True/False 
    
    # Tracking and Overlap
    THRESHOLD_OVERLAPPING_PERCENTAGE = 25 # Minimum overlap for a system to be tracked as the same one (%)
    DURATION = 5              # Minimum duration in hours for a system to be considered "Deep Convection"

    # Spatial/Temporal
    UTC_LOCAL_HOUR = 0        # Time shift (hours) (optional) for local/regional studies.
    
    