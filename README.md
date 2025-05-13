# Neural-Network-Based-Restaurant-Visitor-Forecasting
project description:

    This project explores a neural network-based approach to forecast restaurant visitor counts using historical reservation and calendar data. 
    The goal is to build a predictive model that leverages real-world features such as reservation volumes, reservation timing, store metadata, and holiday indicators to produce accurate visitor estimates.
    The core of the project involves:
    
    Data Integration from multiple sources including Air and HPG reservation platforms
    
    Feature Engineering using reservation statistics, geographic info, and temporal attributes
    
    Model Training using a Multi-Layer Perceptron (MLP) with RMSLE as the primary loss metric
    
    Optimization Techniques such as log1p transformation, batch size adjustment, and learning rate tuning
    
    Loss Visualization to compare the effects of different optimizations across training curves
    
    Through systematic experimentation, the model achieves a stable RMSLE of approximately 0.223, indicating reasonable accuracy in predicting restaurant traffic patterns under various conditions.

csv file description:

    air_visit_data.csv	          
    Actual restaurant visitor counts from the Air platform (main dataset for training)
    
    air_reserve.csv	              
    Reservation data from the Air platform 
    
    hpg_reserve.csv	              
    Reservation data from the HPG platform 
    
    air_store_info.csv	          
    Restaurant metadata from Air, including genre, area, and location (latitude/longitude)
    
    hpg_store_info.csv	          
    Restaurant metadata from HPG, same as above
    
    store_id_relation.csv	        
    Mapping between Air and HPG store IDs
    
    date_info.csv	                
    Calendar data including whether the day is a holiday and what day of the week it is 
