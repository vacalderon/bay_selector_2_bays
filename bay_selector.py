#*******************************************************
#*******************************************************
#****       BAY SELECTOR PROGRAM                    ****
#****       Version (1.1)                           ****
#****       Developed: Victor A. Calderon, PhD, PE  ****
#****                  Murat Melek, PhD, PE         ****
#****                                               ****
#****       Walter P. Moore (c)                     ****
#****                                               ****
#****                                               ****
#*******************************************************
#*******************************************************

# IMPORTS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import scipy as sp
import math as math
#from LibUnitsMUS import *
import os as os
import composite_floor

#DEFINE VARIABLES

# ----------------------------------------------------------------------------
# | VARIABLES 
# ----------------------------------------------------------------------------
#
# girder_span = Span of girder, ft
# max_girder_depth = Max depth of girder, inches
# beam_span = Span of beam, ft
# max_beam_dept = Max depth of beam, inches
# max_beam_spacing = Spacing of internal bay beams, ft
# deck_type = Deck type, based on normal weight concrete, light wight concrete and thickness of concrete filling
# q_sdl = Dead load in psf
# q_ll = Live load in psf
# acc_limit =  Acceleartion limit state for floor vibration, based on AISC Steel Design Guide 11
# q_vib_sdl = Dead load used for vibration analysis, psf
# q_vib_ll = Live load used for vibration analysis, psf
# damping = Damping assumed for virbation analysis
# deflection_critical = Critical deflection L/240 or L/480
# design_priority = 'EmbodiedCarbon', 'Cost', 'Tonnage'
# concrete_type = Type of concrete
# aisc_database_filename = Default to AISC_Shapes_Database_v14.csv

 # DEFINE LOADING CONDITIONS
#
# SUPERIMPOSED DEAD LOAD
q_sdl_list = [20,50]
#
# LIVE LOAD
q_ll_list = [100,50]

# DEFINE LISTS

girder_spans =[20.0, 25.0, 30.0, 35.0, 40.0,45.0,50.0] # [25.0, 30.0, 35.0] #  ft
max_girder_depths = [36] # [48.0,42.0,36.0,30.0,24.0] #in
beam_spans = [20.0, 25.0, 30.0, 35.0, 40.0,45.0,50.0] # #  [20.0] #  ft
max_beam_depth = [48.0] # in
max_beam_spacing = 20
deck_types = ["W3_4.50_NW_20g", "W3_3.50_NW_20g", "W3_2.00_NW_20g", "W3_3.25_LW_20g", "W3_2.50_LW_20g", "W3_2.00_LW_20g", "W2_4.50_NW_20g", "W2_3.50_NW_20g", "W2_2.00_NW_20g", "W2_3.25_LW_20g", "W2_2.50_LW_20g", "W2_2.00_LW_20g"] # ["W3_4.50_NW_20g"] #   
acc_limit = 0.005
q_vib_sdl = 4.0 # psf
q_vib_ll = 11.0 # psf
damping = 0.03 # Higher damping ratios are more appropiate if there are other non structural component such as drywall, sofits, threads...
deflection_criticals = [False] # [True, False] #
design_priorities = ['EmbodiedCarbon'] # ['EmbodiedCarbon', 'Cost', 'Tonnage'] # 
concrete_types = ["WPM_5000_NWC","WPM_5000_LWC"] # ["WPM_3000_NWC","WPM_3500_NWC","WPM_4000_NWC","WPM_5000_NWC", "WPM_6000_NWC","WPM_8000_NWC","WPM_3000_LWC","WPM_3500_LWC","WPM_4000_LWC", "WPM_5000_LWC"] #  
aisc_database_filename = r"C:\Users\victorc\Documents\GitHub\composite_floor_designer\AISC_Shapes_Database_v14.csv"

for q_sdl in q_sdl_list:
    for q_ll in q_ll_list:
        for girder_span in girder_spans:
            for max_girder_depth in max_girder_depths:    
                for beam_span in beam_spans:
                    for deck_type in deck_types:
                        for deflection_critical in deflection_criticals:
                            for design_priority in design_priorities:
                                for concrete_type in concrete_types:
                                    max_beam_depth=max_girder_depth
                                    if deck_type[8:10]!=concrete_type[9:11] :
                                        continue

                                    [response,message] = composite_floor.composite_floor(girder_span, max_girder_depth, beam_span, max_beam_depth, max_beam_spacing, deck_type, q_sdl, q_ll, acc_limit, q_vib_sdl, q_vib_ll, damping, deflection_critical, design_priority, concrete_type, aisc_database_filename)
                                    if message =="Typical Wide-Flange Sections do not work for the given design parameters." :
                                        continue

                                    results_dict={'Super Imposed Dead Load, SD (psf)': q_sdl,
                                                  'Live Load, L (psf)': q_ll,
                                                  'Girder Span': girder_span,
                                                  'Max Girder Depth': max_girder_depth,
                                                  'Beam Span': beam_span,
                                                  'Max Beam Depth': max_beam_depth,
                                                  'Deck Type': deck_type,
                                                  'Design Priority': design_priority,
                                                  'Acceleration Limit': acc_limit,
                                                  'Deflection Critical':deflection_critical}
                                    dict = {k:[v] for k,v in response.items()} 
                                    results_dict.update(dict)
                                    DataFrame_Out = pd.DataFrame(results_dict)
                                    DataFrame_Out.to_csv('results.csv', mode='a', header=False)

#Generating input xstream datbase

xstream_input_df =pd.DataFrame(columns=["type", "support_size",	
                                        "support_material",	"beam_size", "beam_material", 
                                        "skew_angle", "delta_e", "vu_type", 
                                        "vu","pu", "beam_length"])

df = pd.read_csv(r"C:\Users\victorc\Documents\GitHub\composite_floor_designer\results.csv")

def replace_value(x):
    if x > 12:
        return x if x % 5 == 0 else math.ceil(x*1.05 // 5) * 5.0
    else:
        return 12

df['Ultimate Shear at Beam to Girder Connection (kips)'] = df['Ultimate Shear at Beam to Girder Connection (kips)'].apply(replace_value)

for index,row in df.iterrows():
    xstream_input_df["type"]="BG"
    xstream_input_df["support_size"]=df["Girder Section"]
    xstream_input_df["support_material"]="A992"
    xstream_input_df["beam_size"]=df[" Beam Section"]
    xstream_input_df["beam_material"]="A992"
    xstream_input_df["skew_angle"] = 0.0
    xstream_input_df["delta_e"] = 0.0
    xstream_input_df["vu_type"] = "ABS"
    xstream_input_df["vu"] = df['Ultimate Shear at Beam to Girder Connection (kips)']
    xstream_input_df["pu"] = 0.0
    xstream_input_df["beam_length"] = df["Beam Span"]

xstream_input_df.to_excel('xstream_input.xlsx', index=False)

print("ALL ANALYS COMPLETE")