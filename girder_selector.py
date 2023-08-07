#*******************************************************
#*******************************************************
#****       GIRDER SELECTOR PROGRAM                 ****
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
import os as os
import composite_floor
import random
import multiprocessing

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

def generate_random_loads(start, stop, step, count):
    random_list = []
    for _ in range(count):
        random_num = random.uniform(start, stop)
        random_list.append(round(random_num, 2))  # Round to 2 decimal places (optional)
        start += step
    return random_list

def effective_length(span1, span2):
    result = []

    for i in span1:
        span_eff = 0

        for j in span2:
            span_eff = (i + j) / 2
            result.append(span_eff)

    return result

def remove_duplicates_and_values(input_list, values_to_remove):
    # Filter out specific values
    filtered_list = [x for x in input_list if x not in values_to_remove]

    # Convert the filtered list to a set to remove duplicates
    unique_set = set(filtered_list)

    # Convert the set back to a list
    unique_list = list(unique_set)

    return unique_list
# SUPERIMPOSED DEAD LOAD
q_sdl_list = generate_random_loads(15,60,10,5) #psf
# LIVE LOAD
q_ll_list = generate_random_loads(50,150,10,10) #psf

girder_spans =[20.0, 25.0, 30.0, 35.0, 40.0,45.0,50.0] # [25.0, 30.0, 35.0] #  ft
max_girder_depths = [36] # [48.0,42.0,36.0,30.0,24.0] #in
beam_spans_1 = [20.0, 25.0, 30.0, 35.0, 40.0,45.0,50.0] # #  [20.0] #  ft
beam_spans_2 = [20.0, 25.0, 30.0, 35.0, 40.0,45.0,50.0]
max_beam_depth = [36.0] # in
max_beam_spacing = 20
deck_types = ["W3_4.50_NW_20g", "W3_3.50_NW_20g", "W3_2.00_NW_20g", "W3_3.25_LW_20g", "W3_2.50_LW_20g", "W3_2.00_LW_20g", "W2_4.50_NW_20g", "W2_3.50_NW_20g", "W2_2.00_NW_20g", "W2_3.25_LW_20g", "W2_2.50_LW_20g", "W2_2.00_LW_20g"] # ["W3_4.50_NW_20g"] #   
acc_limit = 0.005
q_vib_sdl = 4.0 # psf
q_vib_ll = 11.0 # psf
damping = 0.03 # Higher damping ratios are more appropiate if there are other non structural component such as drywall, sofits, threads...
deflection_criticals = [False] # [True, False] #
design_priorities = ['EmbodiedCarbon'] # ['EmbodiedCarbon', 'Cost', 'Tonnage'] # 
concrete_types = ["WPM_5000_NWC","WPM_5000_LWC"] # ["WPM_3000_NWC","WPM_3500_NWC","WPM_4000_NWC","WPM_5000_NWC", "WPM_6000_NWC","WPM_8000_NWC","WPM_3000_LWC","WPM_3500_LWC","WPM_4000_LWC", "WPM_5000_LWC"] #  
aisc_database_filename = r"./AISC_Shapes_Database_v14.csv"

l_eff=effective_length(beam_spans_1, beam_spans_2)
beam_eff_span = remove_duplicates_and_values(l_eff, beam_spans_1)

def parallel_composite_floor(q_sdl, q_ll):

    for i in q_sdl:
        for j in q_ll_list:
            for girder_span in girder_spans:
                for max_girder_depth in max_girder_depths:    
                    for beam_span in beam_eff_span:
                        for deck_type in deck_types:
                            for deflection_critical in deflection_criticals:
                                for design_priority in design_priorities:
                                    for concrete_type in concrete_types:
                                        max_beam_depth=max_girder_depth
                                        if deck_type[8:10]!=concrete_type[9:11] :
                                            continue
                                        [response,message] = composite_floor.composite_floor(girder_span, max_girder_depth, beam_span, max_beam_depth, max_beam_spacing, deck_type, i, j, acc_limit, q_vib_sdl, q_vib_ll, damping, deflection_critical, design_priority, concrete_type, aisc_database_filename)
                                        if message =="Typical Wide-Flange Sections do not work for the given design parameters." :
                                            continue

                                        results_dict={'Super Imposed Dead Load, SD (psf)': i,
                                                    'Live Load, L (psf)': j,
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

# Parallelize the outermost loops
def parallelize_outer_loops(q_sdl_list, q_ll_list):
    num_processes = multiprocessing.cpu_count()  # You can set the number of parallel processes here
    # Split the loading conditions into chunks for parallel processing
    chunk_size = 2
    sdl_chunks = [q_sdl_list[i:i + 1] for i in range(0, len(q_sdl_list), 1)]
    ll_chunks = [q_ll_list[i:i + 2] for i in range(0, len(q_ll_list), 2)]
    # Create a multiprocessing Queue to store the results from different processes
    result_queue = multiprocessing.Queue()
    # Create and start multiple processes for each chunk
    processes = []
    for sdl_chunk, ll_chunk in zip(sdl_chunks, ll_chunks):
        process = multiprocessing.Process(target=parallel_composite_floor, args=(sdl_chunk, ll_chunk))
        processes.append(process)
        process.start()
    # Wait for all processes to complete
    for process in processes:
        process.join()

if __name__ == "__main__":
    # ... (rest of the code remains unchanged)
    # Loop over the outermost lists and call the parallel function
    parallelize_outer_loops(q_sdl_list, q_ll_list)