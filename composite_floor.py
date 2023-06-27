# Version 1.1

def composite_floor(girder_span, max_girder_depth, beam_span, max_beam_depth, max_beam_spacing, deck_type, q_sdl, q_ll, acc_limit, q_vib_sdl, q_vib_ll, damping, deflection_critical, design_priority, concrete_type, aisc_database_filename):
    import pandas as pd
    import math
    import numpy as np
    from scipy.optimize import fsolve

    class Config:
        def __init__(self):
            self.floor_width = 1000  # ft if not specified.
            self.deflection_critical = deflection_critical
            self.beam_span = 30  # ft
            self.girder_span = 30  # ft

            # stud information
            self.stud_diameter = 0.75  # in
            self.stud_fu = 65  # ksi
            self.cost_per_stud = 1.5  # $ per stud

            # Min. nominal beam width for 2 stud rows:    5 ½”
            self.min_width_two_studs = 5.5
            # Min. nominal beam width for 3 stud rows:    8 ½”
            self.min_width_three_studs = 8.5

            # beam properties
            self.sectionType = "W"
            self.beam_depth_threshold = 12  # inches
            self.max_camber_allowed = 3  # in
            self.steel_cost = 1  # $ per pound

            # Deck Properties
            # topping unit_cost: $10/sq-ft
            # metal deck unit_cost: $10/sq-ft
            # carbon value : kg/tons - deck only

            self.deck_type = "W3_4.50_NW_20g"  # Initial Deck Type

            self.deck_library = {
                "W3_4.50_NW_20g": {"deck_depth": 3, "deck_gage": 20, "concrete_thickness": 4.50, "concrete_type": "NW",
                                "self_weight": 74.8, "max_unshored_span": 10, "carbon_value": 2150.05,
                                'deck_weight': 2.3, 'unit_cost': 20},
                "W3_3.50_NW_20g": {"deck_depth": 3, "deck_gage": 20, "concrete_thickness": 3.50, "concrete_type": "NW",
                                "self_weight": 62.7, "max_unshored_span": 10, "carbon_value": 2150.05,
                                'deck_weight': 2.3, 'unit_cost': 20},
                "W3_2.00_NW_20g": {"deck_depth": 3, "deck_gage": 20, "concrete_thickness": 2.00, "concrete_type": "NW",
                                "self_weight": 44.6, "max_unshored_span": 12, "carbon_value": 2150.05,
                                'deck_weight': 2.3, 'unit_cost': 20},
                "W3_3.25_LW_20g": {"deck_depth": 3, "deck_gage": 20, "concrete_thickness": 3.25, "concrete_type": "LW",
                                "self_weight": 45.8, "max_unshored_span": 12, "carbon_value": 2150.05,
                                'deck_weight': 2.3, 'unit_cost': 20},
                "W3_2.50_LW_20g": {"deck_depth": 3, "deck_gage": 20, "concrete_thickness": 2.50, "concrete_type": "LW",
                                "self_weight": 39.0, "max_unshored_span": 12, "carbon_value": 2150.05,
                                'deck_weight': 2.3, 'unit_cost': 20},
                "W3_2.00_LW_20g": {"deck_depth": 3, "deck_gage": 20, "concrete_thickness": 2.00, "concrete_type": "LW",
                                "self_weight": 34.4, "max_unshored_span": 13, "carbon_value": 2150.05,
                                'deck_weight': 2.3, 'unit_cost': 20},
                "W2_4.50_NW_20g": {"deck_depth": 2, "deck_gage": 20, "concrete_thickness": 4.50, "concrete_type": "NW",
                                "self_weight": 68.6, "max_unshored_span": 8, "carbon_value": 2150.05, 'deck_weight': 2.1,
                                'unit_cost': 20},
                "W2_3.50_NW_20g": {"deck_depth": 2, "deck_gage": 20, "concrete_thickness": 3.50, "concrete_type": "NW",
                                "self_weight": 56.5, "max_unshored_span": 9, "carbon_value": 2150.05, 'deck_weight': 2.1,
                                'unit_cost': 20},
                "W2_2.00_NW_20g": {"deck_depth": 2, "deck_gage": 20, "concrete_thickness": 2.00, "concrete_type": "NW",
                                "self_weight": 38.4, "max_unshored_span": 10, "carbon_value": 2150.05,
                                'deck_weight': 2.1, 'unit_cost': 20},
                "W2_3.25_LW_20g": {"deck_depth": 2, "deck_gage": 20, "concrete_thickness": 3.25, "concrete_type": "LW",
                                "self_weight": 41.1, "max_unshored_span": 10, "carbon_value": 2150.05,
                                'deck_weight': 2.1, 'unit_cost': 20},
                "W2_2.50_LW_20g": {"deck_depth": 2, "deck_gage": 20, "concrete_thickness": 2.50, "concrete_type": "LW",
                                "self_weight": 34.2, "max_unshored_span": 10, "carbon_value": 2150.05,
                                'deck_weight': 2.1, 'unit_cost': 20},
                "W2_2.00_LW_20g": {"deck_depth": 2, "deck_gage": 20, "concrete_thickness": 2.00, "concrete_type": "LW",
                                "self_weight": 29.6, "max_unshored_span": 11, "carbon_value": 2150.05,
                                'deck_weight': 2.1, 'unit_cost': 20}}

            self.deck_self_weight = self.deck_library[self.deck_type]["self_weight"]
            self.deck_max_span = self.deck_library[self.deck_type]["max_unshored_span"]
            self.deck_carbon_value = self.deck_library[self.deck_type]["carbon_value"]
            self.deck_weight = self.deck_library[self.deck_type]["deck_weight"]
            self.concrete_thickness = self.deck_library[self.deck_type]["concrete_thickness"]
            self.deck_depth = self.deck_library[self.deck_type]["deck_depth"]
     

            # Concrete Information
            self.concrete = "WPM_3000_NWC"

            # Steel Information
            self.steel_modulus_of_elasticity = 29000
            self.steel_carbon_value = 1052.35  # kg/tons

            # Loads
            self.q_sdl = 20  # psf
            self.q_ll = 100  # psf
            self.q_cons_ll = 20  # psf
            self.q_ponding = 6  # psf

            # Vibration Check
            # Damping
            self.damping = 0.025
            self.g = 386.4  # in/s2
            self.p_0 = 65  # lb
            self.q_vib_ll = q_vib_ll  # psf
            self.q_vib_sdl = q_vib_sdl  # psf

            # vibration limit
            self.acceleration_limit = 0.005  # .5%g

            self.c_g = 1.8  # 1.8: girder/beam, 1.6: girder/joist
            self.c_j = 2.0  # 2.0: typical, 1.0: edge

            # carbon value: kg/cu-yd
            self.concrete_library = {
                "WPM_3000_LWC": {"concrete_type": "LW", "unit_weight": 110, "concrete_strength": 3, "carbon_value": 398.07},
                "WPM_3500_LWC": {"concrete_type": "LW", "unit_weight": 110, "concrete_strength": 3.5, "carbon_value": 438.86},
                "WPM_4000_LWC": {"concrete_type": "LW", "unit_weight": 110, "concrete_strength": 4, "carbon_value": 438.86},
                "WPM_5000_LWC": {"concrete_type": "LW", "unit_weight": 110, "concrete_strength": 5, "carbon_value": 479.94},
                "WPM_3000_NWC": {"concrete_type": "NW", "unit_weight": 150, "concrete_strength": 3, "carbon_value": 223.39},
                "WPM_3500_NWC": {"concrete_type": "NW", "unit_weight": 150, "concrete_strength": 3.5, "carbon_value": 262.55},
                "WPM_4000_NWC": {"concrete_type": "NW", "unit_weight": 150, "concrete_strength": 4, "carbon_value": 262.55},
                "WPM_5000_NWC": {"concrete_type": "NW", "unit_weight": 150, "concrete_strength": 5, "carbon_value": 311.07},
                "WPM_6000_NWC": {"concrete_type": "NW", "unit_weight": 150, "concrete_strength": 6, "carbon_value": 328.55},
                "WPM_8000_NWC": {"concrete_type": "NW", "unit_weight": 150, "concrete_strength": 8, "carbon_value": 381.03}}

            self.concrete_carbon_value = self.concrete_library[self.concrete]["carbon_value"]

        def get_deck_properties(self):
            self.deck_self_weight = self.deck_library[self.deck_type]["self_weight"]
            self.deck_max_span = self.deck_library[self.deck_type]["max_unshored_span"]
            self.deck_carbon_value = self.deck_library[self.deck_type]["carbon_value"]
            self.deck_weight = self.deck_library[self.deck_type]["deck_weight"]
            self.concrete_thickness = self.deck_library[self.deck_type]["concrete_thickness"]
            self.deck_depth = self.deck_library[self.deck_type]["deck_depth"]
            self.concrete_carbon_value = self.concrete_library[self.concrete]["carbon_value"]


    class Composite_Beam:
        def __init__(self, section_size, beam_database):
            self.section_size = section_size
            self.steel_yield_stress = 50  # ksi
            self.deck_orientation = "perpendicular"
            self._number_of_studs_per_rib = 1
            self.beam_database = beam_database
            self.get_section_properties()
            self.steel_modulus_of_elasticity = 29000  # ksi
            self.distance_to_left_edge = 10000
            self.distance_to_right_edge = 10000
            self.unbraced_length = 1  # ft

        # calculate effective_width
        def calculate_effective_width(self):

            width_based_on_beam_span = float(self.beam_span) / 8

            effective_width_left = min(width_based_on_beam_span, self.distance_to_left_beam / 2, self.distance_to_left_edge)
            effective_width_right = min(width_based_on_beam_span, self.distance_to_right_beam / 2,
                                        self.distance_to_right_edge)

            self.effective_width = effective_width_left + effective_width_right

        def get_section_properties(self):

            beam_row = self.beam_database.loc[self.beam_database['AISC_Manual_Label'] == self.section_size]


            self.section_flange_width = float(beam_row['bf'])
            self.section_flange_thickness = float(beam_row['tf'])
            self.section_web_thickness = float(beam_row['tw'])


            self.section_k_depth = float(beam_row['kdes'] - self.section_flange_thickness)
            self.section_k_thickness = float(beam_row['k1'] + self.section_web_thickness / 2)

   
            self.section_depth = float(beam_row['d'])

            self.section_weight = float(beam_row['W'])

            self.section_area = float(beam_row['A'])

            self.plastic_modulus = float(beam_row['Zx'])

            self.moment_of_inertia = float(beam_row['Ix'])

            self.i_y = float(beam_row['Iy'])
            self.r_y = float(beam_row['ry'])
            self.j_c = float(beam_row['J'])
            self.s_x = float(beam_row['Sx'])
            self.c_w = float(beam_row['Cw'])

            if self.deck_orientation == "perpendicular":
                self.unbraced_length = 1
            else:
                self.unbraced_length = self.beam_span


    # def calculate_carbon(beam, girder, config):
    #     # deck
    #     deck_area = config.beam_span * config.girder_span
    #     deck_weight = config.deck_weight * deck_area * 0.0005  # in tons
    #     deck_carbon = config.deck_carbon_value * deck_weight  # kg

    #     # concrete
    #     average_thickness = config.concrete_thickness + config.deck_depth / 2
    #     concrete_volume = average_thickness / 12 * deck_area * 0.037037  # in cu-yd
    #     concrete_carbon = config.concrete_carbon_value * concrete_volume  # kg

    #     # girder
    #     girder_weight = girder.section_weight * girder.beam_span * 0.0005  # in tons
    #     girder_carbon = girder_weight * config.steel_carbon_value  # kg

    #     # beams
    #     beam_weight = beam.section_weight * beam.beam_span * 0.0005  # in tons
    #     total_beam_weight = beam_weight * (girder.beam_span / beam.beam_spacing + 1)
    #     beam_carbon = total_beam_weight * config.steel_carbon_value  # kg

    #     total_carbon = deck_carbon + concrete_carbon + girder_carbon + beam_carbon

    #     return total_carbon, deck_carbon, concrete_carbon, girder_carbon, beam_carbon

    def calculate_carbon(df, config):
        # deck
        deck_area = config.beam_span * config.girder_span
        deck_weight = config.deck_weight * deck_area * 0.0005  # in tons
        deck_carbon = config.deck_carbon_value * deck_weight  # kg

        # concrete
        average_thickness = config.concrete_thickness + config.deck_depth / 2
        concrete_volume = average_thickness / 12 * deck_area * 0.037037  # in cu-yd
        concrete_carbon = config.concrete_carbon_value * concrete_volume  # kg


        def calculate_girder_carbon(row):
            # girder
            girder_section_weight = row['W']
            
            girder_weight = girder_section_weight * config.girder_span * 0.0005  # in tons
            girder_carbon = girder_weight * config.steel_carbon_value  # kg

            return girder_carbon

        def calculate_beam_carbon(row):
            # beams           
            beam_section_weight = int(row['Beam Section'].split('X')[1])

            beam_weight = beam_section_weight * config.beam_span * 0.0005  # in tons
            total_beam_weight = beam_weight * (config.girder_span / config.beam_spacing + 1)
            beam_carbon = total_beam_weight * config.steel_carbon_value  # kg

            return beam_carbon

        
        df['Girder Embodied Carbon'] = df.apply(lambda x: calculate_girder_carbon(x), axis = 1)

        df['Beam Embodied Carbon'] = df.apply(lambda x: calculate_beam_carbon(x), axis = 1)

        df['Concrete Embodied Carbon'] = concrete_carbon
        df['Deck Embodied Carbon'] = deck_carbon
        df['Total Embodied Carbon'] = df['Beam Embodied Carbon'] + df['Girder Embodied Carbon'] + df['Concrete Embodied Carbon'] + df['Deck Embodied Carbon']

        return df

    def read_aisc_database(config):
        # Read AISC database
        # Remove unnecessary columns
        aisc_database = pd.read_csv(config.aisc_database_filename)

        beam_database_raw = aisc_database.loc[aisc_database['Type'] == config.sectionType]

        beam_database = beam_database_raw.drop(columns=beam_database_raw.iloc[:, 58::])

        # Convert fractions to float
        # this is to be used with the AISC section database
        def convert_to_float(frac_str):
            try:
                return float(frac_str)
            except ValueError:
                num, denom = frac_str.split('/')
                try:
                    leading, num = num.split('  ')
                    whole = float(leading)
                except ValueError:
                    try:
                        leading, num = num.split(' ')
                        whole = float(leading)
                    except ValueError:
                        whole = 0

                frac = float(num) / float(denom)

                if whole < 0:
                    return whole - frac
                else:
                    return whole + frac     

        # sort list by major moment of inertia
        beam_database = beam_database.loc[beam_database['d'] >= config.beam_depth_threshold]

        beam_database = beam_database.sort_values(by=['Ix'], ascending=False)

        beam_database['k1'] = beam_database.apply(lambda x: convert_to_float(x['k1']), axis=1)

        beam_database.apply(lambda x: convert_to_float(x['k1']), axis=1)

        beam_database[['bf', 'tf', 'tw', 'kdes', 'k1', 'd', 'W']] = beam_database[
            ['bf', 'tf', 'tw', 'kdes', 'k1', 'd', 'W']].astype('float')

        return beam_database


    def calculate_beam_demand(beam, config):
        """Calculate Beam Moment and Shear Values """
        """Shear and Moment Values are assigned to the Beam Object"""
        w_comp_beam_weight = beam.section_weight

        # Beam Demands
        # Construction Demands
        beam_trib_width = (min(beam.distance_to_left_beam, beam.distance_to_left_edge) + min(beam.distance_to_right_beam,
                                                                                            beam.distance_to_right_edge))

        # construction dead load is beam self weight + ponding + deck weight
        beam.w_cons_dl = w_comp_beam_weight + (config.deck_self_weight + config.q_ponding) * beam_trib_width

        beam.v_pre_comp = beam.w_cons_dl * beam.beam_span / 2 * 2

        beam.w_cons_ll = config.q_cons_ll * beam_trib_width

        beam.w_ultimate_cons = max(1.2 * beam.w_cons_dl + 1.6 * beam.w_cons_ll, 1.4 * beam.w_cons_dl)

        beam.v_ultimate_cons = beam.w_ultimate_cons / 1000 * beam.beam_span / 2

        beam.m_ultimate_cons_max = beam.w_ultimate_cons / 1000 * beam.beam_span ** 2 / 8 * 12  # kip-in

        # Post Composite Demands

        # total line dead load
        beam.w_dl = w_comp_beam_weight + (config.deck_self_weight + config.q_ponding + config.q_sdl) * beam_trib_width

        # live load
        beam.w_ll = config.q_ll * beam_trib_width

        # shear is calculated to be applied on to the girder
        beam.v_post_comp_ll = beam.w_ll / 1000 * beam.beam_span / 2

        beam.m_post_comp_ll_max = beam.w_ll / 1000 * beam.beam_span ** 2 / 8 * 12  # kip-in 
        
        # sdl
        beam.w_sdl = config.q_sdl * beam_trib_width

        beam.v_sdl = beam.w_sdl / 1000 * beam.beam_span / 2

        beam.m_sdl_max = beam.w_sdl / 1000 * beam.beam_span ** 2 / 8 * 12  # kip-in

        beam.v_selfw = beam.w_cons_dl / 1000 * beam.beam_span / 2

        beam.m_selfw_max = beam.w_cons_dl / 1000 * beam.beam_span ** 2 / 8 * 12  # kip-in

        beam.v_post_comp_dl = (beam.w_dl) / 1000 * beam.beam_span / 2

        beam.w_ultimate = max(1.2 * beam.w_dl + 1.6 * beam.w_ll, 1.4 * beam.w_dl)

        beam.m_ultimate_max = beam.w_ultimate / 1000 * beam.beam_span ** 2 / 8 * 12  # kip-in

        beam.v_ultimate_max = beam.w_ultimate / 1000 * beam.beam_span / 2  # kip

    def calculate_girder_demand(girder, beam, config):
        """ Calculate shear and moment diagrams for the girder object"""

        w_girder_weight = girder.section_weight

        def generate_shear_moment_diagram(applied_force, beam_span, beam_connection_points, beam_force_applied_point, w_self_weight):
            # applied force is multiplied by 2 to account for forces coming from two beams on either side of the girder.
            applied_force = 2 * applied_force

            x = [0]
            number_of_reactions_acting_on_the_girder = sum(beam_force_applied_point) 

            # i-end connection shear
            v_i = applied_force * number_of_reactions_acting_on_the_girder / 2 + (w_self_weight / 1000 * beam_span / 2)
            m_i = 0

            v = [v_i]
            m = [m_i]

            for i in range(1, len(beam_connection_points)):
                if beam_force_applied_point[i] == False:
                    temp_applied_force = 0
                else:
                    temp_applied_force = applied_force
                
                x_i = beam_connection_points[i]

                v_i = v_i - temp_applied_force - (w_self_weight / 1000) * (x_i - x[i - 1])

                v.append(v_i)

                m_i = m_i + v[i - 1] * (x_i - x[i - 1]) - (w_self_weight / 1000) * (x_i - x[i - 1]) ** 2 / 2

                m.append(m_i)

                x.append(x_i)

            return x, v, m

        #Construction Case
        
        girder.x, girder.v_ultimate_cons, girder.m_ultimate_cons = generate_shear_moment_diagram(beam.v_ultimate_cons, girder.beam_span,
                                                                                girder.beam_connection_point, girder.force_applied_point,
                                                                                w_girder_weight)

        _, girder.v_ultimate, girder.m_ultimate = generate_shear_moment_diagram(beam.v_ultimate_max, girder.beam_span,
                                                                                girder.beam_connection_point, girder.force_applied_point,
                                                                                w_girder_weight)

        _, girder.v_selfw, girder.m_selfw = generate_shear_moment_diagram(beam.v_selfw, girder.beam_span,
                                                                        girder.beam_connection_point, girder.force_applied_point, w_girder_weight)

        _, girder.v_post_comp_dl, girder.m_post_comp_dl = generate_shear_moment_diagram(beam.v_post_comp_dl,
                                                                                        girder.beam_span,
                                                                                        girder.beam_connection_point, 
                                                                                        girder.force_applied_point,
                                                                                        w_girder_weight)

        _, girder.v_post_comp_ll, girder.m_post_comp_ll = generate_shear_moment_diagram(beam.v_post_comp_ll,
                                                                                            girder.beam_span,
                                                                                            girder.beam_connection_point, 
                                                                                            girder.force_applied_point, 0)

        _, girder.v_sdl, girder.m_sdl = generate_shear_moment_diagram(beam.v_sdl, girder.beam_span, girder.beam_connection_point, girder.force_applied_point, 0)

        # next three are used for bottom flange stress check
        girder.m_selfw_max = max(girder.m_selfw) * 12  # kip-in

        girder.m_sdl_max = max(girder.m_sdl) * 12  # kip-in
        
        girder.m_post_comp_ll_max = max(girder.m_post_comp_ll) * 12  # kip-in


        girder.m_ultimate_cons_max = max(girder.m_ultimate_cons) * 12  # kip-in

        girder.m_ultimate_max = max(girder.m_ultimate) * 12  # kip-in
        girder.v_ultimate_max = max(girder.v_ultimate)  # kip



    # Pre-composite Strength Check for construction case
    def calculate_noncomposite_flexure(beam):
        # beam_span in ft, converted to inches.

        l_b = beam.unbraced_length

        # calculate l_p
        l_p = 1.76 * beam.r_y * (beam.steel_modulus_of_elasticity / beam.steel_yield_stress) ** 0.5

        # calculate l_r
        r_ts = ((beam.i_y * beam.c_w) ** 0.5 / beam.s_x) ** 0.5
        h_0 = beam.section_depth - beam.section_flange_thickness

        # c_b values from table 3_1
        # no intermediate bracing
        c_b = 1.14

        l_r = 1.95 * r_ts * beam.steel_modulus_of_elasticity / (0.7 * beam.steel_yield_stress) * (
                    (beam.j_c / (beam.s_x * h_0)) + (beam.j_c / (beam.s_x * h_0)) ** 2 + 6.76 * (
                        0.7 * beam.steel_yield_stress / beam.steel_modulus_of_elasticity) ** 2) ** 0.5

        m_p = beam.plastic_modulus * beam.steel_yield_stress

        if l_b <= l_p:
            m_n = m_p
        elif l_b <= l_r:
            # equation f2-2
            m_n = c_b * (m_p - (m_p - 0.7 * beam.steel_yield_stress * beam.s_x) * (l_b - l_p) / (l_r - l_p))

        else:
            # equation f2-4
            f_cr = c_b * (math.pi) ** 2 * beam.steel_modulus_of_elasticity / (l_b / r_ts) ** 2 * (
                        1 + 0.078 * beam.j_c / (beam.s_x * h_0) * (l_b / r_ts) ** 2) ** 0.5

            # equation f2-3
            m_n = f_cr * beam.s_x

        if m_n > m_p:
            m_n = m_p

        phi_m_n = m_n * 0.9

        return phi_m_n

    # Calculate Composite Bending Strength
    # I3.2a - Positive Flexural Strength

    def calculate_composite_beam_flexural_strength(composite_beam, config, percent_composite=100, beam_or_girder="beam",
                                                phi=0.90):
        # determine the location of Plastic Neutral Axis

        concrete_thickness = config.deck_library[config.deck_type]["concrete_thickness"]

        deck_depth = config.deck_library[config.deck_type]["deck_depth"]

        y_con = concrete_thickness + deck_depth  # distance from top of concrete to top of flange

        concrete_type = config.deck_library[config.deck_type]["concrete_type"]

        concrete_strength = config.concrete_library[config.concrete]["concrete_strength"]

        if concrete_type == "NW":
            concrete_weight = 150  # pcf
        else:
            concrete_weight = 110  # pcf

        config.concrete_weight = concrete_weight

        # first calculate 100% composite capacity:

        def force_in_concrete_fully_composite():

            # 0.85f'cAc all concrete in compression

            v_prime_conc = 0.85 * concrete_strength * composite_beam.effective_width * 12 * concrete_thickness

            # full steel shape in tension

            v_prime_steel = composite_beam.section_area * composite_beam.steel_yield_stress

            # use minimum of two:

            v_prime = min(v_prime_conc, v_prime_steel)

            return v_prime

        # now let's calculate the number of studs required for full composite

        def calc_concrete_modulus_of_elasticity(concrete_strength, concrete_weight):
            return (concrete_weight ** 1.5) * (concrete_strength ** 0.5)

        concrete_modulus_of_elasticity = calc_concrete_modulus_of_elasticity(concrete_strength, concrete_weight)

        config.concrete_modulus_of_elasticity = concrete_modulus_of_elasticity

        # Stud Strength

        def calculate_individual_stud_strength(stud_diameter, stud_fu, concrete_modulus_of_elasticity, concrete_strength,
                                            number_of_studs_per_rib=1, deck_orientation="perpendicular"):

            stud_area = math.pi * stud_diameter ** 2 / 4

            # for now assume r_g = 1
            r_g = 1

            r_p = 1

            individual_stud_strength = 0.5 * stud_area * (concrete_strength * 1000 * concrete_modulus_of_elasticity) ** 0.5

            if individual_stud_strength > r_g * r_p * stud_area * stud_fu:
                individual_stud_strength = r_g * r_p * stud_area * stud_fu

            return individual_stud_strength

        individual_stud_strength = calculate_individual_stud_strength(config.stud_diameter, config.stud_fu,
                                                                    concrete_modulus_of_elasticity, concrete_strength,
                                                                    number_of_studs_per_rib=1,
                                                                    deck_orientation="perpendicular")

        max_composite_stud_strength_required = force_in_concrete_fully_composite()

        total_stud_strength = percent_composite / 100 * max_composite_stud_strength_required

        composite_beam.total_stud_strength = total_stud_strength

        # if it is a beam - deck is perpendicular to the beam - studs at flutes
        if beam_or_girder == 'beam':
            # For beam spans <= 30': studs max @ 24"
            # For beam spans > 30': studs max @ 12"

            if composite_beam.beam_span <= 30:
                max_stud_spacing = 24
            else:
                max_stud_spacing = 12

                # Maximum rows of studs allowed:    3
            # Min. nominal beam width for 2 stud rows:    5 ½”
            # Min. nominal beam width for 3 stud rows:    8 ½”

            if composite_beam.section_flange_width < config.min_width_two_studs:
                max_studs_per_row = 1
            elif composite_beam.section_flange_width < config.min_width_three_studs:
                max_studs_per_row = 2
            else:
                max_studs_per_row = 3

            number_of_studs_needed = 2 * math.ceil(total_stud_strength / (0.75 * individual_stud_strength))

            number_of_studs_per_flute = math.ceil(
                number_of_studs_needed / (composite_beam.beam_span * 12 / max_stud_spacing))

            if number_of_studs_per_flute <= 1:
                studs_per_flute = 1
                stud_spacing = max_stud_spacing
                composite_beam.studs_adequate = True
            elif number_of_studs_per_flute <= max_studs_per_row:
                if max_stud_spacing == 24:  # can we reduce spacing to 12 to fit more studs
                    temp_number_of_studs_per_flute = math.ceil(number_of_studs_needed / (
                                composite_beam.beam_span * 12 / 12))  # recalculate number of studs per flute with 12" spacing
                    if (temp_number_of_studs_per_flute * (composite_beam.beam_span * 12 / 12)) > (
                            1.25 * number_of_studs_needed):
                        stud_spacing = max_stud_spacing
                        composite_beam.studs_adequate = True
                    else:
                        number_of_studs_per_flute = temp_number_of_studs_per_flute
                        stud_spacing = 12
                        composite_beam.studs_adequate = True
                else:
                    stud_spacing = max_stud_spacing
                    composite_beam.studs_adequate = True
            else:
                number_of_studs_per_flute = 99
                stud_spacing = 1
                composite_beam.studs_adequate = False

            composite_beam.total_number_of_studs = int(
                math.ceil(number_of_studs_per_flute * composite_beam.beam_span * 12 / stud_spacing))


        # if it is a girder - deck is horizontal to the beam - single line of studs
        else:
            number_of_studs_needed = 2 * math.ceil(total_stud_strength / (0.75 * individual_stud_strength))

            stud_spacing = math.ceil(composite_beam.beam_span * 12 / number_of_studs_needed)  # inches

            max_stud_spacing = 12
            min_stud_spacing = 4.5
            composite_beam.studs_adequate = True
            number_of_studs_per_flute = number_of_studs_needed

            if stud_spacing > max_stud_spacing:
                stud_spacing = max_stud_spacing

            elif stud_spacing < min_stud_spacing:
                stud_spacing = 1
                number_of_studs_per_flute = 99
                composite_beam.studs_adequate = False

            composite_beam.total_number_of_studs = int(math.ceil(composite_beam.beam_span * 12 / stud_spacing))

        def calculate_tension_compression_forces(neutral_axis_depth, calculate_moment=False):

            # assume a compression depth
            concrete_compression_force = total_stud_strength
            steel_compression_depth = neutral_axis_depth - y_con

            if neutral_axis_depth <= y_con:
                # if neutral axis is above top of flange
                # a is the whitney stress block depth

                total_compression_force = concrete_compression_force

                steel_top_flange_compression_force = 0
                steel_web_compression_force = 0

                steel_top_flange_tension_area = composite_beam.section_flange_width * composite_beam.section_flange_thickness
                steel_top_flange_tension_force = steel_top_flange_tension_area * composite_beam.steel_yield_stress

                steel_web_tension_area = composite_beam.section_web_thickness * (
                            composite_beam.section_depth - 2 * composite_beam.section_flange_thickness)
                steel_web_tension_force = steel_web_tension_area * composite_beam.steel_yield_stress

                steel_bottom_flange_tension_area = composite_beam.section_flange_width * composite_beam.section_flange_thickness
                steel_bottom_flange_tension_force = steel_bottom_flange_tension_area * composite_beam.steel_yield_stress

                total_tension_force = steel_top_flange_tension_force + steel_web_tension_force + steel_bottom_flange_tension_force

            elif neutral_axis_depth <= y_con + composite_beam.section_flange_thickness:

                # neutral axis at top flange

                steel_top_flange_compression_area = composite_beam.section_flange_width * steel_compression_depth
                steel_top_flange_compression_force = steel_top_flange_compression_area * composite_beam.steel_yield_stress

                steel_web_compression_force = 0

                total_compression_force = concrete_compression_force + steel_top_flange_compression_force + steel_web_compression_force

                steel_top_flange_tension_area = composite_beam.section_flange_width * (
                            composite_beam.section_flange_thickness - steel_compression_depth)
                steel_top_flange_tension_force = steel_top_flange_tension_area * composite_beam.steel_yield_stress

                steel_web_tension_area = composite_beam.section_web_thickness * (
                            composite_beam.section_depth - 2 * composite_beam.section_flange_thickness)
                steel_web_tension_force = steel_web_tension_area * composite_beam.steel_yield_stress

                steel_bottom_flange_tension_area = composite_beam.section_flange_width * composite_beam.section_flange_thickness
                steel_bottom_flange_tension_force = steel_bottom_flange_tension_area * composite_beam.steel_yield_stress

                total_tension_force = steel_top_flange_tension_force + steel_web_tension_force + steel_bottom_flange_tension_force

            else:

                # neutral axis at k zone
                steel_top_flange_compression_area = composite_beam.section_flange_width * composite_beam.section_flange_thickness
                steel_top_flange_compression_force = steel_top_flange_compression_area * composite_beam.steel_yield_stress

                steel_web_compression_area = composite_beam.section_web_thickness * (
                            steel_compression_depth - composite_beam.section_flange_thickness)
                steel_web_compression_force = steel_web_compression_area * composite_beam.steel_yield_stress

                total_compression_force = concrete_compression_force + steel_top_flange_compression_force + steel_web_compression_force

                steel_top_flange_tension_force = 0

                steel_web_tension_area = composite_beam.section_web_thickness * (
                            composite_beam.section_depth - composite_beam.section_flange_thickness - steel_compression_depth)
                steel_web_tension_force = steel_web_tension_area * composite_beam.steel_yield_stress

                steel_bottom_flange_tension_area = composite_beam.section_flange_width * composite_beam.section_flange_thickness
                steel_bottom_flange_tension_force = steel_bottom_flange_tension_area * composite_beam.steel_yield_stress

                total_tension_force = steel_top_flange_tension_force + steel_web_tension_force + steel_bottom_flange_tension_force

            if calculate_moment:

                effective_concrete_depth = concrete_compression_force / (
                            composite_beam.effective_width * 0.85 * concrete_strength)
                moment_concrete = concrete_compression_force * neutral_axis_depth - effective_concrete_depth / 2

                if steel_top_flange_tension_force == 0:
                    moment_top_flange_compression = steel_top_flange_compression_force * (
                                steel_compression_depth - composite_beam.section_flange_thickness / 2)
                else:
                    moment_top_flange_compression = steel_top_flange_compression_force * steel_compression_depth / 2

                moment_web_compression = steel_web_compression_force * (
                            composite_beam.section_depth / 2 - steel_compression_depth)

                moment_top_flange_tension = steel_top_flange_tension_force * (
                            composite_beam.section_flange_thickness - steel_compression_depth) / 2
                moment_web_tension = steel_web_tension_force * (composite_beam.section_depth / 2 - steel_compression_depth)
                moment_bottom_flange_tension = steel_bottom_flange_tension_force * (
                            composite_beam.section_depth - composite_beam.section_flange_thickness / 2 - steel_compression_depth)

                total_moment = moment_concrete + moment_top_flange_compression + moment_web_compression + moment_top_flange_tension + moment_web_tension + moment_bottom_flange_tension

            else:
                total_moment = 0

            return (total_compression_force, total_tension_force, total_moment)

        def check_equilibrium(neutral_axis_depth):
            (compression_force, tension_force, moment) = calculate_tension_compression_forces(neutral_axis_depth)
            delta_compression_tension_force = compression_force - tension_force

            return delta_compression_tension_force

        neutral_axis_depth = fsolve(check_equilibrium, y_con, xtol=0.001)[0]

        neutral_axis_depth = neutral_axis_depth.round(2)

        # Calculate moment capacity based on neutral axis
        compression_force, tension_force, moment = calculate_tension_compression_forces(neutral_axis_depth, True)

        composite_beam.neutral_axis_depth = neutral_axis_depth

        composite_beam.post_comp_phi_m_n = 0.9 * moment

        composite_beam.post_comp_phi_m_n = composite_beam.post_comp_phi_m_n.round(0)

        # Calculate Lower Bound Moment of Inertia

        def calculate_lower_bound_moment_of_inertia():
            # Based on Equation C-I3-1

            # a_s = cross section area of the steel section
            a_s = composite_beam.section_area

            # d_1 = distance from the compression force in the concrete to the top of the steel section
            # d_3 = distance from the resultant steel tension force for full section tension yield to the top of the steel

            if neutral_axis_depth <= concrete_thickness:
                conc_depth = neutral_axis_depth

                d_1 = y_con - neutral_axis_depth / 2
                d_3 = composite_beam.section_depth / 2

            elif neutral_axis_depth <= y_con:
                # to be fixed to add rib concrete
                conc_depth = concrete_thickness

                d_1 = y_con - concrete_thickness / 2
                d_3 = composite_beam.section_depth / 2
            else:
                # to be fixed to add rib concrete
                conc_depth = concrete_thickness

                d_1 = y_con - concrete_thickness / 2
                d_3 = composite_beam.section_depth / 2

            y_ena = (composite_beam.section_area * d_3 + (total_stud_strength / composite_beam.steel_yield_stress) * (
                        2 * d_3 + d_1)) / (
                                composite_beam.section_area + total_stud_strength / composite_beam.steel_yield_stress)

            composite_beam.i_lower_bound = composite_beam.moment_of_inertia + composite_beam.section_area * (
                        y_ena - d_3) ** 2 + (total_stud_strength / composite_beam.steel_yield_stress) * (
                                                    2 * d_3 + d_1 - y_ena) ** 2

            composite_beam.i_lower_bound = int(composite_beam.i_lower_bound)

            #calculate effective section modulus 

            concrete_to_steel_ratio = concrete_modulus_of_elasticity / composite_beam.steel_modulus_of_elasticity

            full_composite_area = composite_beam.section_area + composite_beam.effective_width * conc_depth * concrete_to_steel_ratio

            transformed_section_y_bar = (composite_beam.section_area * (y_con + composite_beam.section_depth/2) + composite_beam.effective_width * conc_depth * concrete_to_steel_ratio * (conc_depth/2)) / full_composite_area
                
            transformed_moment_of_inertia = composite_beam.moment_of_inertia + composite_beam.section_area * (y_con + composite_beam.section_depth/2 - transformed_section_y_bar)**2 
            + composite_beam.effective_width * conc_depth * concrete_to_steel_ratio * (transformed_section_y_bar - conc_depth/2)**2

            section_modulus_transformed_section = transformed_moment_of_inertia / (y_con + composite_beam.section_depth - transformed_section_y_bar)

            #Equation C-I3-5

            composite_beam.effective_section_modulus = composite_beam.s_x + (total_stud_strength/max_composite_stud_strength_required)**0.5 * (section_modulus_transformed_section - composite_beam.s_x) 

        calculate_lower_bound_moment_of_inertia()

        # bottom flange stress check

        # equation is M_dead / S_steel + (M_sdl + M_live) / S_effective < 50 ksi 

        composite_beam.bottom_flange_stress = composite_beam.m_selfw_max / composite_beam.s_x + (composite_beam.m_sdl_max + composite_beam.m_post_comp_ll_max) / composite_beam.effective_section_modulus


    def calculate_deformations(beam_or_girder, element, config):
        if beam_or_girder == 'beam':

            element.defl_self_weight = 5 / 384 * ((element.w_cons_dl / 1000 / 12) * (element.beam_span * 12) ** 4) / (
                        element.steel_modulus_of_elasticity * element.moment_of_inertia)

            element.defl_dl = 5 / 384 * ((element.w_dl / 1000 / 12) * (element.beam_span * 12) ** 4) / (
                        element.steel_modulus_of_elasticity * element.i_lower_bound)

            element.defl_sdl = 5 / 384 * ((element.w_sdl / 1000 / 12) * (element.beam_span * 12) ** 4) / (
                        element.steel_modulus_of_elasticity * element.i_lower_bound)

            element.defl_ll = 5 / 384 * ((element.w_ll / 1000 / 12) * (element.beam_span * 12) ** 4) / (
                        element.steel_modulus_of_elasticity * element.i_lower_bound)

        elif beam_or_girder == 'girder':

            def deformation_using_moment_area_method(m, element, moment_of_inertia):
                """Calculate mid-span deformation based on moment area method"""

                defl_i = 0
                x = element.x

                for i in range(1, int((len(element.beam_connection_point) - 1) / 2) + 1):
                    defl_i += m[i - 1] * (x[i] - x[i - 1]) * (element.beam_span / 2 - x[i - 1] - (x[i] - x[i - 1]) / 2) + (
                                m[i] - m[i - 1]) * (x[i] - x[i - 1]) / 2 * (
                                        element.beam_span / 2 - x[i - 1] + (x[i] - x[i - 1]) * 2 / 3)

                defl_i = defl_i * (12 ** 3) / (element.steel_modulus_of_elasticity * moment_of_inertia)

                return defl_i

            element.defl_self_weight = deformation_using_moment_area_method(element.m_selfw, element,
                                                                            element.moment_of_inertia)

            element.defl_dl = deformation_using_moment_area_method(element.m_post_comp_dl, element, element.i_lower_bound)

            element.defl_sdl = deformation_using_moment_area_method(element.m_sdl, element, element.i_lower_bound)

            element.defl_ll = deformation_using_moment_area_method(element.m_post_comp_ll, element, element.i_lower_bound)

        max_camber = 0.75 * element.defl_self_weight
        min_camber = 0.75  # in

        # Maximum camber: Consult project design manager if camber exceeds L/180 or 3”.
        max_camber_limit = min(config.max_camber_allowed, element.beam_span * 12 / 180)

        max_camber = min(math.floor(max_camber * 4) / 4, max_camber_limit)

        if element.deflection_critical == True:
            max_defl_dl_ll_camber = element.beam_span * 12 / 480
        else:
            max_defl_dl_ll_camber = element.beam_span * 12 / 240

        max_defl_ll = element.beam_span * 12 / 360

        if element.section_depth < 14 or element.beam_span < 24:
            max_camber = 0

        defl_dead_plus_live = element.defl_self_weight + element.defl_sdl + element.defl_ll

        if (defl_dead_plus_live < max_defl_dl_ll_camber) and (element.defl_ll < max_defl_ll):
            # Camber not required
            element.camber = 0.0
            element.cambered = False
            element.deflection_check_pass = True
        elif (defl_dead_plus_live - min_camber < max_defl_dl_ll_camber) and (element.defl_ll < max_defl_ll):
            # Use minimum camber
            element.camber = min_camber
            element.cambered = True
            element.deflection_check_pass = True
        elif (defl_dead_plus_live - max_camber < max_defl_dl_ll_camber) and (element.defl_ll < max_defl_ll):
            # Camber required
            camber = defl_dead_plus_live - max_defl_dl_ll_camber
            camber = min(max_camber, math.ceil(camber * 4) / 4)

            if (defl_dead_plus_live - camber < max_defl_dl_ll_camber):
                    element.camber = camber
                    element.cambered = True
                    element.deflection_check_pass = True
            else:
                camber -= 0.25

                if (defl_dead_plus_live - camber < max_defl_dl_ll_camber):
                    element.camber = camber
                    element.cambered = True
                    element.deflection_check_pass = True
                    
                else:                   
                    element.deflection_check_pass = False
                    element.camber = 0.0
                    element.cambered = False    
        else:
            element.deflection_check_pass = False
            element.camber = 0.0
            element.cambered = False

        # Total Deflection
        element.defl_dl_ll_camber = round(defl_dead_plus_live - element.camber, 2)
        
        # Total Deflection divided by length
        if element.defl_dl_ll_camber==0:
            element.l_divided_defl_dl_ll_camber = "L/inf"    
        else:
            element.l_divided_defl_dl_ll_camber = "L/"+str(int(element.beam_span * 12 / element.defl_dl_ll_camber))

        # Live Deflection
        element.defl_ll = round(element.defl_ll, 2)
    
        # Live Deflection divided by length
        if element.defl_ll==0:
            element.l_divided_defl_ll = "L/inf" 
        else:    
            element.l_divided_defl_ll = "L/"+str(int(element.beam_span * 12 / element.defl_ll)) 

    def assign_pre_comp_capacity_to_dataframe(row, beam):
        beam.section_size = row['AISC_Manual_Label']
        beam.get_section_properties()
            
        pre_comp_moment_capacity = calculate_noncomposite_flexure(beam)
        return pre_comp_moment_capacity


    def generate_composite_beam_dataframe(beam, beam_database, config, framing_beam="", beam_or_girder_input = "beam"):
        # Iterate over composite percentages
        
        df_design = pd.DataFrame(columns=['AISC_Manual_Label', 'W', 'A', 'd', 'bf', 'Ix', 'pre_comp_mu', 'pre_comp_phi_mn', 'pre_comp_dcr', 'composite_percent', 'post_comp_mu', 'post_comp_phi_mn', 'post_comp_dcr', 'post_comp_i_lower_bound', 
                                        'bottom_flange_stress','defl_dl_ll_camber', 'l_divided_defl_dl_ll_camber', 'defl_ll', 'l_divided_defl_ll', 'camber', 'cambered', 'deflection_check', 'number_of_studs', 'studs_adequate', 'total_stud_strength', 'neutral_axis_depth'])

        # Composite Requirements For beam spans <= 30':    25% min composite action OR studs @ 24" 
        # Composite Requirements For beam spans > 30': 50% min composite action OR studs @ 12"  

        if beam.beam_span <= 30:
            comp_percent_levels = [25, 50, 60, 70, 80, 90, 100]
        else:
            comp_percent_levels = [50, 60, 70, 80, 90, 100]
        

        def assign_post_comp_capacity_to_dataframe(row, beam, comp_percent, framing_beam, beam_or_girder):
            beam.section_size = row['AISC_Manual_Label']
            beam.get_section_properties()

            if beam_or_girder == "beam":
                calculate_beam_demand(beam, config)
            else:
                calculate_girder_demand(beam, framing_beam, config)

            # calculate composite moment capacity and bottom flange stress
            calculate_composite_beam_flexural_strength(beam, config, comp_percent, beam_or_girder)
        
            calculate_deformations(beam_or_girder, beam, config)
       
            return round(beam.m_ultimate_cons_max,0), round(beam.m_ultimate_max,0), beam.post_comp_phi_m_n, int(beam.bottom_flange_stress), beam.i_lower_bound, beam.defl_dl_ll_camber,  beam.l_divided_defl_dl_ll_camber, beam.defl_ll, beam.l_divided_defl_ll, beam.camber, beam.cambered, beam.deflection_check_pass, beam.total_number_of_studs, beam.studs_adequate, beam.total_stud_strength, beam.neutral_axis_depth, beam.v_ultimate_max

        for comp_percent in comp_percent_levels:
        
            temp_database = beam_database[['AISC_Manual_Label', 'W', 'A', 'd', 'bf', 'Ix', 'pre_comp_phi_mn', 'pre_comp_dcr']].copy()
        
            temp_database['composite_percent'] = comp_percent
      
            temp_database[['pre_comp_mu', 'post_comp_mu', 'post_comp_phi_mn', 'bottom_flange_stress', 'post_comp_i_lower_bound', 'defl_dl_ll_camber', 'l_divided_defl_dl_ll_camber', 'defl_ll', 'l_divided_defl_ll', 'camber', 'cambered', 'deflection_check','number_of_studs', 'studs_adequate', 'total_stud_strength', 'neutral_axis_depth', 'vu_shear_at_bg_connection']] = temp_database.apply(lambda x: assign_post_comp_capacity_to_dataframe(x, beam, comp_percent, framing_beam, beam_or_girder_input), axis = 1, result_type="expand")
           
            temp_database = temp_database.loc[temp_database['deflection_check']] 
        
            # check whether post composite moment is less than post composite capacity

            temp_database = temp_database.loc[temp_database['post_comp_phi_mn'] > temp_database['post_comp_mu']]
            # check whether bottom flange stress is less than yield
            temp_database = temp_database.loc[temp_database['bottom_flange_stress'] < beam.steel_yield_stress]
           
            temp_database['post_comp_dcr'] = round(temp_database['post_comp_mu']/temp_database['post_comp_phi_mn'],2) 
        
            temp_database = temp_database.loc[temp_database['studs_adequate']]
        
            temp_database['Stud Cost'] = temp_database['number_of_studs'] * config.cost_per_stud
            
            temp_database["Steel Weight"] = temp_database['W'] * beam.beam_span
        
            temp_database['Steel Cost'] = temp_database['W'] * beam.beam_span * config.steel_cost
        
            temp_database['Total Cost'] = temp_database['Stud Cost'] + temp_database['Steel Cost']

            df_design = pd.concat([df_design, temp_database])
       
        df_design[['W','d','composite_percent','camber']] = df_design[['W','d','composite_percent','camber']].astype(float)

        df_design['Section Family'] = df_design['AISC_Manual_Label'].str[:3]
        
        df_design_with_camber = df_design.loc[df_design['cambered']]
        
        df_design_no_camber = df_design.loc[df_design['cambered']==False]

        df_filtered_with_camber = df_design_with_camber.groupby('AISC_Manual_Label', group_keys=False).apply(lambda x: x.nsmallest(1, 'composite_percent', keep='all'))
        
        df_filtered_with_camber = df_filtered_with_camber.groupby('Section Family', group_keys=False).apply(lambda x: x.nsmallest(1, 'W', keep='all'))

        if len(df_filtered_with_camber) > 0:
            df_filtered_with_camber.reset_index(inplace=True)

        df_filtered_no_camber = df_design_no_camber.groupby('AISC_Manual_Label', group_keys=False).apply(lambda x: x.nsmallest(1, 'composite_percent', keep='all'))
        df_filtered_no_camber = df_filtered_no_camber.groupby('Section Family', group_keys=False).apply(lambda x: x.nsmallest(1, 'W', keep='all'))

        if len(df_filtered_no_camber) > 0:
            df_filtered_no_camber.reset_index(inplace=True)
        
        # concat two dataframes as the final design output

        df_design = pd.concat([df_filtered_no_camber, df_filtered_with_camber], ignore_index=True)
        
        columns_to_keep = ["AISC_Manual_Label", "W", "A", "d", "bf", "Ix", "pre_comp_mu", "pre_comp_phi_mn", "pre_comp_dcr", "composite_percent", "post_comp_mu", "post_comp_phi_mn" , 
                        "post_comp_dcr", "bottom_flange_stress", "post_comp_i_lower_bound", "defl_dl_ll_camber", "l_divided_defl_dl_ll_camber", "defl_ll", "l_divided_defl_ll" , "camber" , "cambered", 
                        "deflection_check", "number_of_studs", 'vu_shear_at_bg_connection', "studs_adequate", "total_stud_strength", "neutral_axis_depth", "Stud Cost", "Steel Cost", "Total Cost", "Section Family", "Steel Weight"]
    
        df_design.reset_index(inplace=True)
        df_design = df_design[columns_to_keep]
        
        df_design = df_design.sort_values(by=['W'], ascending = True)
        df_design.rename(columns={'AISC_Manual_Label':'Section Size'}, inplace= True)
        

        df_design.reset_index(inplace=True)
        
        df_design.drop(columns='index', inplace = True)
        
        return df_design


    def floor_vibration_check(girder, beam, config):
        
        damping = config.damping
        
        concrete_modulus_of_elasticity = config.concrete_modulus_of_elasticity
        
        steel_modulus_of_elasticity = config.steel_modulus_of_elasticity
        
        n = steel_modulus_of_elasticity / (1.35 * concrete_modulus_of_elasticity)
        
        concrete_thickness = config.deck_library[config.deck_type]["concrete_thickness"]
        
        deck_depth = config.deck_library[config.deck_type]["deck_depth"]
        
            
        # Calculate Beam Frequency
            
        beam_effective_width = min(0.4*beam.beam_span, beam.beam_spacing)
            
        beam_transformed_effective_width = beam_effective_width * 12 / n
            
        beam_concrete_area = beam_transformed_effective_width * concrete_thickness
            
        y_bar = beam_concrete_area * (beam.section_depth / 2 + deck_depth + concrete_thickness / 2) / (beam_concrete_area + beam.section_area)
            
        beam_transformed_moment_of_inertia = beam_transformed_effective_width * concrete_thickness**3 / 12 + beam_concrete_area * (beam.section_depth / 2 + deck_depth + concrete_thickness / 2 - y_bar)**2 + beam.moment_of_inertia + beam.section_area * y_bar**2
        
        w_beam_line_load = (config.q_vib_ll + config.q_vib_sdl + config.deck_self_weight) * beam.beam_spacing + beam.section_weight
                
        w_girder_line_load = w_beam_line_load / beam.beam_spacing * beam.beam_span
                        
        deflection_j = 5/384 * (w_beam_line_load * beam.beam_span**4) / (steel_modulus_of_elasticity * beam_transformed_moment_of_inertia) * (12**3 / 1000)
            
        frequency_i = 0.18 * (config.g / deflection_j)**0.5
            
        # transformed slab moment of inertia per unit width in slab span direction
        d_e = config.deck_self_weight / config.concrete_weight * 12
            
        d_s = 12 * d_e**3 / (12 * n)
                    
        # effective beam panel width
        d_j = beam_transformed_moment_of_inertia / (beam.beam_spacing)
        
        beam_effective_panel_width = config.c_j * (d_s / d_j)**0.25 * beam.beam_span # b_j
        if beam_effective_panel_width > (2/3 * config.floor_width):
            beam_effective_panel_width = 2/3 * config.floor_width
                        
        beam_effective_panel_weight = 1.5 * (w_beam_line_load / beam.beam_spacing) * beam_effective_panel_width * beam.beam_span 
            
        # Calculate Girder Frequency
            
        girder_effective_width = min(0.4*girder.beam_span, beam.beam_span)
            
        girder_transformed_effective_width = girder_effective_width * 12 / n
            
        girder_concrete_area_slab = girder_transformed_effective_width * concrete_thickness
            
        girder_concrete_area_flute = girder_transformed_effective_width / 2 * deck_depth
        
            
        y_bar = (girder_concrete_area_slab*(girder.section_depth/2 + deck_depth + concrete_thickness/2) + girder_concrete_area_flute*(girder.section_depth/2 + deck_depth/2)) / (girder_concrete_area_slab + girder_concrete_area_flute + girder.section_area)
            
        girder_transformed_moment_of_inertia = girder_transformed_effective_width*concrete_thickness**3/12 + girder_concrete_area_slab*(girder.section_depth/2+deck_depth+concrete_thickness/2-y_bar)**2 + girder_transformed_effective_width/2 * deck_depth**3 / 12 + girder_concrete_area_flute * (girder.section_depth/2 + deck_depth/2 - y_bar)**2 + girder.moment_of_inertia + girder.section_area * y_bar**2
                
        w_girder_line_load = w_beam_line_load / beam.beam_spacing * beam.beam_span + girder.section_weight
                        
        deflection_g = 5/384 * (w_girder_line_load * girder.beam_span**4) / (steel_modulus_of_elasticity * girder_transformed_moment_of_inertia) * (12**3 / 1000)
            
        frequency_g = 0.18 * (config.g / deflection_g)**0.5
                        
        # effective girder panel width
        d_g = girder_transformed_moment_of_inertia / (beam.beam_span)
        
        girder_effective_panel_width = config.c_g * (d_j / d_g)**0.25 * girder.beam_span # b_g
        if girder_effective_panel_width > (2/3 * config.floor_width):
            girder_effective_panel_width = 2/3 * config.floor_width
                        
        girder_effective_panel_weight = (w_girder_line_load / beam.beam_span) * girder_effective_panel_width * girder.beam_span 
            
            
        # Combined Mode Properties
        # Floor fundamental frequency
        
        f_n = 0.18 * (config.g / (deflection_j + deflection_g))**0.5 # Hz
        
        # if girder span < 30 ft, deflection_g is reduced
        if girder.beam_span < 30:
            deflection_g = girder.beam_span / beam_effective_panel_width * deflection_g
        
        
        # Equivalent panel model panel weight
        effective_panel_weight = deflection_j / (deflection_j + deflection_g) * beam_effective_panel_weight + deflection_g / (deflection_j + deflection_g) * girder_effective_panel_weight 
        
        # Evaluation
        a_p = config.p_0 * math.exp(-0.35*f_n) / (config.damping * effective_panel_weight)
        
        return a_p


    config = Config()

    config.girder_span = girder_span
    config.beam_span = beam_span
    config.deck_type = deck_type
    config.q_sdl = q_sdl
    config.q_ll = q_ll
    config.acceleration_limit = acc_limit
    config.damping = damping
    config.aisc_database_filename = aisc_database_filename
    config.concrete = concrete_type
    
    config.get_deck_properties()
    beam_database = read_aisc_database(config)

    if config.deflection_critical:
        dl_ll_camber_limit = "L/480"
    else:
        dl_ll_camber_limit = "L/240"

    # Instentiate comp_beam object
    comp_beam = Composite_Beam("W12X19", beam_database)
    comp_beam.beam_span = config.beam_span
    comp_beam.deflection_critical = config.deflection_critical

    max_deck_span = min(float(config.deck_max_span), max_beam_spacing)

    number_of_deck_spans = math.ceil(config.girder_span / max_deck_span)
    number_of_beams = number_of_deck_spans - 1 # number of beams supported by the girder

    comp_beam.beam_spacing = config.girder_span / number_of_deck_spans

    config.beam_spacing = comp_beam.beam_spacing

    comp_beam.distance_to_left_beam = comp_beam.beam_spacing / 2
    comp_beam.distance_to_right_beam = comp_beam.beam_spacing / 2
    comp_beam.calculate_effective_width()

    # remove beams deeper than max depth
    beam_database_for_calc = beam_database.loc[beam_database['d'] <= max_beam_depth]


    # Filter out sections that do not satisfy precomposite serviceability check
    calculate_beam_demand(comp_beam, config)

    # deflection check, calculate minimum I required for L/240 limit
    minimum_i_for_construction_case = 5/384 * ((comp_beam.w_cons_dl/1000/12) * (comp_beam.beam_span*12)**3) / (comp_beam.steel_modulus_of_elasticity) * 240

    beam_database_for_calc = beam_database_for_calc.loc[beam_database['Ix'] > minimum_i_for_construction_case]

    # if no results are available send it back
    if beam_database_for_calc.size == 0:
        output_message = "Typical Wide-Flange Sections do not work for the given design parameters."
        return "", output_message

    # Filter out sections that do not satisfy precomposite strength check

    beam_database_for_calc['pre_comp_mu'] = comp_beam.m_ultimate_cons_max

    beam_database_for_calc['pre_comp_phi_mn'] = beam_database_for_calc.apply(lambda x: assign_pre_comp_capacity_to_dataframe(x, comp_beam), axis = 1)

    beam_database_for_calc['pre_comp_dcr'] = round(comp_beam.m_ultimate_cons_max /beam_database_for_calc['pre_comp_phi_mn'],2)

    beam_database_for_calc = beam_database_for_calc.loc[beam_database_for_calc['pre_comp_phi_mn'] > comp_beam.m_ultimate_cons_max]

    # if no results are available send it back
    if beam_database_for_calc.size == 0:
        output_message = "Typical Wide-Flange Sections do not work for the given design parameters."
        return "", output_message

    beam_database_for_calc = beam_database_for_calc.loc[beam_database_for_calc['pre_comp_dcr'] > 0.05]
        
    df_beam_design = generate_composite_beam_dataframe(comp_beam, beam_database_for_calc, config)

    # use first 5 results
    df_beam_design = df_beam_design.nsmallest(3, 'Total Cost', keep='all')

    # df_beam_design is used for the detailed beam design calc output    

    girder = Composite_Beam("W30X99", beam_database)
    girder.get_section_properties()

    girder.beam_span = config.girder_span
    girder.beam_connection_point = np.linspace(0, config.girder_span, number_of_deck_spans+1)

    girder.force_applied_point = [True] * len(girder.beam_connection_point)
    girder.force_applied_point[0] = False
    girder.force_applied_point[-1] = False

    # create a midspan point if the list does not include it
    if number_of_deck_spans % 2!=0:
        temp_connection_point_list = list(girder.beam_connection_point)

        girder.force_applied_point.insert(int((number_of_deck_spans+1)/2), False)
        
        temp_connection_point_list.insert(int((number_of_deck_spans+1)/2), girder.beam_span/2)
        girder.beam_connection_point = temp_connection_point_list

    girder.deflection_critical = config.deflection_critical

    girder.beam_spacing = comp_beam.beam_span

    girder.distance_to_left_beam = comp_beam.beam_span / 2
    girder.distance_to_right_beam = comp_beam.beam_span / 2
    girder.calculate_effective_width()

    #set unbraced length
    girder.unbraced_length = comp_beam.beam_spacing

    girder_database = pd.DataFrame(columns = df_beam_design.columns)
    girder_database['Footfall Acc.'] = ''
    girder_database['Beam Section'] = ''

    # iterate over beam sections
    for i, row in df_beam_design.iterrows():
        comp_beam.section_size = row['Section Size']
        comp_beam.get_section_properties()
        
        calculate_girder_demand(girder, comp_beam, config)
        
        # for each beam design, we will generate a new girder database for calculations

        # remove beams deeper than max depth
        girder_database_for_calc = beam_database.loc[beam_database['d'] <= max_girder_depth]


        # deflection check, calculate minimum I required for L/240 limit
        minimum_i_for_construction_case = 5/384 * ((20/1000/12) * (girder.beam_span*12)**3) / (girder.steel_modulus_of_elasticity) * 240

        # Eliminate Girder Sizes Shallower than the Beam Depth
        girder_database_for_calc = beam_database_for_calc.loc[beam_database_for_calc['d'] >= comp_beam.section_depth]
        
        girder_database_for_calc = girder_database_for_calc.loc[girder_database_for_calc['Ix'] > minimum_i_for_construction_case]

        # if no results are available move to the next
        if girder_database_for_calc.size == 0:
            continue
                
        girder_database_for_calc['pre_comp_mu'] = girder.m_ultimate_cons_max

        girder_database_for_calc['pre_comp_phi_mn'] = girder_database_for_calc.apply(lambda x: assign_pre_comp_capacity_to_dataframe(x, girder), axis = 1)
      
        girder_database_for_calc['pre_comp_dcr'] = girder.m_ultimate_cons_max / girder_database_for_calc['pre_comp_phi_mn']

        girder_database_for_calc['pre_comp_dcr'] = girder_database_for_calc['pre_comp_dcr'].round(2)

        girder_database_for_calc = girder_database_for_calc.loc[girder_database_for_calc['pre_comp_dcr'] <= 0.95]

        # if no results are available move to the next
        if girder_database_for_calc.size == 0:
            continue
           
        temp_girder_database = generate_composite_beam_dataframe(girder, girder_database_for_calc, config, comp_beam, "girder")

        def check_system_acceleration(girder_size, beam, config):
            acc_girder = girder
            acc_girder.section_size = girder_size
            acc_girder.get_section_properties()
            
            acceleration_divided_by_g = floor_vibration_check(acc_girder, beam, config)

            acceleration_divided_by_g = round(acceleration_divided_by_g, 4)

            return acceleration_divided_by_g
        
        
        if len(temp_girder_database) > 0:

            temp_girder_database['Footfall Acc.'] = temp_girder_database.apply(lambda x: check_system_acceleration(x['Section Size'], comp_beam, config), axis = 1)
        
            temp_girder_database['Beam Section'] = row['Section Size']
            temp_girder_database['Beam Camber'] = row['camber']

            #temp_girder_database[['Total Embodied Carbon', 'Deck Embodied Carbon', 'Concrete Embodied Carbon','Girder Embodied Carbon','Beam Embodied Carbon']] = calculate_carbon(comp_beam, girder, config)

            temp_girder_database = temp_girder_database.loc[temp_girder_database['Footfall Acc.'] <= config.acceleration_limit]
        
        else:
            continue

        girder_database = pd.concat([girder_database, temp_girder_database], ignore_index=True)

    # if no results are available send it back
    if girder_database.size == 0:
        output_message = "Typical Wide-Flange Sections do not work for the given design parameters."
        return "", output_message

    girder_database = calculate_carbon(girder_database, config)

    df_beam_design["Beam Weight (lbs)"] = df_beam_design["Steel Weight"] * (girder.beam_span / comp_beam.beam_spacing)
    df_beam_design["Beam Cost ($)"] = df_beam_design["Total Cost"] * (girder.beam_span / comp_beam.beam_spacing)

    
    girder_database.rename(columns={"Section Size": "Girder Section", "A" : "Girder Steel Area (in2)", "bf": "Girder Flange Width (in)",
                                "d": "Girder Depth (in)", 
                                "pre_comp_mu" : "Girder Pre-composite Mu (kip-ft)", "pre_comp_phi_mn": "Girder Pre-composite phi-Mn (kip-ft)",
                                "pre_comp_dcr" : "Girder Pre-composite DCR",
                                "composite_percent": "Girder Percent Composite",
                                "post_comp_mu" : "Girder Post-composite Mu (kip-ft)", "post_comp_phi_mn": "Girder Post-composite phi-Mn (kip-ft)",
                                "post_comp_dcr" : "Girder Post-composite DCR",
                                "bottom_flange_stress": "Girder Bottom Flange Stress (ksi)",
                                "Ix": "Girder Pre_composite Moment of Inertia (in4)", "post_comp_i_lower_bound": "Girder Post-composite Moment of Inertia (in4)",
                                "number_of_studs": "Girder Studs",
                                "Steel Weight": "Girder Weight (lbs)",
                                "Stud Cost": "Girder Stud Cost ($)", 
                                "Steel Cost": "Girder Steel Cost ($)", 
                                "Total Cost": "Girder Cost ($)", 
                                "defl_dl_ll_camber": "Girder Deflection DL+LL-camber (in)",
                                "l_divided_defl_dl_ll_camber": "Girder L / Deflection DL+LL-camber",
                                "defl_ll": "Girder Deflection LL (in)",
                                "l_divided_defl_ll": "Girder L / Deflection LL",
                                "camber": "Girder Camber (in)", "Beam Camber": "Beam Camber (in)", "neutral_axis_depth": "Girder Neutral Axis Depth (in)", 
                                "total_stud_strength": "Girder Total Stud Strength (kips)",
                                "Footfall Acc.": "Footfall Acceleration (g)"}, inplace = True)

    
    columns_to_keep = ["Girder Section", "Beam Section", 
                    "Girder Depth (in)", "Girder Steel Area (in2)", "Girder Flange Width (in)", 
                    "Girder Pre-composite Mu (kip-ft)", "Girder Pre-composite phi-Mn (kip-ft)", "Girder Pre-composite DCR",
                    "Girder Percent Composite", "Girder Studs",
                    "Girder Post-composite Mu (kip-ft)", "Girder Post-composite phi-Mn (kip-ft)", "Girder Post-composite DCR", 
                    "Girder Bottom Flange Stress (ksi)",
                    "Girder Pre_composite Moment of Inertia (in4)", "Girder Post-composite Moment of Inertia (in4)",
                    "Girder Deflection DL+LL-camber (in)",
                    "Girder L / Deflection DL+LL-camber",
                    "Girder Deflection LL (in)",
                    "Girder L / Deflection LL",
                    "Girder Camber (in)", "Beam Camber (in)",
                    "Girder Neutral Axis Depth (in)", "Girder Total Stud Strength (kips)",
                    "Girder Weight (lbs)",                  
                    "Girder Cost ($)"]

    df_girder_design_output = girder_database[columns_to_keep]  

    columns_for_merge = ["Girder Section",
                    "Beam Section",
                    "Girder Depth (in)", 
                    "Girder Percent Composite", 
                    "Girder Studs",
                    "Girder Camber (in)", 
                    "Beam Camber (in)",
                    "Girder Weight (lbs)",                  
                    "Girder Cost ($)", 
                    "Footfall Acceleration (g)",
                    "Deck Embodied Carbon",
                    "Concrete Embodied Carbon",
                    "Girder Embodied Carbon",
                    "Beam Embodied Carbon",
                    "Total Embodied Carbon"]
                    
    df_girder_design = girder_database[columns_for_merge]    

    df_beam_design.rename(columns={"Section Size": "Beam Section", "A" : "Beam Steel Area (in2)", "bf": "Beam Flange Width (in)",
                                "d": "Beam Depth (in)", 
                                "pre_comp_mu" : "Beam Pre-composite Mu (kip-ft)", "pre_comp_phi_mn": "Beam Pre-composite phi-Mn (kip-ft)",
                                "pre_comp_dcr" : "Beam Pre-composite DCR",
                                "composite_percent": "Beam Percent Composite",
                                "post_comp_mu" : "Beam Post-composite Mu (kip-ft)", "post_comp_phi_mn": "Beam Post-composite phi-Mn (kip-ft)",
                                "post_comp_dcr" : "Beam Post-composite DCR",
                                "bottom_flange_stress": "Beam Bottom Flange Stress (ksi)",
                                "Ix": "Beam Pre_composite Moment of Inertia (in4)", "post_comp_i_lower_bound": "Beam Post-composite Moment of Inertia (in4)",
                                "number_of_studs": "Beam Studs",
                                "Stud Cost": "Beam Stud Cost ($)", 
                                "Steel Cost": "Beam Steel Cost ($)", 
                                "defl_dl_ll_camber": "Beam Deflection DL+LL-camber (in)",
                                "l_divided_defl_dl_ll_camber": "Beam L / Deflection DL+LL-camber",
                                "defl_ll": "Beam Deflection LL (in)",
                                "l_divided_defl_ll": "Beam L / Deflection LL",
                                "camber": "Beam Camber (in)", "neutral_axis_depth": "Beam Neutral Axis Depth (in)", "total_stud_strength": "Beam Total Stud Strength (kips)", 'vu_shear_at_bg_connection':"Ultimate Shear at Beam to Girder Connection (kips)"}, inplace = True)

    columns_to_keep = ["Beam Section", 
                    "Beam Depth (in)", "Beam Steel Area (in2)", "Beam Flange Width (in)", 
                    "Beam Pre-composite Mu (kip-ft)", "Beam Pre-composite phi-Mn (kip-ft)", "Beam Pre-composite DCR",
                    "Beam Percent Composite", "Beam Studs",
                    "Beam Post-composite Mu (kip-ft)", "Beam Post-composite phi-Mn (kip-ft)", "Beam Post-composite DCR", 
                    "Beam Bottom Flange Stress (ksi)",
                    "Beam Pre_composite Moment of Inertia (in4)", "Beam Post-composite Moment of Inertia (in4)",
                    "Beam Deflection DL+LL-camber (in)",
                    "Beam L / Deflection DL+LL-camber",
                    "Beam Deflection LL (in)",
                    "Beam L / Deflection LL",            
                    "Beam Camber (in)",
                    "Beam Neutral Axis Depth (in)", "Beam Total Stud Strength (kips)",
                    "Beam Weight (lbs)",                  
                    "Beam Cost ($)",
                    "Ultimate Shear at Beam to Girder Connection (kips)"]

    df_beam_design_output = df_beam_design[columns_to_keep]

    columns_for_merge = ["Beam Section",
                    "Beam Depth (in)", 
                    "Beam Percent Composite", 
                    "Beam Studs",
                    "Beam Camber (in)", 
                    "Beam Weight (lbs)",                  
                    "Beam Cost ($)",
                    "Ultimate Shear at Beam to Girder Connection (kips)"]
                    
    df_beam_design = df_beam_design[columns_for_merge]  

    df_final_design = df_girder_design.merge(df_beam_design, how="left", left_on=["Beam Section", "Beam Camber (in)"], right_on=["Beam Section", "Beam Camber (in)"])

    df_final_design['Beam Spacing (ft)'] = comp_beam.beam_spacing

    df_final_design["Steel Unit Cost ($/sq-ft)"] = (df_final_design["Girder Cost ($)"] + df_final_design["Beam Cost ($)"])/(girder.beam_span * comp_beam.beam_span) 
    df_final_design["Unit Steel Weight (psf)"] = (df_final_design["Girder Weight (lbs)"] + df_final_design["Beam Weight (lbs)"])/(girder.beam_span * comp_beam.beam_span)
   
    df_final_design["Unit Total Embodied Carbon (kg/sq-ft)"] = df_final_design["Total Embodied Carbon"]/(girder.beam_span * comp_beam.beam_span)
    df_final_design["Unit Deck Embodied Carbon (kg/sq-ft)"] = df_final_design["Deck Embodied Carbon"]/(girder.beam_span * comp_beam.beam_span)
    df_final_design["Unit Concrete Embodied Carbon (kg/sq-ft)"] = df_final_design["Concrete Embodied Carbon"]/(girder.beam_span * comp_beam.beam_span)
    df_final_design["Unit Steel Embodied Carbon (kg/sq-ft)"] = (df_final_design["Girder Embodied Carbon"]+df_final_design["Beam Embodied Carbon"])/(girder.beam_span * comp_beam.beam_span)

    df_final_design[["Steel Unit Cost ($/sq-ft)", "Unit Steel Weight (psf)", "Unit Total Embodied Carbon (kg/sq-ft)", "Unit Deck Embodied Carbon (kg/sq-ft)", "Unit Concrete Embodied Carbon (kg/sq-ft)", "Unit Steel Embodied Carbon (kg/sq-ft)"]] = df_final_design[["Steel Unit Cost ($/sq-ft)", "Unit Steel Weight (psf)", "Unit Total Embodied Carbon (kg/sq-ft)", "Unit Deck Embodied Carbon (kg/sq-ft)", "Unit Concrete Embodied Carbon (kg/sq-ft)", "Unit Steel Embodied Carbon (kg/sq-ft)"]].astype("float")

    df_final_design[["Steel Unit Cost ($/sq-ft)", "Unit Steel Weight (psf)", "Unit Total Embodied Carbon (kg/sq-ft)", "Unit Deck Embodied Carbon (kg/sq-ft)", "Unit Concrete Embodied Carbon (kg/sq-ft)", "Unit Steel Embodied Carbon (kg/sq-ft)"]] = round(df_final_design[["Steel Unit Cost ($/sq-ft)", "Unit Steel Weight (psf)", "Unit Total Embodied Carbon (kg/sq-ft)", "Unit Deck Embodied Carbon (kg/sq-ft)", "Unit Concrete Embodied Carbon (kg/sq-ft)", "Unit Steel Embodied Carbon (kg/sq-ft)"]],2)

    if design_priority == 'EmbodiedCarbon':
        selection_criteria = "Unit Total Embodied Carbon (kg/sq-ft)"
    elif design_priority == 'Cost':
        selection_criteria = "Steel Unit Cost ($/sq-ft)"
    else:
        selection_criteria = "Unit Steel Weight (psf)"

    df_final_design = df_final_design.nsmallest(1, selection_criteria, keep='all')

    df_design_calc = df_girder_design_output.merge(df_beam_design_output, how="left", left_on=["Beam Section", "Beam Camber (in)"], right_on=["Beam Section", "Beam Camber (in)"])

    # Filter to match index of df_final_design   
    df_design_calc = df_design_calc[df_design_calc.index.isin(df_final_design.index)]

    df_design_calc['Total Weight (lbs)']= df_design_calc['Beam Weight (lbs)'] + df_design_calc['Girder Weight (lbs)']
    df_design_calc["Total Weight (lbs)"] = df_design_calc["Total Weight (lbs)"].astype(float)
    
    #df_design_calc = df_design_calc.nsmallest(10, selection_criteria, keep='all')

    # Convert to kip-ft
    df_design_calc[["Beam Pre-composite Mu (kip-ft)", "Beam Pre-composite phi-Mn (kip-ft)", "Beam Post-composite Mu (kip-ft)", "Beam Post-composite phi-Mn (kip-ft)", 
    "Girder Pre-composite Mu (kip-ft)", "Girder Pre-composite phi-Mn (kip-ft)", "Girder Post-composite Mu (kip-ft)", "Girder Post-composite phi-Mn (kip-ft)"]] = df_design_calc[["Beam Pre-composite Mu (kip-ft)", "Beam Pre-composite phi-Mn (kip-ft)", "Beam Post-composite Mu (kip-ft)", "Beam Post-composite phi-Mn (kip-ft)", 
    "Girder Pre-composite Mu (kip-ft)", "Girder Pre-composite phi-Mn (kip-ft)", "Girder Post-composite Mu (kip-ft)", "Girder Post-composite phi-Mn (kip-ft)"]] / 12

    #Round Values
    df_design_calc[["Beam Pre-composite Mu (kip-ft)", "Beam Pre-composite phi-Mn (kip-ft)", "Beam Post-composite Mu (kip-ft)", "Beam Post-composite phi-Mn (kip-ft)", 
    "Girder Pre-composite Mu (kip-ft)", "Girder Pre-composite phi-Mn (kip-ft)", "Girder Post-composite Mu (kip-ft)", "Girder Post-composite phi-Mn (kip-ft)", 
    "Beam Total Stud Strength (kips)", "Girder Total Stud Strength (kips)","Ultimate Shear at Beam to Girder Connection (kips)"]] = df_design_calc[["Beam Pre-composite Mu (kip-ft)", "Beam Pre-composite phi-Mn (kip-ft)", "Beam Post-composite Mu (kip-ft)", "Beam Post-composite phi-Mn (kip-ft)", 
    "Girder Pre-composite Mu (kip-ft)", "Girder Pre-composite phi-Mn (kip-ft)", "Girder Post-composite Mu (kip-ft)", "Girder Post-composite phi-Mn (kip-ft)", 
    "Beam Total Stud Strength (kips)", "Girder Total Stud Strength (kips)","Ultimate Shear at Beam to Girder Connection (kips)"]].astype(int)

    output_message = "Success."
    
    df_final_design = df_final_design.iloc[0]

    response = df_final_design.to_dict()

    return response, output_message