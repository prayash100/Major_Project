MAX_BMS_DISCHARGE = 208.3  # Max discharge capacity in MW per 100s
MAX_BMS_CHARGE = 208.3     # Max charge rate in MW per 100s
BMS_SOC_THRESHOLD = 0.2    # Minimum allowable SOC (20%)
BMS_CAPACITY = 5000        # BMS energy capacity in MW (5 GWh over 1 hour)
BMS_TEMPERATURE_RANGE = (-20, 55)  # Safe operating temperature in Celsius


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
load_model = keras.models.load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
from dotenv import load_dotenv
load_dotenv()
import tensorflow as tf
tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow logs 

import pandas as pd
# Step 1: Define grid parameters
renewable_capacity = {'solar':2000, 'wind': 4000}  # MW
conventional_capacity = {'coal': 13000, 'gas': 10000}  # MW

scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

df = pd.read_csv("dataset/corrected_gujarat_load_demand_2024.csv")  # Replace with your file path
df_solar_n_wind = pd.read_csv("dataset/aggregated_solar_wind_data.csv")
df_coal = pd.read_csv("dataset/coal_ramp_schedule.csv")
df["Index"] = range(len(df))
df['Load_Demand_MW'] = df['Load_Demand_MW'].round(2)

sc=pd.read_csv("sceduler.csv")
#change
import csv
import os
# Path to your CSV file
csv_file = 'new/pso/pso_new.csv'

# Define your column names
fieldnames = ['Index', 'solar_needed', 'solar_availability', 'wind_needed', 'wind_availability', 'coal_needed', 'coal_availability', 'gas_needed', 'gas_availability', 'sortfall', 'BMS_status','BMS_action','total_cost', 'frequency_mean','frequency_std_dev', 'frequency_max_deviation', 'frequency_min_deviation', 'within_0_01', 'within_0_02', 'within_0_05']

# Check if file exists, if not, create it and write headers
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
i=0
step = 0 
# Current BMS status
bms_status = {
        'SOC': 0.9,  # Current State of Charge (90%)
        'temperature': 25  # Current temperature in Celsius
    }
all_frequencies = []
for i in range(0,86300, 100):

    df_w = df.iloc[i:200+i]  
    df_coal = pd.read_csv("coal_ramp_schedule.csv")
    load_demand = df_w['Load_Demand_MW'].values

    def load_lstm_model(model_path='model/lstm_load_prediction.h5'):
        return load_model(model_path, compile=False)

    def predict_next_100_seconds(last_200_seconds, model):
        
        
        # Reshape input
        last_200_seconds = np.array(last_200_seconds).reshape(1, -1)
        
        
        last_200_seconds_scaled = scaler_X.transform(last_200_seconds).reshape(1, 200, 1)
        

        # Predict
        predicted_scaled = model.predict(last_200_seconds_scaled)[0]
        

        # Inverse transform
        predicted_actual = scaler_y.inverse_transform(predicted_scaled.reshape(1, -1))[0]

        return predicted_actual

    def renewable_generation(renewable_capacity,i):
        hour_idx = int(i / 3600)
        solar_output = renewable_capacity['solar'] * (np.random.rand() * sc['sol_dif'][hour_idx] + sc['sol_min'][hour_idx] )  # Random output between 20% to 100% of capacity
        wind_output = renewable_capacity['wind'] * (np.random.rand() * sc['wind_dif'][hour_idx]  + sc['wind_min'][hour_idx])   # Random output between 30% to 100% of capacity
        return solar_output, wind_output

    def conventional_generation(conventional_capacity,i):
        hour_idx = int(i / 3600)
        coal_output = conventional_capacity['coal'] * (sc['coal'][hour_idx])
        gas_output = conventional_capacity['gas'] * (sc['gas'][hour_idx])  # Random output between 30% to 100% of capacity
        return coal_output, gas_output

    def grid_frequency(solar_output, wind_output, coal_output, gas_output, load_demand):
        total_generation = solar_output + wind_output + coal_output + gas_output 
        delta_P = total_generation - load_demand  # Power mismatch

        # Inertia constants (H), lower for RES, higher for coal/gas
        H_solar, H_wind, H_coal, H_gas = 0.05, 0.1, 5.0, 3.0
        D_solar, D_wind, D_coal, D_gas= 0.02, 0.03, 0.3, 0.2  # Damping factors

        # Weighted total inertia and damping effect
        H_total = (solar_output * H_solar + wind_output * H_wind +
                coal_output * H_coal + gas_output * H_gas  ) / (total_generation + 1e-6)

        D_total = (solar_output * D_solar + wind_output * D_wind +
                coal_output * D_coal + gas_output * D_gas  ) / (total_generation + 1e-6)

        # Frequency deviation is higher if RES contribution is higher
        frequency_deviation = (delta_P * 0.00008) / (H_total + D_total + 1e-6)  # Prevent div by zero
        return frequency_deviation

    def particle_swarm_optimization(renewable_capacity, conventional_capacity, data, iterations, swarm_size=50):
        swarm = np.random.rand(swarm_size, 4) * [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']]
        velocities = np.zeros_like(swarm)
        best_positions = np.copy(swarm)
        best_deviation = np.inf * np.ones(swarm_size)
        
        global_best_position = best_positions[0]
        global_best_deviation = np.inf

        for _ in range(iterations):
            for i in range(swarm_size):
                solar_output, wind_output, coal_output, gas_output = swarm[i]
                frequency_deviation = grid_frequency(solar_output, wind_output, coal_output, gas_output, data)
                
                # Ensure that frequency_deviation is a scalar value
                if isinstance(frequency_deviation, np.ndarray):
                    frequency_deviation = frequency_deviation[0]  # Take the first element if it's an array

                # Update best deviation for each particle
                if abs(frequency_deviation) < best_deviation[i]:
                    best_positions[i] = swarm[i]
                    best_deviation[i] = abs(frequency_deviation)

                # Update global best solution
                if abs(frequency_deviation) < global_best_deviation:
                    global_best_position = swarm[i]
                    global_best_deviation = abs(frequency_deviation)
            
            inertia_weight = 0.9
            cognitive_weight = 1.5
            social_weight = 1.4
            
            for i in range(swarm_size):
                velocities[i] = inertia_weight * velocities[i] + \
                                cognitive_weight * np.random.rand() * (best_positions[i] - swarm[i]) + \
                                social_weight * np.random.rand() * (global_best_position - swarm[i])
                swarm[i] = swarm[i] + velocities[i]
                swarm[i] = np.clip(swarm[i], 0, [renewable_capacity['solar'], renewable_capacity['wind'], conventional_capacity['coal'], conventional_capacity['gas']])
        
        return  np.trunc(global_best_position * 100) / 100

    def calculate_total_cost(solar_output, wind_output, coal_output, gas_output):
        # Adjusted costs to reflect market prices more realistically
        cost_per_mw = {
            'solar': 50,  # $/MWh
            'wind': 90,   # $/MWh
            'coal': 130,   # $/MWh
            'gas': 60     # $/MWh
        }
        
        # Ensure inputs are NumPy arrays
        solar_output = np.asarray(solar_output)
        wind_output = np.asarray(wind_output)
        coal_output = np.asarray(coal_output)
        gas_output = np.asarray(gas_output)
        
        # Calculate total cost
        total_cost = np.sum(
            solar_output * cost_per_mw['solar'] + 
            wind_output * cost_per_mw['wind'] + 
            coal_output * cost_per_mw['coal'] + 
            gas_output * cost_per_mw['gas']
        )
        
        return total_cost

    import numpy as np

    def frequency_array(solution, load_demand, num_steps=100):
        """
        Parameters:
            solution: np.array of shape (num_steps, 4) => columns: [solar, wind, coal, gas]
            load_demand: np.array of shape (num_steps,) => demand at each time step
            num_steps: int => number of time steps to simulate

        Returns:
            np.array of frequency in Hz for each time step
        """
        # Inertia constants (H), lower for RES, higher for coal/gas
        H_solar, H_wind, H_coal, H_gas = 0.05, 0.1, 5.0, 3.0
        D_solar, D_wind, D_coal, D_gas = 0.02, 0.03, 0.3, 0.2  # Damping factors

        frequency_obtained = []

        for t in range(num_steps):
            solar_output, wind_output, coal_output, gas_output = solution
            total_generation = solar_output + wind_output + coal_output + gas_output
            delta_P = total_generation - load_demand[t]

            H_total = (solar_output * H_solar + wind_output * H_wind +
                    coal_output * H_coal + gas_output * H_gas) / (total_generation + 1e-6)

            D_total = (solar_output * D_solar + wind_output * D_wind +
                    coal_output * D_coal + gas_output * D_gas) / (total_generation + 1e-6)

            frequency_deviation = (delta_P * 0.00008) / (H_total + D_total + 1e-6)
            obtained_hz = 50 + frequency_deviation
            frequency_obtained.append(obtained_hz)

        return np.array(frequency_obtained) 

    def generate_frequency_report(frequencies):
        
        ideal_freq = 50
        frequencies = np.array(frequencies)  # Ensure it's a NumPy array
        all_frequencies.append(frequencies)
        # Calculate deviations
        deviations = np.abs(frequencies - ideal_freq)
        max_deviation = np.max(deviations)
        min_deviation = np.min(deviations)
        mean_frequency = np.mean(frequencies)
        std_dev = np.std(frequencies)

        # Percentage of values within tolerance ranges
        within_0_01 = np.sum(deviations <= 0.01) / len(frequencies) * 100
        within_0_02 = np.sum(deviations <= 0.02) / len(frequencies) * 100
        within_0_05 = np.sum(deviations <= 0.05) / len(frequencies) * 100

        return {
        'frequency_mean': round(mean_frequency,4),
        'frequency_std_dev': round(std_dev,4),
        'frequency_max_deviation': round(max_deviation,4),
        'frequency_min_deviation': round(min_deviation,4),
        'within_0_01': round(within_0_01,2),
        'within_0_02': round(within_0_02,2),
        'within_0_05': round(within_0_05,2)
    }

    data=predict_next_100_seconds(load_demand, load_lstm_model())

    # Run ga 
    solar_out, wind_out = renewable_generation(renewable_capacity, i)
    coal_out, gas_out = conventional_generation(conventional_capacity, i)
    renewable_capacity_av = {'solar':solar_out, 'wind': wind_out}  # MW 
    conventional_capacity_av = {'coal': coal_out, 'gas': gas_out}  # MW
    best_solution = particle_swarm_optimization(renewable_capacity_av, conventional_capacity_av, data, 1000)

    solar_output_pso, wind_output_pso, coal_output_pso, gas_output_pso = best_solution

    # Available power from sources (in MW)
    availability = {
        #change 
        'SOLAR': df_solar_n_wind['solar'].iloc[step],
        'WIND': df_solar_n_wind['wind'].iloc[step],
        'COAL':df_coal['Coal_Power_Availability'][int(i/3600)],  # Available but not max capacity
        'GAS': df_coal['Gas_Power_Availability'][int(i/3600)],  # Available but not max capacity
        'BMS': 0  # Starts with 0 MW, will be updated dynamically
    }
    step += 1
    # Predicted optimal power usage from ML model (in MW)
    optimal_allocation = {
        'SOLAR': solar_output_pso,
        'WIND': wind_output_pso,
        'COAL': coal_output_pso,  # Using only 8074.4 MW even though 12000 MW is available
        'GAS': gas_output_pso   # Using only 9920.94 MW even though 10000 MW is available
    }

    bms_action = {
    'status': 'idle'  # can be 'discharging', 'charging', or 'idle'
    }


    def can_bms_discharge(required_power):
        """Ensure BMS is healthy and SOC > threshold."""
        if bms_status['SOC'] <= BMS_SOC_THRESHOLD:
            return False
        if not (BMS_TEMPERATURE_RANGE[0] <= bms_status['temperature'] <= BMS_TEMPERATURE_RANGE[1]):
            return False
        return True

    def can_bms_charge():
        """Check if BMS can accept charge."""
        if bms_status['SOC'] >= 1.0:
            return False
        if not (BMS_TEMPERATURE_RANGE[0] <= bms_status['temperature'] <= BMS_TEMPERATURE_RANGE[1]):
            return False
        return True
    
    def allocate_power():
        total_optimal = sum(optimal_allocation.values())
        total_availability = sum([
            availability['COAL'],
            availability['GAS'],
            availability['SOLAR'],
            availability['WIND'],
            availability['BMS']
        ])

        hour_of_day = int(i / 3600) % 24  # 0 to 23

        if total_availability < total_optimal:
            shortfall = round(total_optimal - total_availability, 2)

            if can_bms_discharge(shortfall):

                # 🛡️ SOC protection: no discharge below 40%
                if bms_status['SOC'] < 0.2:
                    return  # Preserve battery for emergency

                # 📉 Tiered Discharge Strategy
                if shortfall > 150:
                    discharge_limit = MAX_BMS_DISCHARGE
                elif shortfall > 50:
                    discharge_limit = MAX_BMS_DISCHARGE * 0.7
                else:
                    discharge_limit = MAX_BMS_DISCHARGE * 0.4

                # 🔁 Modify discharge range based on SOC (more aggressive if > 80%)
                if bms_status['SOC'] > 0.8:
                    discharge_limit *= 1.2
                elif bms_status['SOC'] < 0.5:
                    discharge_limit *= 0.8

                allowed_by_soc = bms_status['SOC'] * BMS_CAPACITY
                bms_discharge = round(min(shortfall, discharge_limit, allowed_by_soc), 2)

                availability['BMS'] = bms_discharge
                bms_status['SOC'] = round(bms_status['SOC'] - (bms_discharge / BMS_CAPACITY), 4)
                bms_action['status'] = 'discharging'
            return  # Skip charging logic if there is a shortfall

        # ⚡ Surplus Power Distribution
        coal_surplus = round(availability['COAL'] - optimal_allocation['COAL'], 2)
        gas_surplus = round(availability['GAS'] - optimal_allocation['GAS'], 2)
        wind_surplus = round(availability['WIND'] - optimal_allocation['WIND'], 2)
        solar_surplus = round(availability['SOLAR'] - optimal_allocation['SOLAR'], 2)
        total_surplus = round(coal_surplus + gas_surplus + wind_surplus + solar_surplus, 2)

        # 🌞 Solar Priority Charging (11am to 3pm)
        if total_surplus > 0 and can_bms_charge():
            charging_multiplier = 1.2 if 11 <= hour_of_day <= 15 else 1.0
            max_charge = MAX_BMS_CHARGE * charging_multiplier
            max_allowed_by_soc = (1 - bms_status['SOC']) * BMS_CAPACITY

            bms_charge = round(min(total_surplus, max_charge, max_allowed_by_soc), 2)
            availability['BMS'] = bms_charge
            bms_status['SOC'] = round(bms_status['SOC'] + (bms_charge / BMS_CAPACITY), 2)
            bms_action['status'] = 'charging'
    # Execute power allocation
    allocate_power()

    total_cost = calculate_total_cost(min(availability['SOLAR'],optimal_allocation['SOLAR']) , min(availability['WIND'],optimal_allocation['WIND']),availability['COAL'], availability['GAS'])

    actual={'SOLAR': min(availability['SOLAR'],optimal_allocation['SOLAR']),
        'WIND':min(availability['WIND'],optimal_allocation['WIND']),
        'COAL': availability['COAL'], 
        'GAS':availability['GAS'] }
    filtered_availability = dict(list(actual.items())[:4])
    # Ensure load_demand is a list of floats
    solution = list(filtered_availability.values())

    a=frequency_array(solution, data)
    frequency_stats = generate_frequency_report(a)
    total_optimal = optimal_allocation['COAL'] + optimal_allocation['GAS']+ optimal_allocation['SOLAR'] + optimal_allocation['WIND']
    total_availability =  availability['COAL'] + availability['GAS']+ availability['SOLAR'] + availability['WIND'] 
    row = {
    'Index': int(i / 100),
    'solar_needed': round(optimal_allocation['SOLAR'],2),
    'solar_availability': round(availability['SOLAR'],2),
    'wind_needed': round(optimal_allocation['WIND'],2),
    'wind_availability': round(availability['WIND'],2),
    'coal_needed': round(optimal_allocation['COAL'],2),
    'coal_availability': round(availability['COAL'],2),
    'gas_needed': round(optimal_allocation['GAS'],2),
    'gas_availability': round(availability['GAS'],2),
    'sortfall': round( total_optimal-total_availability,2),
    'BMS_status': round(bms_status['SOC'],2),
    'BMS_action': bms_action['status'],
    'total_cost': round(total_cost,2),
    **frequency_stats  
    }

    # Write to CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(row) 
all_frequencies = np.array(all_frequencies)
np.save("new/pso/freq/pso_frequency.npy", all_frequencies)    
