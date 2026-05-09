## Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import json

from strategies import (
    S1_reactive_Update,
    S2_trendmonitor_Update,
    S3_SolarRadiationFF_Update,
    S4_predictive_Update,
    S5_HybridFF_MPC_Update
)

## CSV file reading
input_filename = "common_strategy_input_P1_nominal.csv"
data = pd.read_csv(input_filename)

## JSON configuration reading
with open("config.json") as f:
    config = json.load(f)
    

## GENERAL CONFIG
general = config["general"]

sample_time = general["sample_time_sec"]
hold_time = general["hold_time_sec"]
strategy = general["strategy"]

## COMMON CONFIG
limits = config["limits"]
cooling = config["cooling"]

tempLow = limits["tempLow"]
tempRef = limits["tempRef"]
tempMedium = limits["tempMedium"]
tempHigh = limits["tempHigh"]
tempMax = limits["tempMax"]
tempLimits = limits["tempLimits"]

coolingPower = cooling["power"]
coolingEnergy = cooling["energy"]

temp_max = config["safety_checks"]["temp_max"]

## STRATEGY CONFIG
strategyConfig = config[strategy]

## S2 CONFIG
if strategy == "S2":
    s2 = strategyConfig

    tempRiseThreshold = s2["tempRiseThreshold"]
    movingAvgWindowSize = s2["movingAvgWindowSize"]
    trendWindowSize = s2["trendWindowSize"]

elif strategy == "S3":
    s3 = strategyConfig

    solRadMovingAvgWindowSize = s3["solRadMovingAvgWindowSize"]
    solRadTrendWindowSize = s3["solRadTrendWindowSize"]

    ffIndexWindowSize = s3["ffIndexWindowSize"]

    kSolRadLevel = s3["kSolRadLevel"]
    kSolRadTrend = s3["kSolRadTrend"]

elif strategy == "S4":
    s4 = strategyConfig

    horizonLength = s4["horizon"]

    thermal = s4["thermal_model"]
    control = s4["control"]
    weights = s4["weights"]

    alphaG = thermal["alphaG"]
    kThrottle = thermal["kThrottle"]
    gamma = thermal["gamma"]
    internalResistance = thermal["internalResistance"]
    modelParam_a = thermal["modelParam_a"]
    modelParam_b = thermal["modelParam_b"]

    propulsionPowerGain = control["propulsionPowerGain"]
    throttleWeightInLoad = control["throttleWeightInLoad"]
    throttleRampEco = control["throttleRampEco"]
    throttleRampPower = control["throttleRampPower"]

    solarMovingAvgWindow = control["solarMovingAvgWindow"]

    wEnergy = weights["wEnergy"]
    wTemp = weights["wTemp"]
    wSwitch = weights["wSwitch"]
    wTerminal = weights["wTerminal"]

elif strategy == "S5":
    s5 = strategyConfig

    horizonLength = s5["horizon"]

    thermal = s5["thermal_model"]
    control = s5["control"]
    weights = s5["weights"]

    alphaG = thermal["alphaG"]
    kThrottle = thermal["kThrottle"]
    gamma = thermal["gamma"]
    internalResistance = thermal["internalResistance"]
    modelParam_a = thermal["modelParam_a"]
    modelParam_b = thermal["modelParam_b"]

    propulsionPowerGain = control["propulsionPowerGain"]

    throttleWeightInLoad = control["throttleWeightInLoad"]
    throttleRampEco = control["throttleRampEco"]
    throttleRampPower = control["throttleRampPower"]

    solarMovingAvgWindow = control["solarMovingAvgWindow"]

    solRadMovingAvgWindowSize = control["solRadMovingAvgWindowSize"]
    solRadTrendWindowSize = control["solRadTrendWindowSize"]

    kSolRadLevel = control["kSolRadLevel"]
    kSolRadTrend = control["kSolRadTrend"]

    wEnergy = weights["wEnergy"]
    wTemp = weights["wTemp"]
    wSwitch = weights["wSwitch"]
    wTerminal = weights["wTerminal"]
    wFFBias = weights["wFFBias"]

## Extracting relevant columns from the data
pv = data["PV_value"].values
solar = data["solar"].values
ambient = data["ambient_temp"].values
throttle = data["throttle"].values
op_mode = data["operation_mode"].values
sim_batteryTemp = data["battery_temp"].values

## Safety checks function to ensure water presence and prevent overheating
def check_coolant(waterIn: bool):
    if not waterIn:
        return 0, 66
    return None, None

def check_overheat(temp: float, temp_max: float):
    if temp >= temp_max:
        max_cooling_level = len(tempLimits)
        return max_cooling_level, 67
    return None, None

## Initial conditions
soc = 0.8
sim_batteryTemp = data["battery_temp"].values

rawHistory = []
filteredHistory = []
pv_raw_history = []
pv_filtered_history = []
ffIndexHistory = []

coolingLevels = []
switch_times = []

## Main control loop
i = 0
prevCoolingLevel = 0
time_since_last_switch = 0.0
prevThrottle = throttle[0]

while i < len(data):

    temp = sim_batteryTemp[i]
    pv_val = data["PV_value"].iloc[i]
    waterIn = True

    # Safety check
    override_level, error_code = check_coolant(waterIn)

    if override_level is None:
        override_level, error_code = check_overheat(temp, temp_max)

    if override_level is not None:
        level = override_level
        print(f"ERROR CODE: {error_code}, CoolingLevel set to {level}")
        coolingLevels.append(level)
        switch_times.append(i * sample_time)
        prevCoolingLevel = level
        i += 1
        continue

    if strategy == "S1":
        level = S1_reactive_Update(temp, tempLimits)

    elif strategy == "S2":
        fTemp, trend, level = S2_trendmonitor_Update(
            temp,
            prevCoolingLevel,
            tempLimits,
            tempRiseThreshold,
            movingAvgWindowSize,
            trendWindowSize,
            filteredHistory,
            rawHistory
        )

    elif strategy == "S3":
        level, filtered_pv, trend, ff_index = S3_SolarRadiationFF_Update(
            temp,
            pv_val,
            pv_raw_history,
            pv_filtered_history,
            ffIndexHistory,
            solRadMovingAvgWindowSize,
            solRadTrendWindowSize,
            ffIndexWindowSize,
            sample_time,
            kSolRadLevel,
            kSolRadTrend,
            tempLimits
        )

    elif strategy == "S4":

        currentThrottle = throttle[i]
        operationMode = op_mode[i]

        if i + horizonLength < len(data):

            forecastAmbient = ambient[i:i+horizonLength]
            forecastSolar = solar[i:i+horizonLength]

            level = S4_predictive_Update(
                batteryTempFiltered=sim_batteryTemp,
                prevCoolingLevel=prevCoolingLevel,
                currentThrottle=currentThrottle,
                prevThrottle=prevThrottle,
                operationMode=operationMode,
                forecastSolarRadiation=forecastSolar,
                forecastAmbientTemp=forecastAmbient,
                sample_time_sec=sample_time,
                horizonLength=horizonLength,
                alphaG=alphaG,
                kThrottle=kThrottle,
                throttleWeightInLoad=throttleWeightInLoad,
                throttleRampEco=throttleRampEco,
                throttleRampPower=throttleRampPower,
                internalResistance=internalResistance,
                modelParam_a=modelParam_a,
                modelParam_b=modelParam_b,
                coolingPowerLevel=coolingPower,
                coolingEnergyLevel=coolingEnergy,
                wEnergy=wEnergy,
                wTemp=wTemp,
                wSwitch=wSwitch,
                wTerminal=wTerminal,
                tempRef=tempRef,
                tempHighLimit=tempHigh,
                tempMaxLimit=tempMax,
                solarMovingAvgWindow=solarMovingAvgWindow
            )
        else:
            level = prevCoolingLevel

    elif strategy == "S5":

        currentThrottle = throttle[i]
        operationMode = op_mode[i]

        if i + horizonLength < len(data):

            forecastAmbient = ambient[i:i+horizonLength]
            forecastSolar = solar[i:i+horizonLength]

            level, ffLevel, ffIndex, solRadTrend, bestCost = S5_HybridFF_MPC_Update(
                batteryTempFiltered=sim_batteryTemp,
                prevCoolingLevel=prevCoolingLevel,
                pvSolarRadiationRaw=pv_val,
                pvSolarRadiationRawHistory=pv_raw_history,
                pvSolarRadiationFilteredHistory=pv_filtered_history,
                solRadMovingAvgWindowSize=solRadMovingAvgWindowSize,
                solRadTrendWindowSize=solRadTrendWindowSize,
                currentThrottle=currentThrottle,
                prevThrottle=prevThrottle,
                operationMode=operationMode,
                forecastSolarRadiation=forecastSolar,
                forecastAmbientTemp=forecastAmbient,
                sample_time_sec=sample_time,
                horizonLength=horizonLength,
                kSolRadLevel=kSolRadLevel,
                kSolRadTrend=kSolRadTrend,
                ffToMedium=ffToMedium,
                ffToHigh=ffToHigh,
                alphaG=alphaG,
                kThrottle=kThrottle,
                throttleWeightInLoad=throttleWeightInLoad,
                throttleRampEco=throttleRampEco,
                throttleRampPower=throttleRampPower,
                internalResistance=internalResistance,
                modelParam_a=modelParam_a,
                modelParam_b=modelParam_b,
                coolingPowerLevel=coolingPower,
                coolingEnergyLevel=coolingEnergy,
                wEnergy=wEnergy,
                wTemp=wTemp,
                wSwitch=wSwitch,
                wTerminal=wTerminal,
                wFFBias=wFFBias,
                tempRef=tempRef,
                tempMediumLimit=tempMedium,
                tempHighLimit=tempHigh,
                tempMaxLimit=tempMax,
                solarMovingAvgWindow=solarMovingAvgWindow
            )
        else:
            level = prevCoolingLevel

    time_since_last_switch += sample_time

    if level != prevCoolingLevel:
        if time_since_last_switch >= hold_time:
            prevCoolingLevel = level
            switch_times.append(time_since_last_switch)
            time_since_last_switch = 0
        else:
            level = prevCoolingLevel
            time_since_last_switch += sample_time
            switch_times.append(time_since_last_switch)
    else:
        level = prevCoolingLevel
        time_since_last_switch += sample_time
        switch_times.append(time_since_last_switch)

    prevThrottle = throttle[i]

    i += 1

    coolingLevels.append(level)

output = pd.DataFrame({
    "Time": switch_times,
    "CoolingLevel": coolingLevels
})

base_name = input_filename.replace("common_strategy_input_", "").replace(".csv", "")
filename = f"{strategy}_{base_name}_output.csv"
output.to_csv(filename, index=False)
