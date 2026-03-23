## Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import List
import math

## CSV file reading
data = pd.read_csv("S4data.csv")

solar = data["solar"].values
ambient = data["ambient_temp"].values
throttle = data["throttle"].values
battery_temp = data["battery_temp"].values
op_mode = data["operation_mode"].values

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


## S4 logic: Predictive Control
def S4_predictive_Update(
    batteryTempFiltered,
    prevCoolingLevel,
    currentThrottle,
    prevThrottle,
    operationMode,
    forecastSolarRadiation,
    forecastAmbientTemp,
    sampleTimeSec,
    horizonLength,
    alphaG,
    kThrottle,
    throttleWeightInLoad,
    throttleRampEco,
    throttleRampPower,
    internalResistance,
    modelParam_a,
    modelParam_b,
    coolingPowerLevel,
    coolingEnergyLevel,
    wEnergy,
    wTemp,
    wSwitch,
    wTerminal,
    tempRef,
    tempHighLimit,
    tempMaxLimit,
    solarMovingAvgWindow
):

    # Horizon arrays initialization  
    forecastThrottle = [0] * horizonLength
    Qgen = [0] * horizonLength
    I_ch_rawHistory = [0] * horizonLength

    # Throttle prediction on the horizon
    dThrottle = currentThrottle - prevThrottle
    print(f"dThrottle: {dThrottle:.4f}")
    if operationMode == "eco": # if we are in eco mode 
        rampLimit = throttleRampEco # we use the eco ramp limit 
    else:
        rampLimit = throttleRampPower # otherwise we are in power mode and we use the lighter limit for throttle change

    # We use the clamp function to ensure that the throttle change does not exceed the ramp limit and that the throttle stays within [0, 1]
    forecastThrottle[0] = clamp(currentThrottle, 0, 1)

    # We then predict the throttle for the rest of the horizon based on the previous throttle and the change, while respecting the ramp limits
    for k in range(1, horizonLength): 
        delta = clamp(dThrottle, -rampLimit, rampLimit)
        forecastThrottle[k] = forecastThrottle[k - 1] + delta 
        forecastThrottle[k] = clamp(forecastThrottle[k], 0, 1) 

    # Heat prediction on the horizon with moving average for solar radiation
    for k in range(horizonLength):
        I_ch_rawHistory[k] = alphaG * forecastSolarRadiation[k]
        sum = 0
        count = 0
        startIndex = k - solarMovingAvgWindow + 1

        if startIndex < 0:
            startIndex = 0
        
        for i in range(startIndex, k+1):
            sum += I_ch_rawHistory[i]
            count += 1

        I_ch = sum / count if count > 0 else 0

        # We calculate the throttle load 
        loadThrottle = kThrottle * forecastThrottle[k] 

        # The total load is a combination of the current load and the throttle load
        loadTotal = (
            (1 - throttleWeightInLoad) * I_ch
            + throttleWeightInLoad * loadThrottle
        )

        # The heat generation (Qgen) is then calculated based on the total load and the internal resistance
        Qgen[k] = internalResistance * (loadTotal ** 2)

    # Optimization
    bestCost = float('inf')
    bestLevel = prevCoolingLevel

    # Cooling level inspection loop
    for u_candidate in range(4): # we have 4 cooling levels (0, 1, 2, 3)
        if abs(u_candidate - prevCoolingLevel) > 1: 
            continue
        Temp_sim = batteryTempFiltered
        cost = 0
        u_prev = prevCoolingLevel

        for k in range(horizonLength):
            # We simulate the battery temperature at each step of the horizon by applying the thermal model
            Temp_sim = Temp_sim + sampleTimeSec * (modelParam_a * (forecastAmbientTemp[k] - Temp_sim) + modelParam_b * Qgen[k] - coolingPowerLevel[u_candidate])
        
            if Temp_sim > tempMaxLimit: 
                cost = float("inf")
                break
    
            overTemp = max(0, Temp_sim - tempRef) 

            # We calculate the stage cost for this candidate cooling level
            stageCost = (
                wEnergy * coolingEnergyLevel[u_candidate]
                + wTemp * (overTemp ** 2)
                + wSwitch * ((u_candidate - u_prev) ** 2)
            )

            cost += stageCost
            u_prev = u_candidate

        if cost < float("inf"):
            overTemp = max(0, Temp_sim - tempRef)
            terminalCost = wTerminal * (overTemp ** 2)
            cost += terminalCost

        if cost < bestCost:
            bestCost = cost
            bestLevel = u_candidate
    
    if batteryTempFiltered >= tempHighLimit:
        return 3 
    
    return bestLevel

## Simulation loop for S4
results = []

prevCoolingLevel = 0
prevThrottle = throttle[0]

horizonLength = 10
sampleTimeSec = 1

for t in range(len(data) - horizonLength):

    # Inputs
    currentThrottle = throttle[t]
    operationMode = op_mode[t]
    batteryTempFiltered = battery_temp[t]

    # Forecast of solar radiation and ambient temperature on the horizon
    forecastSolar = solar[t:t+horizonLength] 
    forecastAmbient = ambient[t:t+horizonLength]

    # Function call
    coolingLevel = S4_predictive_Update(
        batteryTempFiltered=batteryTempFiltered,
        prevCoolingLevel=prevCoolingLevel,
        currentThrottle=currentThrottle,
        prevThrottle=prevThrottle,
        operationMode=operationMode,
        forecastSolarRadiation=forecastSolar,
        forecastAmbientTemp=forecastAmbient,
        sampleTimeSec=sampleTimeSec,
        horizonLength=horizonLength,

        alphaG=0.01,
        kThrottle=1.0,
        throttleWeightInLoad=0.5,
        throttleRampEco=0.02,
        throttleRampPower=0.05,

        internalResistance=0.01,
        modelParam_a=0.1,
        modelParam_b=0.01,

        coolingPowerLevel=[0, 0.5, 1.0, 1.5],
        coolingEnergyLevel=[0, 1, 2, 3],

        wEnergy=1.0,
        wTemp=5.0,
        wSwitch=0.5,
        wTerminal=2.0,

        tempRef=30,
        tempHighLimit=40,
        tempMaxLimit=50,

        solarMovingAvgWindow=3
    )

    print(f"t={t} | Temp={batteryTempFiltered:.2f} | Throttle={currentThrottle:.2f} | Cooling={coolingLevel}" )

    results.append(coolingLevel)
    

    # Update previous values for the next iteration
    prevCoolingLevel = coolingLevel
    prevThrottle = currentThrottle