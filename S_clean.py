## Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import json

## CSV file reading
input_filename = "common_strategy_input_P5_power_mode.csv"
data = pd.read_csv(input_filename)

## JSON configuration reading
with open("config.json") as f:
    config = json.load(f)


## GLOBAL CONFIGURATION
hysteresis_time = config["hysteresis_time"]
nominalEnergy = config["nominal_energy"]
horizonLength = config["horizon"]
strategy = config["strategy"]

thermal = config["thermal_model"]
control = config["control"]
limits = config["limits"]
cooling = config["cooling"]
weights = config["mpc_weights"]
trend = config["trend_monitor"]

alphaG = thermal["alphaG"]
kThrottle = thermal["kThrottle"]
gamma = thermal["gamma"]
internalResistance = thermal["internalResistance"]
modelParam_a = thermal["modelParam_a"]
modelParam_b = thermal["modelParam_b"]

coolingPower = cooling["power"]
coolingEnergy = cooling["energy"]

wEnergy = weights["wEnergy"]
wTemp = weights["wTemp"]
wSwitch = weights["wSwitch"]
wTerminal = weights["wTerminal"]
wFFBias = weights["wFFBias"]

tempLow = limits["tempLow"]
tempRef = limits["tempRef"]
tempMedium = limits["tempMedium"]
tempHigh = limits["tempHigh"]
tempMax = limits["tempMax"]

propulsionPowerGain = control["propulsionPowerGain"]
throttleWeightInLoad = control["throttleWeightInLoad"]
throttleRampEco = control["throttleRampEco"]
throttleRampPower = control["throttleRampPower"]

solarMovingAvgWindow = control["solarMovingAvgWindow"]
solRadMovingAvgWindowSize = control["solRadMovingAvgWindowSize"]
solRadTrendWindowSize = control["solRadTrendWindowSize"]

kSolRadLevel = control["kSolRadLevel"]
kSolRadTrend = control["kSolRadTrend"]
ffToMedium = control["ffToMedium"]
ffToHigh = control["ffToHigh"]

tempRiseThreshold = trend["tempRiseThreshold"]
movingAvgWindowSize = trend["movingAvgWindowSize"]
trendWindowSize = trend["trendWindowSize"]

temp_max = config["safety_checks"]["temp_max"]


## Extracting relevant columns from the data
pv = data["PV_value"].values
solar = data["solar"].values
ambient = data["ambient_temp"].values
throttle = data["throttle"].values
op_mode = data["operation_mode"].values
sim_batteryTemp = data["battery_temp"].values


## Clamping function definition
def clamp(x, xmin, xmax):
    return max(xmin, min(x, xmax))


## Safety checks function to ensure water presence and prevent overheating
def check_coolant(waterIn: bool):
    if not waterIn:
        raise RuntimeError("ERROR: No coolant in system!")

def check_overheat(temp: float, temp_max: float):
    if temp >= temp_max:
        raise RuntimeError("ERROR: Emergency overheat!")

## Initial conditions
soc = 0.8
sim_batteryTemp = data["battery_temp"].values[0]

rawHistory = []
filteredHistory = []
pv_raw_history = []
pv_filtered_history = []

coolingLevels = []
soc_history = []
temp_history = []


## S5 logic: Hybrid Feedforward + MPC Control
def S5_HybridFF_MPC_Update(
    batteryTempFiltered: float,
    prevCoolingLevel: int,
    pvSolarRadiationRaw: float,
    pvSolarRadiationRawHistory: List[float],
    pvSolarRadiationFilteredHistory: List[float],
    solRadMovingAvgWindowSize: int,
    solRadTrendWindowSize: int,
    currentThrottle: float,
    prevThrottle: float,
    operationMode: str,
    forecastSolarRadiation: List[float],
    forecastAmbientTemp: List[float],
    hysteresis_time_sec: float,
    horizonLength: int,
    kSolRadLevel: float,
    kSolRadTrend: float,
    ffToMedium: float,
    ffToHigh: float,
    alphaG: float,
    kThrottle: float,
    throttleWeightInLoad: float,
    throttleRampEco: float,
    throttleRampPower: float,
    internalResistance: float,
    modelParam_a: float,
    modelParam_b: float,
    coolingPowerLevel: List[float],
    coolingEnergyLevel: List[float],
    wEnergy: float,
    wTemp: float,
    wSwitch: float,
    wTerminal: float,
    wFFBias: float,
    tempRef: float,
    tempMediumLimit: float,
    tempHighLimit: float,
    tempMaxLimit: float,
    solarMovingAvgWindow: int
):
    forecastThrottle = [0] * horizonLength 
    Qgen = [0] * horizonLength
    I_ch_rawHistory = [0] * horizonLength

    maxHistorySize = max(solRadMovingAvgWindowSize, solRadTrendWindowSize + 1)

    if len(pvSolarRadiationRawHistory) == maxHistorySize:
        pvSolarRadiationRawHistory.pop(0)
    pvSolarRadiationRawHistory.append(pvSolarRadiationRaw)

    lastIndex = len(pvSolarRadiationRawHistory) - 1
    startIndex = max(0, lastIndex - solRadMovingAvgWindowSize + 1)

    sumVal = 0
    count = 0

    # Calculate the sum and count of valid entries for the moving average
    for i in range(startIndex, lastIndex + 1):
        sumVal += pvSolarRadiationRawHistory[i]
        count += 1

    if count > 0:
        pvSolarRadiationFilteredRaw = sumVal / count
    else:
        pvSolarRadiationFilteredRaw = 0

    if len(pvSolarRadiationFilteredHistory) == maxHistorySize:
        pvSolarRadiationFilteredHistory.pop(0)
    pvSolarRadiationFilteredHistory.append(pvSolarRadiationFilteredRaw)

    if len(pvSolarRadiationFilteredHistory) >= solRadTrendWindowSize + 1:
        lastIndex = len(pvSolarRadiationFilteredHistory) - 1
        pastIndex = lastIndex - solRadTrendWindowSize

        dG = pvSolarRadiationFilteredHistory[lastIndex] - pvSolarRadiationFilteredHistory[pastIndex]
        dT = solRadTrendWindowSize * hysteresis_time_sec
        solRadTrend = dG / dT if dT > 0 else 0
    else:
        solRadTrend = 0

    ffIndex = kSolRadLevel * pvSolarRadiationFilteredRaw + kSolRadTrend * solRadTrend

    if ffIndex < ffToMedium:
        ffLevel = 1
    elif ffIndex < ffToHigh:
        ffLevel = 2
    else:
        ffLevel = 3
    
    ## Throttle prediction
    dThrottle = currentThrottle - prevThrottle
    if operationMode == "eco":
        rampLimit = throttleRampEco
    else:
        rampLimit = throttleRampPower

    forecastThrottle[0] = clamp(currentThrottle, 0, 1)

    for k in range(1, horizonLength):
        delta =clamp(dThrottle, -rampLimit, rampLimit)
        forecastThrottle[k] = forecastThrottle[k-1] + delta
        forecastThrottle[k] = clamp(forecastThrottle[k], 0, 1)

    ## Solar radiation processing and feedforward calculation
    for k in range(horizonLength):
        I_ch_rawHistory[k] = alphaG * forecastSolarRadiation[k]
        sumVal = 0
        count = 0
        startIndex = max(0, k - solarMovingAvgWindow + 1)

        for i in range(startIndex, k+1):
            sumVal += I_ch_rawHistory[i]
            count += 1
        
        I_ch = sumVal / count if count > 0 else 0
        loadThrottle = kThrottle * forecastThrottle[k]
        loadTotal = (1 - throttleWeightInLoad) * I_ch + throttleWeightInLoad * loadThrottle
        Qgen[k] = internalResistance * loadTotal**2

    ## Optimization
    bestCost = float('inf')
    bestLevel = prevCoolingLevel

    for u_candidate in range(4):
        if abs(u_candidate - prevCoolingLevel) > 1:
            continue
        Temp_sim = batteryTempFiltered
        cost = 0
        u_prev = prevCoolingLevel
        for k in range(horizonLength):
            Temp_sim = Temp_sim + hysteresis_time_sec * (
                modelParam_a * (forecastAmbientTemp[k] - Temp_sim)
                + modelParam_b * Qgen[k]
                - coolingPowerLevel[u_candidate]
            )

            if Temp_sim > tempMaxLimit:
                cost = float('inf')
                break
            
            if Temp_sim > tempRef:
                overTemp = Temp_sim - tempRef
            else:
                overTemp = 0
            
            ffBiasCost = wFFBias * (u_candidate - ffLevel)**2
            
            stageCost = (
                wEnergy * coolingEnergyLevel[u_candidate]
                + wTemp * overTemp**2
                + wSwitch * (u_candidate - u_prev)**2
                + ffBiasCost
            )

            cost += stageCost
            u_prev = u_candidate

        ## Terminal cost for final temperature deviation
        if cost < float('inf'):
            overTemp = max(0, Temp_sim - tempRef)
            cost += wTerminal * overTemp**2

        if cost < bestCost:
            bestCost = cost
            bestLevel = u_candidate
    
    ## Feedback
    if batteryTempFiltered >= tempHighLimit:
        bestLevel = 3
    elif batteryTempFiltered >= tempMediumLimit:
        bestLevel = max(bestLevel, 2)

    return bestLevel, ffLevel, ffIndex, solRadTrend, bestCost


## S4 logic: Predictive Control
def S4_predictive_Update(
    batteryTempFiltered,
    prevCoolingLevel,
    currentThrottle,
    prevThrottle,
    operationMode,
    forecastSolarRadiation,
    forecastAmbientTemp,
    hysteresis_time_sec,
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
    ## Debug print statement for throttle change
    ##print(f"dThrottle: {dThrottle:.4f}")
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
        sumVal = 0
        count = 0
        startIndex = k - solarMovingAvgWindow + 1

        if startIndex < 0:
            startIndex = 0
        
        for i in range(startIndex, k+1):
            sumVal += I_ch_rawHistory[i]
            count += 1

        I_ch = sumVal / count if count > 0 else 0

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
            Temp_sim = Temp_sim + hysteresis_time_sec * (modelParam_a * (forecastAmbientTemp[k] - Temp_sim) + modelParam_b * Qgen[k] - coolingPowerLevel[u_candidate])
        
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

## S3 logic: Solar Radiation Feedforward Control
def S3_SolarRadiationFF_Update(
    batteryTempFiltered: float,
    pvSolarRadiationRaw: float,
    pvSolarRadiationRawHistory: List[float],
    pvSolarRadiationFilteredHistory: List[float],
    solarRadMovingAvgWindowSize: int,
    solRadTrendWindowSize: int,
    hysteresis_time_sec: float,
    kSolRadLevel: float,
    kSolRadTrend: float,
    ffToMedium: float,
    ffToHigh: float,
    tempMediumLimit: float,
    tempHighLimit: float
):
    # Ensuring we have enough history for both moving average and trend analysis
    maxHistorySize = max(solarRadMovingAvgWindowSize, solRadTrendWindowSize + 1)

    # FIFO history update for solar radiation raw data
    if len(pvSolarRadiationRawHistory) == maxHistorySize:
        pvSolarRadiationRawHistory.pop(0)
    pvSolarRadiationRawHistory.append(pvSolarRadiationRaw)

    # Moving average calculation for solar radiation
    lastIndex = len(pvSolarRadiationRawHistory) - 1
    startIndex = max(0, lastIndex - solarRadMovingAvgWindowSize + 1)
    sumVal = 0
    count = 0

    # Calculate the sum and count of valid entries for the moving average
    for i in range(startIndex, lastIndex + 1):
        sumVal += pvSolarRadiationRawHistory[i]
        count += 1

    if count > 0:
        pvSolarRadiationFilteredRaw = sumVal / count
    else:
        pvSolarRadiationFilteredRaw = 0

    # Filtered history storing
    if len(pvSolarRadiationFilteredHistory) == maxHistorySize:
        pvSolarRadiationFilteredHistory.pop(0)
    pvSolarRadiationFilteredHistory.append(pvSolarRadiationFilteredRaw)

    # Trend calculation for solar radiation
    if len(pvSolarRadiationFilteredHistory) >= solRadTrendWindowSize + 1:
        lastIndex = len(pvSolarRadiationFilteredHistory) - 1
        pastIndex = lastIndex - solRadTrendWindowSize
        dG = pvSolarRadiationFilteredHistory[lastIndex] - pvSolarRadiationFilteredHistory[pastIndex]
        dT =solRadTrendWindowSize * hysteresis_time_sec
        solRadTrend = dG / dT if dT > 0 else 0
    else:
        solRadTrend = 0

    # Feedforward load calculation based on the filtered solar radiation and its trend
    ffIndex = kSolRadLevel * pvSolarRadiationFilteredRaw + kSolRadTrend * solRadTrend

    # Cooling level decision logic based on the feedforward index and temperature thresholds
    if ffIndex < ffToMedium:
        ffLevel = 1
    elif ffIndex < ffToHigh:
        ffLevel = 2
    else:
        ffLevel = 3

    if batteryTempFiltered >= tempHighLimit:
        coolingLevel = 3
    elif batteryTempFiltered >= tempMediumLimit:
        if ffLevel < 2:
            coolingLevel = 2
        else:
            coolingLevel =ffLevel
    else:
        coolingLevel = ffLevel

    return coolingLevel, pvSolarRadiationFilteredRaw, solRadTrend, ffIndex

## S2 logic: Trend Monitoring Control
def S2_trendmonitor_Update(temp, prevCoolingLevel, tempLowLimit, tempMediumLimit, tempHighLimit, tempRiseThreshold, movingAvgWindowSize, trendWindowSize, filteredHistory, rawHistory):

    maxHistorySize = max(movingAvgWindowSize, trendWindowSize + 1)

    if len(rawHistory) == maxHistorySize:
        rawHistory.pop(0)
    rawHistory.append(temp)

# Moving average calculation
    lastIndex = len(rawHistory) - 1
    startIndex = max(0, lastIndex - movingAvgWindowSize + 1) 
    sumVal = 0
    count = 0

# Calculate the sum and count of valid entries for the moving average
    for i in range(startIndex, lastIndex + 1):
        sumVal += rawHistory[i]
        count += 1
    filteredTemp = sumVal / count if count > 0 else 0
    ## Debug print statements for monitoring the filtering process
    ##print(f"Raw Temp: {temp:.2f}, Filtered Temp: {filteredTemp:.2f}")

# Fifo history update for trend analysis --> if the filtered history exceeds the maximum size, remove the oldest entry
    if len(filteredHistory) == maxHistorySize:
        filteredHistory.pop(0)
    filteredHistory.append(filteredTemp)

# Trend calculation
    if len(filteredHistory) >= trendWindowSize + 1: 

        lastIndex = len(filteredHistory) - 1
        pastIndex = lastIndex - trendWindowSize 

        tempTrend = (filteredHistory[lastIndex] -
                     filteredHistory[pastIndex]) / trendWindowSize
    else:
        tempTrend = 0
    ## Debug print statements for trend monitoring
    ##print(f"Temp Trend: {tempTrend:.2f}")

## Cooling level decision logic based on the filtered temperature and trend
    if filteredTemp >= tempHighLimit:
        coolingLevel = 3

    elif filteredTemp >= tempMediumLimit:
        if tempTrend >= tempRiseThreshold:
            if prevCoolingLevel < 3:
                coolingLevel = prevCoolingLevel + 1
            else:
                coolingLevel = 3
        else:
            coolingLevel = 2

    elif tempTrend >= tempRiseThreshold:
        coolingLevel = 2

    else:
        coolingLevel = 1

    return filteredTemp, tempTrend, coolingLevel

## S1 logic: Reactive Control
def S1_reactive_Update(temp, tempLowLimit, tempMediumLimit, tempHighLimit):
    if temp < tempLowLimit:
        return 0
    elif temp < tempMediumLimit:
        return 1
    elif temp < tempHighLimit:
        return 2
    else:
        return 3

## Main control loop
i = 0
prevCoolingLevel = 0
prevThrottle = throttle[0]

while i < len(data):

    waterIn = True

    try:
        check_coolant(waterIn)
        check_overheat(sim_batteryTemp, tempMax)
    except RuntimeError as e:
        print(e)
        break

    temp = sim_batteryTemp
    pv_val = data["PV_value"].iloc[i]

    if strategy == "S1":
        level = S1_reactive_Update(temp, tempLow, tempMedium, tempHigh)

    elif strategy == "S2":
        fTemp, trend, level = S2_trendmonitor_Update(
            temp,
            prevCoolingLevel,
            tempLow,
            tempMedium,
            tempHigh,
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
            solRadMovingAvgWindowSize,
            solRadTrendWindowSize,
            hysteresis_time,
            kSolRadLevel,
            kSolRadTrend,
            ffToMedium,
            ffToHigh,
            tempMedium,
            tempHigh
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
                hysteresis_time_sec=hysteresis_time,
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
                hysteresis_time_sec=hysteresis_time,
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


    soc_history.append(soc)
    temp_history.append(sim_batteryTemp)
    coolingLevels.append(level)

    prevCoolingLevel = level
    prevThrottle = throttle[i]

    i += 1

output = pd.DataFrame({
    "SOC_sim": soc_history,
    "BatteryTemp_sim": temp_history,
    "CoolingLevel": coolingLevels
})

base_name = input_filename.replace("common_strategy_input_", "").replace(".csv", "")
filename = f"{strategy}_{base_name}_output.csv"
output.to_csv(filename, index=False)