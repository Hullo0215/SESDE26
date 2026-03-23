import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List

data = pd.read_csv("S4data.csv")
solar = data["solar"].values
ambient = data["ambient_temp"].values
throttle = data["throttle"].values
battery_temp = data["battery_temp"].values
op_mode = data["operation_mode"].values


def clamp(x, xmin, xmax):
    return max(xmin, min(x, xmax))


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
    sampleTimeSec: float,
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
        dT = solRadTrendWindowSize * sampleTimeSec
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
        sumval = 0
        count = 0
        startIndex = max(0, k - solarMovingAvgWindow + 1) if startIndex > 0 else 0

        for i in range(startIndex, k):
            sumval += I_ch_rawHistory[i]
            count += 1
        
        I_ch = sumval / count if count > 0 else 0
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
            Temp_sim = Temp_sim + sampleTimeSec * (
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

## Simulation loop for S5
pv_raw_history = []
pv_filtered_history = []
results = []
ffLevels = []
ffIndices = []
trends = []

prevCoolingLevel = 0
prevThrottle = throttle[0]

horizonLength = 10
sampleTimeSec = 1

for t in range(len(data) - horizonLength):
    currentThrottle = throttle[t]
    operationMode = op_mode[t]
    batteryTempFiltered = battery_temp[t]

    forecastSolar = solar[t:t+horizonLength]
    forecastAmbient = ambient[t:t+horizonLength]

    coolingLevel, ffLevel, ffIndex, trend, bestCost = S5_HybridFF_MPC_Update(
        batteryTempFiltered=batteryTempFiltered,
        prevCoolingLevel=prevCoolingLevel,
        pvSolarRadiationRaw=solar[t],
        pvSolarRadiationRawHistory=pv_raw_history,
        pvSolarRadiationFilteredHistory=pv_filtered_history,
        solRadMovingAvgWindowSize=5,
        solRadTrendWindowSize=3,
        currentThrottle=currentThrottle,
        prevThrottle=prevThrottle,
        operationMode=operationMode,
        forecastSolarRadiation=forecastSolar,
        forecastAmbientTemp=forecastAmbient,
        sampleTimeSec=sampleTimeSec,
        horizonLength=horizonLength,
        kSolRadLevel=0.01,
        kSolRadTrend=0.5,
        ffToMedium=5,
        ffToHigh=10,
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
        wFFBias=2.0,
        tempRef=30,
        tempMediumLimit=35,
        tempHighLimit=42,
        tempMaxLimit=50,
        solarMovingAvgWindow=3
    )

    results.append(coolingLevel)
    ffLevels.append(ffLevel)
    ffIndices.append(ffIndex)
    trends.append(trend)

    prevCoolingLevel = coolingLevel
    prevThrottle = currentThrottle

    print(f"t={t} | Temp={batteryTempFiltered:.2f} | Throttle={currentThrottle:.2f} | Cooling={coolingLevel} | FF_Level={ffLevel} | FF_Index={ffIndex:.2f} | Trend={trend:.4f} | Best_Cost={bestCost:.2f}" )