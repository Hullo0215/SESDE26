## Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import List
import keyboard
import time

## CSV file reading
battery = pd.read_csv("pvTemp.csv")

## Sample time for the control loop (in seconds)
sample_time = 1.0

## Thresholds and parameters
rawHistory = []
filteredHistory = []

pv_raw_history = []
pv_filtered_history = []

prevCoolingLevel = 1
coolingLevels = []

## Variable for mode switching
mode = "S1"

## S3 logic: Solar Radiation Feedforward Control
def S3_SolarRadiationFF_Update(
    batteryTempFiltered: float,
    pvSolarRadiationRaw: float,
    pvSolarRadiationRawHistory: List[float],
    pvSolarRadiationFilteredHistory: List[float],
    solarRadMovingAvgWindowSize: int,
    solRadTrendWindowSize: int,
    sampleTimeSec: float,
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
        dT =solRadTrendWindowSize * sampleTimeSec
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
    print(f"Raw Temp: {temp:.2f}, Filtered Temp: {filteredTemp:.2f}")

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
    print(f"Temp Trend: {tempTrend:.2f}")

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

while i < len(battery):
    
    last_mode = mode

    # Mode selection based on keyboard input
    if keyboard.is_pressed("1"):
        mode = "S1"
    elif keyboard.is_pressed("2"):
        mode = "S2"
    elif keyboard.is_pressed("3"):
        mode = "S3"

    # If the mode has changed, print the new mode
    if mode != last_mode:
        print(f"Mode → {mode}")
        last_mode = mode

    temp = battery["Temperature"].iloc[i]
    pv = battery["PV_value"].iloc[i]

    if mode == "S1":

        level = S1_reactive_Update(temp,25,35,45)

    elif mode == "S2":

        fTemp, trend, level = S2_trendmonitor_Update(
            temp,
            prevCoolingLevel,
            25,
            35,
            45,
            0.5,
            5,
            3,
            filteredHistory,
            rawHistory
        )

    elif mode == "S3":

        level, filtered_pv, trend, ff_index = S3_SolarRadiationFF_Update(
            temp,
            pv,
            pv_raw_history,
            pv_filtered_history,
            5,
            3,
            sample_time,
            0.01,
            0.5,
            5,
            10,
            35,
            42
        )

    coolingLevels.append(level)

    print(
        f"Step {i} | Mode {mode} | Temp {temp:.2f} | PV {pv:.2f} | Cooling {level}"
    )

    prevCoolingLevel = level

    i += 1

    time.sleep(sample_time)