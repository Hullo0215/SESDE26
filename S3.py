## Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import List

## CSV file reading
battery = pd.read_csv("pvTemp.csv")

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

pv_raw_history = []
pv_filtered_history = []

## Lists to store the results
trends = []
coolingLevels = []

for i in range(len(battery)):

    temp = battery['Temperature'][i]
    pv = battery['PV_value'][i]

    level, filtered_pv, trend, ff_index = S3_SolarRadiationFF_Update(
        batteryTempFiltered=temp,
        pvSolarRadiationRaw=pv,
        pvSolarRadiationRawHistory=pv_raw_history,
        pvSolarRadiationFilteredHistory=pv_filtered_history,
        solarRadMovingAvgWindowSize=5,
        solRadTrendWindowSize=3,
        sampleTimeSec=10,
        kSolRadLevel=0.01,
        kSolRadTrend=0.5,
        ffToMedium=5,
        ffToHigh=10,
        tempMediumLimit=35,
        tempHighLimit=42
    )

    coolingLevels.append(level)
    trends.append(trend)

    print(f"Battery Temp: {temp}°C, PV: {pv} W/m², Filtered PV: {filtered_pv} W/m², Trend: {trend}, FF Index: {ff_index} → Cooling Level: {level}")
    
# Visualization
plt.figure(figsize=(12,8))

# Temperature
plt.subplot(3,1,1)
plt.plot(battery['Temperature'], label="Battery Temperature", color='#8dc6bf')
plt.ylabel("Temperature (°C)")
plt.title("S3 Feedforward Control Visualization")
plt.legend()
plt.grid()

# PV Radiation
plt.subplot(3,1,2)
plt.plot(battery['PV_value'], label="Raw PV Solar Radiation", color='#fdb462')
plt.ylabel("Solar Radiation (W/m²)")
plt.legend()
plt.grid()

# Trend
plt.subplot(3,1,3)
plt.plot(trends, label="Solar Radiation Trend", color='#ef3b2c')
plt.xlabel("Measurement index / Time")
plt.ylabel("Trend")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()