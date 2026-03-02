## Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import time

## CSV file reading
batteryTemp = pd.read_csv("batteryTemp.csv")

## Thresholds and parameters
tempLowLimit = 25
tempMediumLimit = 35
tempHighLimit = 45
tempRiseThreshold = 0.5

movingAvgWindowSize = 5
trendWindowSize = 3

## Max history size for both moving average and trend analysis
maxHistorySize = max(movingAvgWindowSize, trendWindowSize + 1)

## Lists to store history and results
rawHistory = [] # batteryTempRawHistory
filteredHistory = [] 
coolingLevels = []

## Function to update cooling level based on temperature, moving average, and trend analysis
def S2_trendmonitor_Update(temp, prevCoolingLevel):

    global rawHistory, filteredHistory

# Raw history update --> if the history exceeds the maximum size, remove the oldest entry
    if len(rawHistory) == maxHistorySize:
        rawHistory.pop(0)
# Add the new temperature to the raw history
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
# Calculate the moving average
    filteredTemp = sumVal / count if count > 0 else 0

# Fifo history update for trend analysis --> if the filtered history exceeds the maximum size, remove the oldest entry
    if len(filteredHistory) == maxHistorySize:
        filteredHistory.pop(0)
# Add the new filtered temperature to the filtered history
    filteredHistory.append(filteredTemp)

# Trend calculation
    if len(filteredHistory) >= trendWindowSize + 1: 

        lastIndex = len(filteredHistory) - 1
        pastIndex = lastIndex - trendWindowSize 

        tempTrend = (filteredHistory[lastIndex] -
                     filteredHistory[pastIndex]) / trendWindowSize
    else:
        tempTrend = 0

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

## Lists to store the results
filteredTemps = []
trends = []
prevCoolingLevel = 1

## Simulate the control loop for each temperature reading and apply the S2_trendmonitor_Update function
for temp in batteryTemp['Temperature']:

    fTemp, trend, level = S2_trendmonitor_Update(temp, prevCoolingLevel)
    
# Store the results in the respective lists
    filteredTemps.append(fTemp)
    trends.append(trend)
    coolingLevels.append(level)
    prevCoolingLevel = level # Update the previous cooling level for the next iteration

    print(f"Temp={temp:.2f} → Filtered={fTemp:.2f} "
          f"Trend={trend:.3f} → Level={level}")
