## Scenario 1: Reactive Control

## Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

## batteryTemp: current temperature of the battery
batteryTemp = pd.read_csv('batteryTemp.csv')
print(batteryTemp)

## Thresholds
tempLowLimit = 25
tempMediumLimit = 35
tempHighLimit = 45

coolingLevels = []

def S1_reactive_Update(temp):
    if temp < tempLowLimit:
        return 0
    elif temp < tempMediumLimit:
        return 1
    elif temp < tempHighLimit:
        return 2
    else:
        return 3
    

for i in range(len(batteryTemp)):
    temp = batteryTemp['Temperature'][i]
    level = S1_reactive_Update(temp)
    coolingLevels.append(level)
    print(f"Battery Temp: {temp}°C, Cooling Level: {level}")

## Add cooling levels to the batteryTemp DataFrame
batteryTemp['CoolingLevel'] = coolingLevels
print(batteryTemp)

## Visualization
plt.figure(figsize=(12,6))

plt.plot(batteryTemp['Temperature'], label="Battery Temperature", color='#8dc6bf')

plt.axhline(tempLowLimit, linestyle='--', label="Low threshold", color='#fb836f')
plt.axhline(tempMediumLimit, linestyle='--', label="Medium threshold", color='#c1549c')
plt.axhline(tempHighLimit, linestyle='--', label="High threshold", color='#7e549f')


plt.xlabel("Measurement index / Time")
plt.ylabel("Value")
plt.title("Reactive Battery Cooling Control - S1")

plt.legend()
plt.grid()
plt.show()