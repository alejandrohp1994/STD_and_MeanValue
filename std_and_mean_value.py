import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

INPUT_FILE = "Florida_temp_mean_1895_2017.csv"

FOLDER_LOCATION = "Enter Location here"
FILENAME = "Enter File Name here"

OUTPUT_FILE = FOLDER_LOCATION + FILENAME + ".csv"




plt.style.use('seaborn-whitegrid')

pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 60)

plt.rcParams['figure.figsize'] = (15, 10)

temp= pd.read_csv(INPUT_FILE)

temp_year = temp.iloc[:,0].values #int64
temp_temp = temp.iloc[:,1].values #float
temp_anom = temp.iloc[:,2].values #float

#looks for nan value on list and changes it for its neighbors

def changing_nan(arr): 
    index = 0
    for index in range(len(arr)):
        if np.isnan(arr[index]):
            arr[index] = (arr[index-1] + arr[index + 1])/2
            arr[index] = round(arr[index], 2)
    
def mean(arr):
    return round(np.mean(arr), 2)

def std(arr):
    return round(np.std(arr), 2)

changing_nan(temp_temp)
changing_nan(temp_anom)

mean_temp = mean(temp_temp)
mean_anom = mean(temp_anom)
std_temp = std(temp_temp)
std_anom = std(temp_anom)

#printing the mean and std

def printing_mean_std():
    print('The mean value of the temperature is: {}'.format(mean_temp))
    print('The mean value of the anomaly is: {}'.format(mean_anom))
    print('The standard deviation value of the temperature is: {}'.format(std_temp))
    print('The standard deviation value of the anomaly is: {}'.format(std_anom))


#first plot showing the temperatures

def firstplot(): 
    fig, first_his = plt.subplots()
    first_his.plot(temp_year, temp_temp)
    first_his.set_xlabel('Year', fontsize = 20)
    first_his.set_ylabel('Temperature', fontsize = 20)
    
#second plot showing the histogram

def secondplot(): 
    fid, ax = plt.subplots(1, 1)
    ax.hist(temp_temp, bins = len(temp_temp))


#importing libraries for graphs
    
import scipy
import scipy.ndimage

#kernel with 5 elements

kernel_5 = [0.2,0.2,0.2,0.2,0.2]     

Smoothed_signal = np.correlate(temp_temp, kernel_5, mode = 'same')
Smoothed_signal1 = scipy.ndimage.correlate(temp_temp, kernel_5, mode = 'nearest')

def thirdplot():   
    fig, diagram = plt.subplots(3, 1, constrained_layout = True)
    
    diagram[0].plot(temp_year, temp_temp)
    diagram[0].set_xlabel('Year', fontsize = 20)
    diagram[0].set_ylabel('Temperature', fontsize = 20)
    diagram[0].set_title('Raw Graph', fontsize = 20)
    
    diagram[1].plot(temp_year, Smoothed_signal)
    diagram[1].set_xlabel('Year', fontsize = 20)
    diagram[1].set_ylabel('Temperature', fontsize = 20)
    diagram[1].set_title('NumPy Correlate Graph', fontsize = 20)

    diagram[2].plot(temp_year, Smoothed_signal1)
    diagram[2].set_xlabel('Year', fontsize = 20)
    diagram[2].set_ylabel('Temperature', fontsize = 20)
    diagram[2].set_title('SciPy Correlate Graph', fontsize = 20)

# kernel with 21 elements

kernel_21 = np.zeros(21, dtype = 'float64')
kernel_21.fill(1/21) 

conv = np.convolve(temp_temp, kernel_21, mode = 'same')
conv1 = scipy.ndimage.convolve(temp_temp, kernel_21, mode = 'nearest')

def fourthplot():   
    fig, diagram = plt.subplots(3, 1, constrained_layout = True)
    
    diagram[0].plot(temp_year, temp_temp)
    diagram[0].set_xlabel('Year', fontsize = 20)
    diagram[0].set_ylabel('Temperature', fontsize = 20)
    diagram[0].set_title('Raw Graph', fontsize = 20)
    
    diagram[1].plot(temp_year, conv)
    diagram[1].set_xlabel('Year', fontsize = 20)
    diagram[1].set_ylabel('Temperature', fontsize = 20)
    diagram[1].set_title('NumPy Convolve Graph', fontsize = 20)

    diagram[2].plot(temp_year, conv1)
    diagram[2].set_xlabel('Year', fontsize = 20)
    diagram[2].set_ylabel('Temperature', fontsize = 20)
    diagram[2].set_title('SciPy Convolve Graph', fontsize = 20)
        

#exporting data as data file 

def exporting():
    data_updated = {}
    data_updated['Year'] = temp_year
    data_updated['Temperature'] = temp_temp
    data_updated['Anomaly'] = temp_anom
    data_updated['Smoothed Signal'] = Smoothed_signal
    
    finalized_table = pd.DataFrame(data = data_updated)
    
    finalized_table.to_csv(OUTPUT_FILE, index = False)

if __name__ == '__main__':
    
    #print(temp)
    #print(temp[:5])
    
    print(temp_year)
    print(temp_temp)
    #print(temp_anom)

    #printing_mean_std()
    
    firstplot()
    secondplot()
    thirdplot()
    fourthplot()
    
    exporting()