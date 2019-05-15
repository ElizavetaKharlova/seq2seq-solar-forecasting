import pandas as pd
import numpy as np
import datetime
import argparse
from sklearn.preprocessing import StandardScaler
import gzip, pickle

# preprocess data
# 0 mean and unit variance 
def process_weather_load(weather_data, load_data):
    # drop the columns with insufficient values
    weather_data = weather_data.drop(['MW1U', 'Pool_price', 'ANC1', 'Interchange', 'Calgary_hmdx', 'Calgary_wind_chill', 'Calgary_weather', 'Edmonton_hmdx', 'Edmonton_wind_chill', 'Edmonton_weather', 'McMurray_hmdx', 'McMurray_wind_chill', 'McMurray_weather'], axis=1)

    col_names = weather_data.columns
    # replace the NaNs with interpolated values
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.
            
            Input:
            - y, 1d numpy array with possible NaNs
            Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
            to convert logical indices of NaNs to 'equivalent' indices
            Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            """
        
        return np.isnan(y), lambda z: z.nonzero()[0]
    for i in range(1,len(col_names)):
        y = weather_data[col_names[i]]
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        weather_data[col_names[i]] = y


    # get training and test sets 'Interchange',
    X_old = weather_data[['Calgary_temp', 'Calgary_dew_point_temp', 'Calgary_rel_hum', 'Calgary_wind_dir', 'Calgary_wind_spd', 'Calgary_visibility', 'Calgary_stn_press', 'Edmonton_temp', 'Edmonton_dew_point_temp', 'Edmonton_rel_hum', 'Edmonton_wind_dir', 'Edmonton_wind_spd', 'Edmonton_visibility', 'Edmonton_stn_press', 'McMurray_temp', 'McMurray_dew_point_temp', 'McMurray_rel_hum', 'McMurray_wind_dir', 'McMurray_wind_spd', 'McMurray_visibility', 'McMurray_stn_press']]

    X_weather = []
    s = pd.Series([float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')], index=['Calgary_temp', 'Calgary_dew_point_temp', 'Calgary_rel_hum', 'Calgary_wind_dir', 'Calgary_wind_spd', 'Calgary_visibility', 'Calgary_stn_press', 'Edmonton_temp', 'Edmonton_dew_point_temp', 'Edmonton_rel_hum', 'Edmonton_wind_dir', 'Edmonton_wind_spd', 'Edmonton_visibility', 'Edmonton_stn_press', 'McMurray_temp', 'McMurray_dew_point_temp', 'McMurray_rel_hum', 'McMurray_wind_dir', 'McMurray_wind_spd', 'McMurray_visibility', 'McMurray_stn_press'])

    for i in range(len(X_old)):
        X_weather.append(s)
        X_weather.append(s)
        X_weather.append(s)
        X_weather.append(X_old.ix[i])
    X_weather = np.array(X_weather)

    # replace the NaNs with interpolated values
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    for i in range(0,X_weather.shape[1]):
        y = X_weather[:,i]
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        X_weather[:,i] = y

    # Read the load file
    xls = pd.ExcelFile(load_data)
    for i in range(2013,2019):
        f = pd.read_excel(xls, str(i))
        if i > 2013:
            X_load = np.concatenate((X_load,f), axis=0)
        else:
            X_load = f
    # subtract dapa points after 08-16-2018 because the weather data is missing (add later)
    X_load = X_load[:197184]

    print("X", X_weather.shape, "Xgen", X_load.shape)

    # Scale the data
    scalerX_weather, scalerX_load, scalerY = StandardScaler(), StandardScaler(), StandardScaler()
    scalerX_weather.fit(X_weather)
    X_load = X_load[:,3]
    scalerX_load.fit(X_load.reshape(len(X_load),1))
    X_weather = scalerX_weather.transform(X_weather)
    X_load = scalerX_load.transform(X_load.reshape(len(X_load),1))
    return X_weather, X_load

# Plot the data
import matplotlib.pyplot as plt
def plot_data_array(data_arrays):
    
    fig, ax = plt.subplots(data_arrays.shape[1] , sharex=True, figsize=(20, 15))
    
    for array in range(data_arrays.shape[1] ):
        ax[array].plot(data_arrays[:, int(array)], c=[1, 0, 0], label="Data Set")
        
        if array == 0:
            ax[array].legend()

    ax[data_arrays.shape[1] - 1].set_xlabel("time in samples")
    return fig, ax


load_data = './data/20190207-15-min-system-load-MV.xlsx'
weather_data = pd.read_csv('./data/Market_behaviour.csv')
weather, load = process_weather_load(weather_data, load_data)
# Create the dictionary
# Target: position of the target array (load)
# sw_steps: sliding window size in length of samples
# num_steps: the number of steps in the forecast in the lenght of samples
# forecast_steps: length of the forecast in desired sample size
# num_tiles: number of tiles in the probability distribution
specs_dict = {'target': 21,
            'sw_steps': 7*96,
            'num_steps':96,
            'forecast_steps':24,
            'num_tiles':33}

data = np.append(weather, load, axis=1)
fault = np.zeros([data.shape[0],1]) # array describing the faults in the training data
data = np.append(data, fault, axis=1)
plot_data_array(data)
plt.show()

NetworkData = {'dataset':data, 'specs':specs_dict}

f = gzip.open('Lizas_data.pklz','wb')
pickle.dump(NetworkData,f)
f.close()
