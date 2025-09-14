"""
Collect the CEEMDAN dataset

Considerations:
    - 


version: 0.0.1
date: 02/07/2025

copyright Copyright (c) 2025

References:
[1]
"""

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import pywt
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset/'


class Collector:
    # Read the data. The files are in csv format.
    # The frequency of the data is 0.1 Hz (10 seconds per register).


    def __init__(self, base_path):
        self.base_path = base_path
        pass

    
    def read_data(
            self, file, serie_size, window_size, predict_steps, batch_size=None,
            year=2017, freq_transform=True, scales=np.arange(0.5, 10, 0.1),
            ):
        
        if file=='final_la_haute_R0711.csv':
            P_col = 'P_avg'
            date_col = 'Date_time'
        if file=='T1.csv':
            P_col = 'Date/Time'
            date_col = 'LV ActivePower (kW)'


        df = pd.read_csv(os.path.join(base_path, file))

        if file=='T1.csv':
            df = df.dropna(axis = 0, how ='any')

        df['Date'] = pd.to_datetime(df[date_col])
        df = self.create_date_feats(df)
        new_data=df[['Month','Year','Date', P_col]]
        new_data=new_data[new_data.Year == year]

        signal = df['P_avg'].values[0:serie_size]
        time = df['Date'].values[0:serie_size]

        X, y = create_sliding_windows_and_targets(signal, window_size=window_size, predict_steps=predict_steps)

        if freq_transform:
            coefficients_list = []
            coefficients_list = create_freq_transform(X)
            dataset = CoefficientsDataset(coefficients_list, y)

            return dataset

        else:
            # 2. Dividir em treino/teste
            # split = int(0.8 * len(X))
            # X_train, y_train = X[:split], y[:split]
            # X_test, y_test = X[split:], y[split:]

            # X_train shape: (num_samples, window_size)
            # Normalização x
            # scaler_X = StandardScaler()
            # X_train_scaled = scaler_X.fit_transform(X_train)  # shape permanece (num_samples, window_size)
            # X_test_scaled = scaler_X.transform(X_test)

            # Normalização y
            # y_train = y_train.reshape(-1,1)
            # y_test = y_test.reshape(-1,1)
            # scaler_y = StandardScaler()
            # y_train_scaled = scaler_y.fit_transform(y_train)
            # y_test_scaled = scaler_y.transform(y_test)

            # Criar datasets (normalmente o dataset adiciona a dimensão da feature)
            # train_dataset = PowerSeriesDataset(X_train_scaled, y_train_scaled)
            # test_dataset = PowerSeriesDataset(X_test_scaled, y_test_scaled)

            # 6. Criar DataLoaders
            # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            dataset = PowerSeriesDataset(X, y)

            return dataset


    def create_date_feats(self, df):

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        return df


def create_sliding_windows_and_targets(signal, window_size=100, predict_steps=1):
    X_windows = []
    y_targets = []

    for i in range(len(signal) - window_size - predict_steps + 1):
        window = signal[i : i + window_size]
        target = signal[i + window_size : i + window_size + predict_steps]
        X_windows.append(window)
        y_targets.append(target)

    return np.array(X_windows), np.array(y_targets)


def create_freq_transform(
        X, log=False, plot=False, transform_method='wavelets', scales=np.arange(0.5, 10, 0.1)):

    coefficients_list = []

    for x_window in X:
        
        # Assign the signal series of the window
        signal = x_window
        # Create a time index (Need to change to the real date index)
        time = np.arange(0, len(signal))
        
        if transform_method=='wavelets':
            # Parameters for CWT
            # Choose Haar wavelet for CWT
            wavelet = 'mexh'

            # Perform CWT
            coefficients, frequencies = pywt.cwt(signal, scales, wavelet)

            # coefficients.shape = (num_scales, signal_length)

            if plot:
                fig, ax = plt.subplots(2, figsize=(12, 6))

                # Plot the signal
                ax[0].plot(time, signal)
                ax[0].set_title('Original P_avg Signal')
                ax[0].set_xlabel('Time Index')
                ax[0].set_ylabel('Amplitude')

                # Plot scalogram
                pcm = ax[1].pcolormesh(time, scales, coefficients, shading='auto', cmap='jet')
                ax[1].set_ylabel('Scale')
                ax[1].set_xlabel('Time Index')
                ax[1].set_title('Scalogram (CWT) of P_avg signal')

                # Add colorbar for scalogram
                fig.colorbar(pcm, ax=ax[1], label='Magnitude')

                # Invert the y axis. Generally in scalogram its commom to see the scales inverted
                # and the frequencies as it is
                ax[1].invert_yaxis()

                if log:
                    pass
                    # ax[1].set_yscale('log')
                    # Invert the y-axis because extent flips it
                    # ax[1].invert_yaxis()  # to keep scale increasing from bottom to top


                plt.tight_layout()
                plt.show()

            coefficients_list.append(coefficients)

        if transform_method=='stft':
                f, t, Zxx = stft(signal, window='hann', nperseg=10, noverlap=4)
                spectrogram = np.abs(Zxx)

                if plot:
                    plt.figure(figsize=(12, 5))
                    plt.pcolormesh(t, f, spectrogram, shading='gouraud')
                    plt.title('STFT Magnitude Spectrogram')
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.colorbar(label='Magnitude')
                    plt.tight_layout()
                    plt.show()
                coefficients_list.append(spectrogram)

    return np.array(coefficients_list)


# Coefficients Dataset
class CoefficientsDataset(Dataset):
    def __init__(self, coefficients, targets):
        self.X = torch.tensor(coefficients, dtype=torch.float32).unsqueeze(1)  # shape: (B, 1, S, T)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PowerSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X: array-like de shape (num_amostras, seq_len) ou (num_amostras, seq_len, num_features)
        y: array-like de shape (num_amostras, horizon) ou (num_amostras, horizon, num_features)
        """
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (num_amostras, seq_len, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CreateTrainTest():

    def __init__(self):
        pass

    def create_data(self, df, months=[1, 2], look_back=1, data_partition=0.8):


        data1=df.loc[df['Month'].isin(months)]
        data1=df.reset_index(drop=True)
        data1=df.dropna()
        data1=df['P_avg']
        # datas_wind=pd.DataFrame(datas)
        dfs=data1
            
        
        datasetss2=pd.DataFrame(dfs)
        datasets=datasetss2.values
        
        train_size = int(len(datasets) * data_partition)
        test_size = len(datasets) - train_size
        train, test = datasets[0:train_size], datasets[train_size:len(datasets)]

        trainX, trainY = self.create_lookback_data(train, look_back)
        testX, testY = self.create_lookback_data(test, look_back)

        X_train=pd.DataFrame(trainX)
        Y_train=pd.DataFrame(trainY)
        X_test=pd.DataFrame(testX)
        Y_test=pd.DataFrame(testY)
        sc_X = StandardScaler()
        sc_y = StandardScaler()

        X= sc_X.fit_transform(X_train)
        y= sc_y.fit_transform(Y_train)
        X1= sc_X.fit_transform(X_test)
        y1= sc_y.fit_transform(Y_test)
        y=y.ravel()
        y1=y1.ravel()

        return X, y, X1, y1

    def create_lookback_data(self, dataset, look_back=1):

        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)




def test_1():

    file = 'final_la_haute_R0711.csv'

    collector = Collector(base_path)
    df = collector.read_data(file)

    create_train_test = CreateTrainTest()
    # X and y are train samples. X1, y1 are test samples.
    X, y, X1, y1 = create_train_test.create_data(df, months=[1, 2], look_back=8, data_partition=0.8)

    print(df.iloc[0:1])

    file = 'T1.csv'

    collector = Collector(base_path)
    df = collector.read_data(file)

    print(df.iloc[0:1])

    print("Collector and Preprocess Working")



if __name__=='__main__':
    df = test_1()
    print(df)