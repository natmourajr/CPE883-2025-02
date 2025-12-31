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

import sys
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import pywt
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

sys.path.append("/home/felipe/doutorado/CPE883-2025-02/experiments/CEEMDAN/")
from utils import corr_per_scale

base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset/'


class Collector:
    # Read the data. The files are in csv format.
    # The frequency of the data is 0.1 Hz (10 seconds per register).


    def __init__(self, base_path):
        self.base_path = base_path
        pass

    
    def read_data(
            self, file, serie_size, window_size, predict_steps, batch_size=None,
            year=2017, freq_transform=True, transform_method='wavelets', scales=np.arange(0.5, 10, 0.1),
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

        if freq_transform:
            coefficients = create_freq_transform(signal, transform_method=transform_method, scales=scales, plot=False)
            # coefs_analysis(coefficients)
            # corr_per_scale(coefficients, signal, plot=False)
            X, y = create_sliding_windows_and_targets(signal, coefficients, window_size=window_size, predict_steps=predict_steps)
            dataset = CoefficientsDataset(X, y)

        else:
            X, y = create_sliding_windows_and_targets(signal, None, window_size=window_size, predict_steps=predict_steps)
            dataset = PowerSeriesDataset(X, y)

        return dataset


    def create_date_feats(self, df):

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        return df


def create_sliding_windows_and_targets(signal, coefficients=None, window_size=100, predict_steps=1):
    X_windows = []
    y_targets = []

    # Se for 1D, mantém 1D
    if coefficients is None:
        data = signal  # shape: (length,)
    else:
        data = coefficients.T  # shape: (length, num_features)

    n_samples = data.shape[0]

    for i in range(n_samples - window_size - predict_steps + 1):
        window = data[i : i + window_size]          # shape: (window_size,) ou (window_size, num_features)
        target = signal[i + window_size + predict_steps - 1]  # pega apenas 1 valor futuro
        X_windows.append(window)
        y_targets.append(target)

    X_windows = np.array(X_windows)  # (num_samples, window_size) ou (num_samples, window_size, num_features)
    y_targets = np.array(y_targets)[:, np.newaxis]  # (num_samples, 1)

    return X_windows, y_targets


def create_freq_transform(signal, log=False, plot=False, transform_method='wavelets', scales=np.arange(0.5, 10, 0.1)):
    """
    Aplica transformada de frequência em uma série temporal completa.
    
    Args:
        signal (np.array): série temporal 1D completa
        log (bool): aplicar escala logarítmica nos plots
        plot (bool): exibir plots de signal + espectrograma
        transform_method (str): 'wavelets' ou 'stft'
        scales (np.array): escalas para CWT (wavelets)
    
    Returns:
        np.array: coeficientes da transformada (num_scales x signal_length)
    """

    if transform_method == 'wavelets':
        wavelet = 'mexh'  # wavelet usada
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet)  # (num_scales, signal_length)

        if plot:
            time = np.arange(len(signal))
            fig, ax = plt.subplots(2, figsize=(12, 6))
            ax[0].plot(time, signal)
            ax[0].set_title('Original Signal')
            ax[0].set_xlabel('Time')
            ax[0].set_ylabel('Amplitude')

            pcm = ax[1].pcolormesh(time, scales, coefficients, shading='auto', cmap='jet')
            ax[1].set_ylabel('Scale')
            ax[1].set_xlabel('Time')
            ax[1].set_title('Scalogram (CWT)')
            fig.colorbar(pcm, ax=ax[1], label='Magnitude')
            ax[1].invert_yaxis()

            if log:
                ax[1].set_yscale('log')
                ax[1].invert_yaxis()

            plt.tight_layout()
            plt.show()

        return coefficients

    elif transform_method == 'stft':
        # 10 minutos de intervalo da série de energia eólica coletada
        fs = 1/600  # Hz
        f, t, Zxx = stft(signal, fs=fs, window='hann', nperseg=288, noverlap=144)  
        # 288 pontos = 2 dias de dados (com 10min cada amostra)

        spectrogram = np.abs(Zxx)

        if plot:
            time = np.arange(len(signal))
            fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

            # --- Sinal original ---
            ax[0].plot(time, signal)
            ax[0].set_title('Sinal Original')
            ax[0].set_ylabel('Amplitude')

            # --- STFT / Espectrograma ---
            pcm = ax[1].pcolormesh(t, f*86400, spectrogram, shading='gouraud', cmap='jet')  # f*86400: ciclos/dia
            ax[1].set_title('STFT - Espectrograma em Ciclos por Dia')
            ax[1].set_xlabel('Tempo [amostras]')
            ax[1].set_ylabel('Frequência [ciclos/dia]')
            ax[1].set_ylim(0, 72)  # até 10 ciclos/dia
            fig.colorbar(pcm, ax=ax[1], label='Magnitude')

            plt.tight_layout()
            plt.show()

        return spectrogram

    else:
        raise ValueError("transform_method must be 'wavelets' or 'stft'")


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

    def create_data(self, df, months=[1, 2], look_back=6, data_partition=0.8):


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

    def create_lookback_data(self, dataset, look_back=6):

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