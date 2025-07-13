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
from sklearn.preprocessing import StandardScaler

base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset/'


class Collector:
    # Read the data. The files are in csv format.
    # The frequency of the data is 0.1 Hz (10 seconds per register).


    def __init__(self, base_path):
        self.base_path = base_path
        pass

    
    def read_data(self, file, year=2017):
        
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
    
        return new_data

    def create_date_feats(self, df):

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        return df


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