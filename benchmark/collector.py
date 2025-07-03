"""
Collect the benchmark datasets

Datasets:

- W3: https://github.com/ricardovvargas/3w_dataset.git

Considerations:
    - 


version: 0.0.1
date: 02/07/2025

copyright Copyright (c) 2025

References:
[1]

"""

import pandas as pd


base_path = 'data/Down_Hole_Safety_Valve_Spurious_Closure/1.csv'


class Collector:
    # Read the data. The files are in csv format.
    # The frequency of the data is 0.1 Hz (10 seconds per register).


    def __init__(self, base_path):
        self.base_path = base_path

    
    def read_data(self):

        # Example path to a sample CSV file
        sample_file_base_path = base_path

        # Load the CSV file
        df = pd.read_csv(sample_file_base_path)

        # Show the first few rows
        print(df.head())
        pass

        return df


def test():

    collector = Collector(base_path)
    df = collector.read_data()



if __name__=='__main__':
    df = test()
