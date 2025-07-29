import sys
import os

sys.path.append(f'{os.environ.get("path3W","../../../")}'+'3W-toolkit')

import toolkit as tk

import numpy as np
import pandas as pd
from pickle import load, dump

class Loader3W(object):
    def __init__(self):
        self.real_instances, self.simulated_instances, self.drawn_instances = tk.get_all_labels_and_files()
        self.stats = {'wells': {}, 'ids': set(), 'features': {}}

    def load_real_instance(self,idx=0):
        return tk.load_instance(self.real_instances[idx])
    
    def load_simulated_instance(self, idx=0):
        return tk.load_instance(self.simulated_instances[idx])
    
    def load_drawn_instance(self, idx=0):
        return tk.load_instance(self.drawn_instances[idx])
    
    def get_ids_from_wells_with_event_type(self, events_ids: list):
        ids_set = set()
        for i in range(len(self.real_instances)):
            dataset: pd.DataFrame = self.load_real_instance(i)
            if dataset['state'].isin(events_ids).any():
                ids_set.add(i)
        self.stats['ids'] = ids_set

    def extract_stats(self, feat:list):
        for f in feat:
            df = pd.DataFrame()
            for i in self.stats['ids']:
                dataset = self.load_real_instance(i)
                df = pd.concat([df, dataset.loc[:, ['well',f]]])
            self.stats['features'][f] = {
                'mean': df[f].mean(),
                'std': df[f].std(),
                'min': df[f].min(),
                'max': df[f].max(),
                'count': df[f].count()
                }
            wells = df['well'].unique()
            for well in wells:
                df_well = df[df['well'] == well]
                if well not in self.stats['wells']:
                    self.stats['wells'][well] = {}
                self.stats['wells'][well][f]={
                    'mean': np.nan_to_num(df_well[f].mean(), copy=True, nan=0),
                    'std': np.nan_to_num(df_well[f].std(), copy=True, nan=0),
                    'min': np.nan_to_num(df_well[f].min(), copy=True, nan=0),
                    'max': np.nan_to_num(df_well[f].max(), copy=True, nan=0),
                    'relative_max': np.nan_to_num(df_well[f].max(), copy=True, nan=0) / self.stats['features'][f]['max'],
                    'count': df_well[f].count()
                    }
    
    def save_stats(self, filename='stats.pkl'):
        with open(filename, 'wb') as f:
            dump(self.stats, f)

    def load_stats(self, filename='stats.pkl'):
        with open(filename, 'rb') as f:
            self.stats = load(f)

    def preprocess(self):
        for i in self.stats['ids']:
            dataset = self.load_real_instance(i)[['well']+list(self.stats['features'].keys())+['state']]
            for well in dataset['well'].unique():
                if well not in self.stats['wells']:
                    raise ValueError(f'Well {well} not found in stats')
                for f in self.stats['features']:
                    dataset.loc[dataset['well']==well, f] = dataset.loc[dataset['well']==well, f].fillna(self.stats['wells'][well][f]['mean'])
                    dataset.loc[dataset['well']==well, f] -= self.stats['wells'][well][f]['mean']
                    if self.stats['wells'][well][f]['std'] != 0:
                        dataset.loc[dataset['well']==well, f] /= self.stats['wells'][well][f]['std']
                    dataset.loc[dataset['well']==well, f'{f}_relative_max'] = 0.0 if np.isnan(self.stats['wells'][well][f]['relative_max']) else \
                        self.stats['wells'][well][f]['relative_max']
                dataset.loc[dataset['well']==well, 'state'] = dataset.loc[dataset['well']==well, 'state'].fillna(0)
                
            for s in range(10):
                dataset[f'state-{s}'] = (dataset['state'] == s).astype(int)
            yield dataset
                    

        


if __name__ == "__main__":
    loader = Loader3W
    print("Real Instances:", loader.load_real_instance())