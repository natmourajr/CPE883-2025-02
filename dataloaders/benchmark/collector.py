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
import numpy as np
import seaborn as sns
import logging
import warnings
import sys
sys.path.append('stac')
# import nonparametric_tests as stac
from matplotlib import pyplot as plt
from time import time
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

logging.getLogger('tsfresh').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore')

class Collector3W(Dataset):
    def __init__(self, data_path, undesirable_event_code=1,
                 sample_size_default=60,
                 sample_size_normal_period=5,
                 max_samples_per_period=15,
                 downsample_rate=60,
                 real=True, simulated=False, drawn=False,
                 train=True,
                 transform=None,
                 disable_progressbar=True):
        
        self.data_path = Path(data_path)
        self.undesirable_event_code = undesirable_event_code
        self.sample_size_default = sample_size_default
        self.sample_size_normal_period = sample_size_normal_period
        self.max_samples_per_period = max_samples_per_period
        self.downsample_rate = downsample_rate
        self.real = real
        self.simulated = simulated
        self.drawn = drawn
        self.train = train
        self.transform = transform
        self.disable_progressbar = disable_progressbar
        
        self.columns = ['timestamp','P-PDG','P-TPT','T-TPT','P-MON-CKP','T-JUS-CKP','P-JUS-CKGL','T-JUS-CKGL','QGL','class']
        self.normal_class_code = 0

        # Busca as instâncias que possuem o evento indesejável
        self.instances = self.get_instances_with_undesirable_event()
        # Carrega e dá downsample
        self.df_instances, _ = self.load_and_downsample_instances()

        # Amostras e labels
        self.df_samples = pd.DataFrame()
        self.df_labels = pd.DataFrame()
        self.sample_id = 0

        # Extrai amostras de treino ou teste
        if self.train:
            self.df_samples, self.df_labels, self.sample_id = self.extract_samples_train(self.df_instances, 
                                                                                        self.df_samples, 
                                                                                        self.df_labels, 
                                                                                        self.sample_id)
        else:
            self.df_samples, self.df_labels, self.sample_id = self.extract_samples_test(self.df_instances,
                                                                                       self.df_samples,
                                                                                       self.df_labels,
                                                                                       self.sample_id)

    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, idx):
        sample_id = self.df_labels.iloc[idx]['instance']
        y = self.df_labels.iloc[idx]['y']
        sample_df = self.df_samples[self.df_samples['id'] == sample_id].drop(columns=['id', 'timestamp'])
        X = torch.tensor(sample_df.values, dtype=torch.float32)
        if self.transform:
            X = self.transform(X)
        return X, y

    # Funções internas baseadas no seu código original

    def class_and_file_generator(self):
        for class_path in self.data_path.iterdir():
            if class_path.is_dir():
                class_code = int(class_path.stem)
                for instance_path in class_path.iterdir():
                    if instance_path.suffix == '.csv':
                        if (self.simulated and instance_path.stem.startswith('SIMULATED')) or \
                           (self.drawn and instance_path.stem.startswith('DRAWN')) or \
                           (self.real and (not instance_path.stem.startswith('SIMULATED')) and
                            (not instance_path.stem.startswith('DRAWN'))):
                            yield class_code, instance_path

    def get_instances_with_undesirable_event(self):
        instances = pd.DataFrame(self.class_and_file_generator(),
                                 columns=['class_code', 'instance_path'])
        idx = instances['class_code'] == self.undesirable_event_code
        return instances.loc[idx].reset_index(drop=True)

    def load_instance(self, instance_path):
        try:
            well, instance_id = instance_path.stem.split('_')
            df = pd.read_csv(instance_path, sep=',', header=0)
            assert (df.columns == self.columns).all(), f'invalid columns in file {instance_path}: {list(df.columns)}'
            return df
        except Exception as e:
            raise Exception(f'error reading file {instance_path}: {e}')

    def load_and_downsample_instances(self):
        df_instances = pd.DataFrame()
        instance_id = 0
        for _, row in self.instances.iterrows():
            _, instance_path = row
            df = self.load_instance(instance_path).iloc[::self.downsample_rate, :]
            df['instance_id'] = instance_id
            instance_id += 1
            df_instances = pd.concat([df_instances, df])
        df_instances['source'] = 'train' if self.train else 'test'
        return df_instances.reset_index(drop=True), instance_id

    def extract_samples_train(self, df, df_samples_train, df_y_train, sample_id):
        instance = df['instance_id'].iloc[0]
        ols = list(df['class'])
        set_ols = set(int(ol) for ol in ols if not pd.isna(ol))

        df_vars = df.drop(['source', 'class'], axis=1).fillna(0)

        # Normal period
        if self.normal_class_code in set_ols:
            f_idx = ols.index(self.normal_class_code)
            l_idx = len(ols) - 1 - ols[::-1].index(self.normal_class_code)
            max_samples = l_idx - f_idx + 1 - self.sample_size_normal_period
            if max_samples > 0:
                num_samples = min(self.max_samples_per_period, max_samples)
                step_max = 1 if num_samples == max_samples else (max_samples - 1) // (self.max_samples_per_period - 1)
                step = min(self.sample_size_normal_period, step_max)
                for idx in range(num_samples):
                    f_idx_c = l_idx - self.sample_size_normal_period + 1 - (num_samples - 1 - idx) * step
                    l_idx_c = f_idx_c + self.sample_size_normal_period
                    df_sample = df_vars.iloc[f_idx_c:l_idx_c, :]
                    df_sample.insert(0, 'id', sample_id)
                    df_samples_train = pd.concat([df_samples_train, df_sample], ignore_index=True)
                    df_y_train = pd.concat([df_y_train, pd.DataFrame([{'instance': instance, 'y': self.normal_class_code}])], ignore_index=True)

                    sample_id += 1

        # Transient period
        transient_code = self.undesirable_event_code + 100
        if transient_code in set_ols:
            f_idx = ols.index(transient_code)
            f_idx = max(f_idx - (self.sample_size_default - 1), 0)
            l_idx = len(ols) - 1 - ols[::-1].index(transient_code)
            max_samples = l_idx - f_idx + 1 - self.sample_size_default
            if max_samples > 0:
                num_samples = min(self.max_samples_per_period, max_samples)
                step_max = 1 if num_samples == max_samples else (max_samples - 1) // (self.max_samples_per_period - 1)
                step = min(np.inf, step_max)
                for idx in range(num_samples):
                    f_idx_c = f_idx + idx * step
                    l_idx_c = f_idx_c + self.sample_size_default
                    df_sample = df_vars.iloc[int(f_idx_c):int(l_idx_c), :]
                    df_sample.insert(0, 'id', sample_id)
                    df_samples_train = pd.concat([df_samples_train, df_sample], ignore_index=True)
                    df_y_train = pd.concat([df_y_train, pd.DataFrame([{'instance': instance, 'y': transient_code}])], ignore_index=True)
                    sample_id += 1

        # In-regime period
        if self.undesirable_event_code in set_ols:
            f_idx = ols.index(self.undesirable_event_code)
            f_idx = max(f_idx - (self.sample_size_default - 1), 0)
            l_idx = len(ols) - 1 - ols[::-1].index(self.undesirable_event_code)
            l_idx = min(l_idx + (self.sample_size_default - 1), len(ols) - 1)
            max_samples = l_idx - f_idx + 1 - self.sample_size_default
            if max_samples > 0:
                num_samples = min(self.max_samples_per_period, max_samples)
                step_max = 1 if num_samples == max_samples else (max_samples - 1) // (self.max_samples_per_period - 1)
                step = min(self.sample_size_default, step_max)
                for idx in range(num_samples):
                    f_idx_c = f_idx + idx * step
                    l_idx_c = f_idx_c + self.sample_size_default
                    df_sample = df_vars.iloc[int(f_idx_c):int(l_idx_c), :]
                    df_sample.insert(0, 'id', sample_id)
                    df_samples_train = pd.concat([df_samples_train, df_sample], ignore_index=True)
                    df_y_train = pd.concat([df_y_train, pd.DataFrame([{'instance': instance, 'y': self.undesirable_event_code}])], ignore_index=True)
                    sample_id += 1

        return df_samples_train, df_y_train, sample_id

    def extract_samples_test(self, df, df_samples_test, df_y_test, sample_id):
        instance = df['instance_id'].iloc[0]
        ols = list(df['class'].fillna(method='ffill'))
        df_vars = df.drop(['source', 'class'], axis=1).fillna(0)

        f_idx = 0
        l_idx = len(df) - 1
        max_samples = l_idx - f_idx + 1 - self.sample_size_default
        if max_samples > 0:
            num_samples = min(3 * self.max_samples_per_period, max_samples)
            step_max = 1 if num_samples == max_samples else (max_samples - 1) // (3 * self.max_samples_per_period - 1)
            step = min(np.inf, step_max)
            for idx in range(num_samples):
                f_idx_c = f_idx + idx * step
                l_idx_c = f_idx_c + self.sample_size_default
                df_sample = df_vars.iloc[int(f_idx_c):int(l_idx_c), :]
                df_sample.insert(0, 'id', sample_id)
                df_samples_test = df_samples_test.append(df_sample)
                df_y_test = df_y_test.append({'instance': instance, 'y': ols[int(l_idx_c)]}, ignore_index=True)
                sample_id += 1

        return df_samples_test, df_y_test, sample_id


# Exemplo de uso

from torch.utils.data import DataLoader

dataset = Collector3W(data_path='3w_dataset/data', undesirable_event_code=1, train=True)

loader = DataLoader(dataset, batch_size=16, shuffle=True)

for X, y in loader:
    print(X.shape)  # (batch_size, time_steps, features)
    print(y)
    break