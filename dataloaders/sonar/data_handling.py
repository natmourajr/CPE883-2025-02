import numpy as np

import itertools
import random
import torch 

class_map = {
    "ClassA": 0,
    "ClassB": 1,
    "ClassC": 2,
    "ClassD": 3,
}

class LoroCV:
    def __init__(self, n_splits, shuffle=True, window_size=None, overlap=None, test_overlap=None, random_seed=42):
        self.shuffle = shuffle
        self.n_splits = n_splits
        self.window_size = window_size
        self.overlap = overlap
        self.test_overlap = test_overlap
        self.seed = random_seed

    def split(self, lofar_data):
        ship_sets, ship_cycles = self._prepare_cycler(lofar_data)
        for _ in range(self.n_splits):
            test_ship_list = list()
            train_ship_list = list()
            for cls_name, run in lofar_data.items():
                cls_ship_set = ship_sets[cls_name]
                cls_ship_cycle = ship_cycles[cls_name]

                test_ship = next(cls_ship_cycle)
                train_ships = cls_ship_set - {test_ship}

                test_ship_list.extend([(cls_name, test_ship)])
                train_ship_list.extend([(cls_name, ship) for ship in train_ships])
            # print(test_ship_list)
            # print(train_ship_list)
            # print()
            # print()
            X_train, y_train, coords_train = self.flatten(lofar_data, train_ship_list, test=False)
            X_test, y_test, coords_test = self.flatten(lofar_data, test_ship_list, test=True)

            yield X_train, y_train, X_test, y_test, coords_train, coords_test

    def flatten(self, lofar_data, ship_list, test):
        trgt, data, coords = list(), list(), list()
        for cls_name, ship in ship_list:
            Sxx, f, t = lofar_data[cls_name][ship]

            # Sxx windowing, must check if its working properly
            if test and (self.test_overlap is not None):
                overlap = self.test_overlap
            else:
                overlap = self.overlap
                
            if self.window_size and overlap:
                H = self.window_size
                O = self.overlap
                step = H - O
                windows = [Sxx[i:i+H] for i in range(0, Sxx.shape[0] - H + 1, step)]
                Sxx = np.array(windows)

            trgt.append(class_map[cls_name] * np.ones(Sxx.shape[0]))
            data.append(Sxx)
            # mesh = np.meshgrid(f, t)
            frequency_time_pairs = np.array(np.meshgrid(f, t)).T.reshape(-1, 2)
            coords.append(frequency_time_pairs)
        
        trgt = np.concatenate(trgt)
        data = np.concatenate(data, axis=0)
        coords = np.concatenate(coords, axis=0)
        return data, trgt, coords

    def _prepare_cycler(self, lofar_data):
        shuffle = self.shuffle

        ship_sets = dict()
        ship_cycles = dict()
        for cls_name, run in lofar_data.items():
            keys_list = list(run.keys())  # Convert the keys to a list
            if shuffle:
                random.Random(self.seed).shuffle(keys_list)  # Shuffle the list of keys with a fixed seed

            ship_sets[cls_name] = set(keys_list)  # Convert the shuffled list to a set
            ship_cycles[cls_name] = itertools.cycle(keys_list)  # Create a cycle from the shuffled list

        return ship_sets, ship_cycles



class CustomDataloader:
    def __init__(self, data, target, is2d=False, device='cpu'):
        if is2d:
            # data = torch.tensor(data.reshape(data.shape[0], 1, data.shape[1], data.shape[2]), dtype=torch.float32).to(device)
            data = torch.tensor(data, dtype=torch.float32).to(device)
        else:
            data = torch.tensor(data.reshape(data.shape[0], -1), dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.long).to(device)

        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
class DeepOnetDataLoader:
    def __init__(self, data, target, coords, is2d=False, device='cpu'):
        # data = torch.tensor(data.reshape(data.shape[0], -1), dtype=torch.float32).to(device)
        data = torch.tensor(data, dtype=torch.float32, device=device)
        target = torch.tensor(target, dtype=torch.long).to(device)
        coords = torch.tensor(coords, dtype=torch.float32).to(device)

        self.data = data
        self.target = target
        self.coords = coords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.coords[idx][:]