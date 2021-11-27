import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Hyper_parameters import HyperParams


class GTZANDataset(Dataset):
    '''
    Custom torch dataloader
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


def get_label(filename):
    '''
    filename example: classical.00000.wav
    '''
    genre = filename.split(".")[0]
    label = HyperParams.genres.index(genre)
    return label


def load_dataset(name, flatten=False):
    x, y = [], []
    path = os.path.join(HyperParams.feature_path, name)
    for root, _, files in os.walk(path):
        for file in files:
            data = np.load(os.path.join(root, file))
            label = get_label(file)
            x.append(data)
            y.append(label)
    x, y = np.stack(x), np.stack(y)
    if flatten:
        x = x.reshape(x.shape[0], -1)
    return x, y

def get_ndarrays(test=False, flatten=False):
    if not test:
        x_train, y_train = load_dataset("train", flatten)
        x_valid, y_valid = load_dataset("valid", flatten)
        x_test, y_test = load_dataset("test", flatten)

        # normalize
        mean = np.mean(x_train)
        std = np.std(x_train)
        x_train = (x_train-mean)/std
        x_valid = (x_valid-mean)/std
        x_test = (x_test-mean)/std

        return x_train, y_train, x_test, y_test, x_valid, y_valid
    else:
        x_valid, y_valid = load_dataset("valid", flatten)
        x_test, y_test = load_dataset("test", flatten)

        # normalize
        mean = np.mean(x_valid)
        std = np.std(x_valid)
        x_valid = (x_valid-mean)/std
        x_test = (x_test-mean)/std

        return x_valid, y_valid, x_test, y_test, x_test, y_test

def get_loaders(test=False):
    if not test:
        x_train, y_train = load_dataset("train")
        x_valid, y_valid = load_dataset("valid")
        x_test, y_test = load_dataset("test")

        # normalize
        mean = np.mean(x_train)
        std = np.std(x_train)
        x_train = (x_train-mean)/std
        x_valid = (x_valid-mean)/std
        x_test = (x_test-mean)/std

        train = GTZANDataset(x_train, y_train)
        valid = GTZANDataset(x_valid, y_valid)
        test = GTZANDataset(x_test, y_test)

        train_loader = DataLoader(
            train, batch_size=HyperParams.batch_size, shuffle=True, drop_last=False)
        valid_loader = DataLoader(
            valid, batch_size=HyperParams.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(
            test, batch_size=HyperParams.batch_size, shuffle=False, drop_last=False)

        return train_loader, valid_loader, test_loader
    else:
        x_valid, y_valid = load_dataset("valid")

        # normalize
        mean = np.mean(x_valid)
        std = np.std(x_valid)
        x_valid = (x_valid-mean)/std

        valid = GTZANDataset(x_valid, y_valid)

        valid_loader = DataLoader(
            valid, batch_size=HyperParams.batch_size, shuffle=False, drop_last=False)

        return valid_loader, valid_loader, valid_loader