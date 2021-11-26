import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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


def load_dataset():
    x, y = [], []
    for root, _, files in os.walk(HyperParams.feature_path):
        for file in files:
            data = np.load(os.path.join(root, file))
            label = get_label(file)
            x.append(data)
            y.append(label)
    return np.stack(x), np.stack(y)


def get_loaders(normalize=True):
    print("Loading dataset...")
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    if normalize:
        # normalize
        x_train = (x_train-x_train.mean())/x_train.std()
        x_test = (x_test-x_train.mean())/x_train.std()

    train = GTZANDataset(x_train, y_train)
    test = GTZANDataset(x_test, y_test)

    train_loader = DataLoader(
        train, batch_size=HyperParams.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(
        test, batch_size=HyperParams.batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader

if __name__ == '__main__':
    train, test = get_loaders()
    print(len(train), len(test))