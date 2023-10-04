import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


def train_data_augmentation():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])


def test_data_augmentation():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y.numpy())
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def data_splitter(data_file, split_ratio=0.8):
    dataset = np.load(data_file, allow_pickle=True)
    print(dataset.shape)

    X_np = np.array([i[0] for i in dataset])
    X = torch.from_numpy(X_np).view(-1, 128, 128)
    X = X / 255.0

    y_np = np.array([i[1] for i in dataset])
    y = torch.from_numpy(y_np)

    y_labels = []
    for i in range(len(y)):
        real = torch.argmax(y[i])
        y_labels.append(real.item())

    print('Total datapoints: ', len(y_labels))
    print('Cat datapoints: ', y_labels.count(0))
    print('Dog datapoints: ', y_labels.count(1))
    y_labels = torch.tensor(y_labels)

    train_size = int(split_ratio * len(dataset))
    train_x = X[:train_size]
    train_y = y_labels[:train_size]
    test_x = X[train_size:]
    test_y = y_labels[train_size:]

    return train_x, train_y, test_x, test_y
