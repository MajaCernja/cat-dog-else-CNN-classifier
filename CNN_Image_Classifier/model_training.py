import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CNN_Image_Classifier.cnn import ImageClassifier
from helpers.data_augmentation import train_data_augmentation, test_data_augmentation, ImgDataset, data_splitter


def training_run(epochs, model, device, criterion, optimizer, train_loader, test_loader):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        print('Epoch: ', epoch + 1)
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y.long())
            loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(y_pred.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
            train_loss += loss.item()
            print('Batch: {} | Time: {:.2f}s | Train Acc: {:.4f} Loss: {:.4f}'.format(
            i + 1, time.time() - epoch_start_time, train_acc / (64*(i+1)), train_loss / (i+1)))

        train_acc /= len(train_loader.dataset)
        train_loss /= (i + 1)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            for i, (test_x, test_y) in enumerate(test_loader):
                test_x, test_y = test_x.to(device), test_y.to(device)
                test_pred = model(test_x)
                batch_loss = criterion(test_pred, test_y.long())
                test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == test_y.cpu().data.numpy())
                test_loss += batch_loss.item()

            test_acc /= len(test_loader.dataset)
            test_loss /= (i+1)
            test_accuracies.append(test_acc)
            test_losses.append(test_loss)

        print('Epoch: {} | Time: {:.2f}s | Train Acc: {:.4f} Loss: {:.4f} | Test Acc: {:.4f} Loss {:.4f}'.format(
            epoch + 1, time.time() - epoch_start_time, train_acc, train_loss, test_acc, test_loss))

    return train_accuracies, train_losses, test_accuracies, test_losses


