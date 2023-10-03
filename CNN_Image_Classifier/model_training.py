import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from CNN_Image_Classifier.cnn import ImageClassifier
from helpers.data_augmentation import train_data_augmentation, test_data_augmentation, img_data_augmentation, data_splitter

batch_size = 1
epochs = 1

model = ImageClassifier()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_x, train_y, test_x, test_y = data_splitter('../processed_img_data.npy')
print('Train X shape: ', train_x.shape, 'Train Y shape', train_y.shape)
print(train_y)
train_set = img_data_augmentation(train_x, train_y, transform=train_data_augmentation())
test_set = img_data_augmentation(test_x, test_y, transform=test_data_augmentation())


def main(epochs=epochs, model=model, device=device, criterion=criterion, optimizer=optimizer, batch_size=batch_size, train_set=train_set, test_set=test_set):
    for epoch in range(epochs):
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        print('Epoch: ', epoch + 1)
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            print(data)
            print(len(data))
            optimizer.zero_grad()
            x, y = data[0].to(device), data[1].to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y.long())
            loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(y_pred.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
            train_loss += loss.item()
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                test_x, test_y = data[0].to(device), data[1].to(device)

                test_pred = model(test_x)
                batch_loss = criterion(test_pred, test_y.long())
                test_acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == test_y.cpu().data.numpy())
                test_loss += batch_loss.item()

        print('Epoch: {} | Time: {}s | Train Acc: {} Loss: {} | Test Acc: {} Loss {}'.format
              (epoch + 1,
               time.time() - epoch_start_time,
               train_acc / len(train_loader.dataset),
               train_loss / len(train_loader.dataset),
               test_acc / len(test_loader.dataset),
               test_loss / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
