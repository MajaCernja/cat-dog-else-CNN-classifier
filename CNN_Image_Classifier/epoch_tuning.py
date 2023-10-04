import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CNN_Image_Classifier.cnn import ImageClassifier
from CNN_Image_Classifier.model_training import training_run
from helpers.data_augmentation import train_data_augmentation, test_data_augmentation, ImgDataset, data_splitter
from matplotlib import pyplot as plt

batch_size = 64
num_epochs = 20
all_results = {}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
model = ImageClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_x, train_y, test_x, test_y = data_splitter('../processed_img_data.npy')
train_x = train_x[:1280]
train_y = train_y[:1280]
test_x = test_x[:512]
test_y = test_y[:512]
train_set = ImgDataset(train_x, train_y, transform=train_data_augmentation())
test_set = ImgDataset(test_x, test_y, transform=test_data_augmentation())
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


def epoch_tuning(epochs=num_epochs):
    results = training_run(epochs, model, device, criterion, optimizer, train_loader, test_loader)
    torch.save(model, 'cat_dog_classifier.pth')
    print('-----------------------------------')

    return results


if __name__ == '__main__':
    all_results = epoch_tuning()
    x = [i for i in range(1, num_epochs + 1)]
    train_accuracies = all_results[0]
    train_losses = all_results[1]
    test_accuracies = all_results[2]
    test_losses = all_results[3]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_accuracies, label='Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.savefig('epoch_tuning_train_accuracy.png')

    plt.subplot(1, 2, 2)
    plt.plot(x, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.savefig('epoch_tuning_test_accuracy.png')

    plt.show()
