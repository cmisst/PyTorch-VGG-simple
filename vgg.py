import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt

import math
from collections import OrderedDict


class VGG(nn.Module):
    # implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3,
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            # TODO: convolutional layer, input channels 3, output channels 8, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 3
            # TODO: convolutional layer, input channels 16, output channels 32, filter size 3
            # TODO: convolutional layer, input channels 32, output channels 32, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 4
            # TODO: convolutional layer, input channels 32, output channels 64, filter size 3
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 5
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(
            # TODO: fully-connected layer (64->64)
            # TODO: fully-connected layer (64->10)
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            print(np.shape(images.numpy()))
            print(np.histogram(labels.numpy()))
            images = images.to(device)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            scores = net.forward(images)
            loss = criterion(scores,labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0

    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    class_count = np.shape(np.unique(testloader.dataset.test_labels))[0]
    matrix = torch.zeros((class_count, class_count))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                matrix[labels[i],predicted[i]] += 1
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return matrix.numpy()



def main(mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torch.utils.data.Subset(
        torchvision.datasets.CIFAR10(root='./data', train=True,
                download=True, transform=transform), range(10000))

    trainloader = torch.utils.data.DataLoader(trainset,  
                    batch_size=100, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)


    if mode=='retrain':
        net = VGG().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train(trainloader, net, criterion, optimizer, device)
        torch.save(net, 'vgg.pt')
    else:
        net = torch.load('vgg.pt')
        print('load')


    plt.imsave('vgg.pdf',test(testloader, net, device))


if __name__== "__main__":
    torch.nn.Module.dump_patches = True
    import sys
    if len(sys.argv)==1:
        main('load')
    else:
        print(sys.argv[1])
        main(sys.argv[1])
