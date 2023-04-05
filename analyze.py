""" James Marcel
    CS5330 Project 5
    April 3 2023
    Analyzing network
"""

#import statements
import torch 
import torchvision 
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image, ImageChops
from skimage import io
import pandas as pd
import os
import cv2 as cv

#Class Definitions
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = nn.Conv2d(10,20,kernel_size = 5)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        #conv1 followed by max pool with relu
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #conv2 followed by max pool with relu
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #flattening tensor
        x = x.view(-1,320)
        #fully connected linear layer with 50 nodes
        x = F.relu(self.fc1(x))
        #not sure why this is here
        x = F.dropout(x, training = self.training)
        #final fully-connected linear layer with 10 ndes then log_softmax
        x = self.fc2(x)
        return F.log_softmax(x)





#plot test images
def filterPlot(data):

    fig = plt.figure()
    for i in range(10):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(data[i,0], cmap = 'viridis', interpolation = 'none')
        plt.title(f"Filter {i}")
        plt.xticks([])
        plt.yticks([])
    plt.show()


#plot test images
def alterPlot(data,test):

    fig = plt.figure()
    adjust = 0
    for i in range(10):
        adjust += 1
        plt.subplot(5,4,i + adjust)
        plt.tight_layout()
        plt.imshow(data[i,0], cmap = 'gray', interpolation = 'none')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(5,4,i + adjust + 1)
        plt.tight_layout()
        kernel = np.array(data[i,0])
        image = np.array(test[0,0])
        filt = cv.filter2D(image,-1,kernel) 
        plt.imshow(filt, cmap = 'gray', interpolation = 'none')
        plt.xticks([])
        plt.yticks([])
    plt.show()



def main(argv):
    torch.backends.cudnn.enabled = False
    network = NeuralNet()
    network.load_state_dict(torch.load("model.pth"))
    network.eval()

    print(network)

    conv1Weights = []

    for i in range(10):
        conv1Weights.append(network.conv1.weight[i,0])
        #print("Weight ", i)
        #print(network.conv1.weight[i].shape)
        #print(network.conv1.weight[i,0], "\n")
    print(network.conv1.weight.shape)

    #loading test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/',train = False, download = True,
                                transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                ])),
                                batch_size = 64, shuffle = False)

    #preparing test data
    examples = enumerate(test_loader)
    print("examples: ",examples)
    batch_idx, (example_data, example_targets) = next(examples)


    with torch.no_grad():
        #filterPlot(network.conv1.weight)
        alterPlot(network.conv1.weight, example_data)







if __name__ == "__main__":
    main(sys.argv)