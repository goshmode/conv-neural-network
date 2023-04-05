""" James Marcel
    CS5330 Project 5
    April 3 2023
    Loading network from .pth file and evaluating my greek handwriting samples
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

#Class Definitions
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = nn.Conv2d(10,20,kernel_size = 5)
        self.conv2_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,3)

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

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )


#plot test images
def testPlot(model, example_data):
    output = model(example_data)
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap = 'gray', interpolation = 'none')
        plt.title(f"Prediction: {output.data.max(1,keepdim=True)[1][i].item()}")
        plt.xticks([])
        plt.yticks([])
    plt.show()


#plot test images
def imgPlot(model, examples, names):
    output = model(examples)
    print(output)
    
    fig = plt.figure()
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.tight_layout()
        plt.imshow(examples[i][0], cmap = 'gray', interpolation = 'none')
        plt.title(f"Prediction: {names[output.data.max(1,keepdim=True)[1][i].item()]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main(argv):
    torch.backends.cudnn.enabled = False
    network = NeuralNet()
    network.load_state_dict(torch.load("greekmodel.pth"))
    network.eval()


    # DataLoader for handwritten Greek data set
    greek_hand = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( "greekhand/",
                                          transform = torchvision.transforms.Compose( 
                                            [torchvision.transforms.ToTensor(),
                                            GreekTransform(),
                                            torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 12,
        shuffle = False )

    print("Model Load Successful.\n")

    #preparing test data
    examples = enumerate(greek_hand)
    print("examples: ",examples)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    #plotting 3x3 grid of examples
    #testPlot(network, example_data)
    
    
    examples = enumerate(greek_hand)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    names = {0: "Alpha", 1: "Beta", 2: "Gamma"}
    imgPlot(network,example_data,names)




if __name__ == "__main__":
    main(sys.argv)