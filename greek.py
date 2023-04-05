""" James Marcel
    CS5330 Project 5
    April 3 2023
    Transfer Learning on Greek Letters
"""

#import statements
import torch 
import torchvision 
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
import sys

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


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )


#some functions

#plotting model performance
def plot(train_counter, train_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color = "green")
    #print(len(test_counter), len(test_losses))
    plt.legend(["Train Loss"], loc = "upper right")
    plt.xlabel("Number of Training Examples Seen")
    plt.ylabel("Negative Log Likelihood Loss")
    plt.show()


#training function
def train(epoch,network,optimizer,train_loader,log_interval,train_losses, train_counter):
    network.train()
    correct = 0
    for batch_idx, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1) * len(train_loader.dataset)))
    return correct

    
#runs handwriting number model on greek letters and determines which class they belong to
def main(argv):

    #setting a few variables
    n_epochs = 500
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 9

    torch.backends.cudnn.enabled = False
    torch.manual_seed(10)

    network = NeuralNet()
    network.load_state_dict(torch.load("model.pth"))
    network.eval()

    print(network)

    # freezes the parameters for the whole network
    for param in network.parameters():
        param.requires_grad = False


    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( "greek/",
                                          transform = torchvision.transforms.Compose( 
                                            [torchvision.transforms.ToTensor(),
                                            GreekTransform(),
                                            torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = True )
    
    # DataLoader for handwritten Greek data set
    greek_handwritten = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( "greekhand/",
                                          transform = torchvision.transforms.Compose( 
                                            [torchvision.transforms.ToTensor(),
                                            GreekTransform(),
                                            torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = False )


    #preparing test data
    examples = enumerate(greek_train)
    print("examples: ",examples)
    batch_idx, (example_data, example_targets) = next(examples)

    
    network.fc2 = nn.Linear(50,3)
    print(network)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)

    #Training the datasets for n_epochs times
    train_losses = []
    train_counter = []

    for epoch in range(1,n_epochs + 1):
        
        correct = train(epoch,network,optimizer,greek_train,log_interval,train_losses,train_counter)
        print(f"Epoch {epoch}: {correct}/ 27 for {(correct / 27) * 100}% accuracy.\n")
        if (correct/27 == 1):
            print(f"Reached 100 %  accuracy after {epoch} epochs.\n")
            break
    

    print("finished training")
    torch.save(network.state_dict(), 'greekmodel.pth')
    plot(train_counter,train_losses)





if __name__ == "__main__":
    main(sys.argv)