""" James Marcel
    CS5330 Project 5
    April 3 2023
    Loading network from .pth file
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


#custom handwriting dataset
class handSet(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.numbers = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.numbers)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.numbers.iloc[idx,0])

        value = torch.tensor(int(self.numbers.iloc[idx,1]))
        image = Image.open(img_name)

        imgArray = np.array(image)
        #inverting image
        imgArray = 255 - imgArray
        image = Image.fromarray(imgArray)


        if self.transform:
            image = self.transform(image)
        dimensions = [1,28,28]
        result = torch.ones(dimensions)
        print("image shape ", image.shape)
        #result = torch.tensor(image[0].clone().detach())
        result[0] = torch.tensor(image[0].clone().detach())

        print("result shape: ",result.shape)
        #torch.unsqueeze(result,0)
        #print("after unsqueeze ", result.shape)

        return(result, value)





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
def imgPlot(model, examples):
    output = model(examples)
    #print(output)
    
    fig = plt.figure()
    for i in range(10):
        print(i,"\n")
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(examples[i][0], cmap = 'gray', interpolation = 'none')
        plt.title(f"Prediction: {output.data.max(1,keepdim=True)[1][i].item()}")
        plt.xticks([])
        plt.yticks([])
    plt.show()

def main(argv):
    torch.backends.cudnn.enabled = False
    network = NeuralNet()
    network.load_state_dict(torch.load("model.pth"))
    network.eval()

    #Loading Test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/',train = False, download = True,
                                transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                ])),
                                batch_size = 64, shuffle = True)

    print("Model Load Successful.\n")

    #preparing test data
    examples = enumerate(test_loader)
    print("examples: ",examples)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    #plotting 3x3 grid of examples
    #testPlot(network, example_data)

    #plotting my handwriting examples
    #handwriting = ["imgs/0.png","imgs/1.png","imgs/2.png","imgs/3.png","imgs/4.png","imgs/5.png","imgs/6.png","imgs/7.png","imgs/8.png","imgs/9.png",]
    

    transformList = [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = 0.1307,std = 0.3081)]
    makeTensor = torchvision.transforms.Compose(transformList)

    examples = handSet("handwriting.csv", "imgs/", makeTensor)
    print(len(examples))

    loader = torch.utils.data.DataLoader(examples, batch_size = 10,shuffle = False)
    
    examples = enumerate(loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    imgPlot(network,example_data)




if __name__ == "__main__":
    main(sys.argv)