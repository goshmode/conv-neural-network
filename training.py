""" James Marcel
    CS5330 Project 5
    April 3 2023
    Recognition using Deep Networks
"""

#import statements
import torch 
import torchvision 
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


#training function
def train(epoch,network,optimizer,train_loader,log_interval,train_losses, train_counter):
    network.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')

#test function
def test(network, test_loader,test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average = False).item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({(correct / len(test_loader.dataset)) * 100:.0f}%)\n")


#plot test images
def testPlot(example_data, example_targets):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap = 'gray', interpolation = 'none')
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.show()


#plotting model performance
def plot(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color = "green")
    print(len(test_counter), len(test_losses))
    plt.scatter(test_counter, test_losses, color = "blue")
    plt.legend(["Train Loss", "Test Loss"], loc = "upper right")
    plt.xlabel("Number of Training Examples Seen")
    plt.ylabel("Negative Log Likelihood Loss")
    plt.show()


def modelSave(model):
    torch.save(model.state_dict(), 'model.pth')
    print("Network Saved as model.pth")



#Main function 
def main(argv):
    #setting a few variables
    n_epochs = 0
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    torch.backends.cudnn.enabled = False 
    torch.manual_seed(42)

    #Loading Training data
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train = True, download = True,
                                transform= torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307),(0.3081))
                                ])),
                                batch_size = batch_size_train, shuffle = True)

    #Loading Test data
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/',train = False, download = True,
                                transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,),(0.3081,))
                                ])),
                                batch_size = batch_size_test, shuffle = True)


    
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    #initializing network and optimizer
    network = NeuralNet()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)

    #saving accuracy data to these lists
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    
    #Training and testing the datasets for n_epochs times
    test(network, test_loader,test_losses)
    for epoch in range(1,n_epochs + 1):
        train(epoch,network,optimizer,train_loader,log_interval,train_losses,train_counter)
        test(network, test_loader,test_losses)


    #plotting a few training images
    testPlot(example_data, example_targets)


    #plotting stuff
    plot(train_counter, train_losses, test_counter, test_losses)

    #saving model
    modelSave(network)

    



if __name__ == "__main__":
    main(sys.argv)