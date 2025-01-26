import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt # 可视化训练结果
import os # 文件操作

from torch.utils.tensorboard import SummaryWriter 

class FCNN(nn.Module):
    # def a full-connected neural network classifier
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()

        self.conv1=nn.Conv2d(3,30,kernel_size=5) # 30*28*28
        self.pool = nn.MaxPool2d(2,2) # 8*14*14
        self.conv2=nn.Conv2d(30,100,kernel_size=3) # 100*12*12
        # 100*6*6
        self.conv3=nn.Conv2d(100,256,kernel_size=3) # 256*4*4

        #14*5*5
        self.fc1=nn.Linear(256*4*4,80)
        self.fc2=nn.Linear(80,64)
        self.fc3=nn.Linear(64,out_channels)

        # initialize parameters
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)


    def forward(self, x: torch.Tensor): 
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=F.relu(self.conv3(x))
        x=torch.flatten(x,1)

        x=F.relu(self.fc1(x))
        x=F.tanh(self.fc2(x))
        x=self.fc3(x)

        return x



def test(model,epoch=-1):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
    '''
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    # create testing dataset
    # create dataloader
    # test
        # forward
        # calculate accuracy
    

    accurate_num,test_num=0.0,0.0
    model.eval()
    for i,data in enumerate(testloader,0):
        inputs,labels=data
        inputs,labels=inputs.to(device),labels.to(device)
        # forward
        outputs=model(inputs)

        accurate_num+=(outputs.argmax(1)==labels).sum()
        test_num+=labels.shape[0]
    
    test_acc = accurate_num/test_num
    if epoch==-1:
        print(f'test-acc: {test_acc}')
    else:
        print(f'[Epoch:{epoch+1}] test-acc: {test_acc}')

    return test_acc



if __name__ == '__main__':

    batch_size=64

    test_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)
    device = torch.device("cuda:0"if torch.cuda.is_available() else"cpu") # GPU

    model=torch.load("final_pt.pt")
    test(model)