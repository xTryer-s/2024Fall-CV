import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt # 可视化训练结果
import os # 文件操作
from models import VGG, ResNet, ResNext
from torch.utils.tensorboard import SummaryWriter 

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
        transforms.Normalize((0.476, 0.521, 0.489), (0.478, 0.456, 0.498))])

    # test_transform = transforms.Compose(
    # [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)
    device = torch.device("cuda:0"if torch.cuda.is_available() else"cpu") # GPU

    model=torch.load("vgg30.pt")
    test(model)
    model=torch.load("resnet30.pt")
    test(model)
    model=torch.load("resnext30.pt")
    test(model)