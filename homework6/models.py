import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define multiple convolution and downsampling layers
        self.features=nn.Sequential(
            #multiple convolution layers

            self.vgg_block(3,64),
            self.vgg_block(64,64),
            nn.MaxPool2d(2,2),#=>64*16*16

            self.vgg_block(64,128),
            self.vgg_block(128,128),
            nn.MaxPool2d(2,2),#=>128*8*8

            self.vgg_block(128,256),
            self.vgg_block(256,256),
            self.vgg_block(256,256),
            nn.MaxPool2d(2,2),#=>256*4*4

            self.vgg_block(256,512),
            self.vgg_block(512,512),
            self.vgg_block(512,512),
            nn.MaxPool2d(2,2),#=>512*2*2

            self.vgg_block(512,512),
            self.vgg_block(512,512),
            self.vgg_block(512,512),
            nn.MaxPool2d(2,2),#=>512*1*1

        )
        # 2. define full-connected layer to classify
        self.classfier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(512,512,bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512,128,bias=True),
            nn.Tanh(),
            nn.Dropout(), # dropout
            nn.Linear(128,10,bias=True)# class num = 10(cifar-10)
        )

        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
    def vgg_block(self,in_channel,out_channel):
        vgg_seq=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        return vgg_seq

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x = self.features(x)
        # flatten x
        x = torch.flatten(x,1)
        # classification
        out = self.classfier(x)
        return out


class ResBlock(nn.Module):
    ''' residual block'''
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        '''
        in_channel: number of channels in the input image.
        out_channel: number of channels produced by the convolution.
        stride: stride of the convolution.
        '''
        # 1. define double convolution
             # convolution
             # batch normalization
             # activate function
             # ......
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout()
        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        self.conv3=None
        if in_channel!=out_channel or stride!=1:# size or channel change
            self.conv3=nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride)
        
        # Note: we are going to implement 'Basic residual block' by above steps, you can also implement 'Bottleneck Residual block'

    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        out = self.relu(self.bn(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn(self.conv2(out))
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        if self.conv3 != None:
            # 3. Add the output of the convolution and the original data (or from 2.)
            out+=self.bn(self.conv3(x))
        else:
            out+=x
        # 4. relu
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    '''residual network'''
    def __init__(self):
        super().__init__()

        # input: 3*32*32
        # 1. define convolution layer to process raw RGB image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        # 64*16*16

        # 2. define multiple residual blocks
        self.res_layer1 = self.make_resnet_layer(64,64,3,stride=1) # 8*8
        self.res_layer2 = self.make_resnet_layer(64,128,3,stride=2) # 4*4 
        self.res_layer3 = self.make_resnet_layer(128,256,3,stride=2) # 2*2
        self.res_layer4 = self.make_resnet_layer(256,512,3,stride=2) # 1*1

        # 3. define full-connected layer to classify
        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))

        # classfier
        self.fc = nn.Linear(512,10)

        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

    def make_resnet_layer(self,in_channels,out_channels,blocks_num,stride):
        ret_layers=[]
        ret_layers.append(ResBlock(in_channels,out_channels,stride))
        for i in range(blocks_num-1):
            ret_layers.append(ResBlock(out_channels,out_channels,stride=1))
        return nn.Sequential(*ret_layers)
    
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        x=self.conv1(x)
        x=self.res_layer1(x)
        x=self.res_layer2(x)
        x=self.res_layer3(x)
        x=self.res_layer4(x)
        x= self.global_avgpool(x)

        # flatten x
        x=torch.flatten(x,1)
        # classification
        out=self.fc(x)
        return out
    

class ResNextBlock(nn.Module):
    '''ResNext block'''
    def __init__(self, in_channel, out_channel, bottle_neck, group, stride):
        super().__init__()
        # in_channel: number of channels in the input image
        # out_channel: number of channels produced by the convolution
        # bottle_neck: int, bottleneck= out_channel / hidden_channel 
        # group: number of blocked connections from input channels to output channels
        # stride: stride of the convolution.

        hidden_channel = out_channel//bottle_neck

        # 1. define convolution
        # 1x1 convolution
        # batch normalization
        # activate function

        self.conv1=nn.Conv2d(in_channel,hidden_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 convolution
        # batch normalization
        self.conv2=nn.Conv2d(hidden_channel,hidden_channel,kernel_size=3,stride=stride,padding=1,groups=group,bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channel)

        # 1x1 convolution
        # batch normalization
        self.conv3 = nn.Conv2d(hidden_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        # 2. if in_channel != out_channel or stride != 1, deifine 1x1 convolution layer to change the channel or size.
        self.conv4 = None
        if in_channel!=out_channel or stride!=1:# size or channel change
            self.conv4 = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride,padding=0,bias=False)
        self.bn4 = nn.BatchNorm2d(out_channel)

        self.dropout=nn.Dropout()
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # 1. convolve the input
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        # 2. if in_channel != out_channel or stride != 1, change the channel or size of 'x' using 1x1 convolution.
        if self.conv4!=None:
            # 3. Add the output of the convolution and the original data (or from 2.)
            out+=self.bn4(self.conv4(x))
        else:
            out+=x
        # 4. relu
        out = self.relu(out)
        return out


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. define convolution layer to process raw RGB image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        # 64*16*16

        # 2. define multiple residual blocks
        # make_layer(in_channel, out_channel, bottle_neck, group, stride,block_nums)
        self.resnext_layer1=self.make_resnext_layer(64,64,1,32,1,3) # 64*8*8
        self.resnext_layer2=self.make_resnext_layer(64,128,2,32,2,4) # 128*4*4
        self.resnext_layer3=self.make_resnext_layer(128,256,2,32,2,3) # 256*2*2
        self.resnext_layer4=self.make_resnext_layer(256,512,2,32,2,3) # 512*1*1

        self.global_avgpool = nn.AdaptiveAvgPool2d((1,1))
        # 3. define full-connected layer to classify
        self.fc = nn.Linear(512,10)

        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)

    def make_resnext_layer(self, in_channel,out_channel, bottle_neck, group, stride,blocks_num):
        ret_layers=[]
        ret_layers.append(ResNextBlock(in_channel,out_channel,bottle_neck,group,stride))
        for i in range(blocks_num-1):
            ret_layers.append(ResNextBlock(out_channel,out_channel,bottle_neck,group,stride=1))
        return nn.Sequential(*ret_layers)
    
    def forward(self, x: torch.Tensor):
        # x: input image, shape: [B * C * H* W]
        # extract features
        out = self.conv1(x)

        # resnext network:
        out = self.resnext_layer1(out)
        out = self.resnext_layer2(out)
        out = self.resnext_layer3(out)
        out = self.resnext_layer4(out)

        out = self.global_avgpool(out)

        # flatten x
        out = torch.flatten(out,1)
        # classification
        out = self.fc(out)
        return out

