import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt # 可视化训练结果
import os # 文件操作

from torch.utils.tensorboard import SummaryWriter 


class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # out_channels: number of categories. For CIFAR-10, it's 10
        self.fc1=nn.Linear(in_channels,out_channels)

    def forward(self, x: torch.Tensor):

        # flatten the input x        
        y=x.view(x.size(0),-1)
        y=self.fc1(y)
        return y


class FCNN(nn.Module):
    # def a full-connected neural network classifier
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # hidden_channels
        # out_channels: number of categories. For CIFAR-10, it's 10

        # full connected layer
        # activation function
        # full connected layer
        # ......
        self.conv1=nn.Conv2d(3,30,kernel_size=5) # 30*28*28
        self.pool = nn.MaxPool2d(2,2) # 8*14*14
        self.conv2=nn.Conv2d(30,100,kernel_size=3) # 100*12*12
        # 100*6*6
        self.conv3=nn.Conv2d(100,256,kernel_size=3) # 256*4*4

        #14*5*5
        self.fc1=nn.Linear(256*4*4,80)
        self.fc2=nn.Linear(80,64)
        self.fc3=nn.Linear(64,out_channels)

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


def train(model, optimizer, scheduler, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
        optimizer: Adamw or SGD
        scheduler: step or cosine
        args: configuration
    '''


    if not os.path.exists(local_path):
        os.makedirs(local_path)
    child_folder = f'{local_path}/{args.model}-{args.optimizer}-{args.scheduler}'
    
    if not os.path.exists(child_folder):
        os.makedirs(child_folder)

    # record hyper parameters
    with open(f'{local_path}/{args.model}-{args.optimizer}-{args.scheduler}/log.txt','a',encoding='utf-8') as tmp_ouptut:
        tmp_ouptut.write(f'epoch num={epoch_max}\nbatch size={batch_size}\nAdamW_lr={adamw_lr}\n')
        tmp_ouptut.write(f'SGD_lr={sgd_lr}\nSGD_momentum={sgd_momentum}\nsteplr_gamma={steplr_gamma}\n')
        tmp_ouptut.write(f'steplr_stepsize={steplr_stepsize}\ncoslr_tmax={coslr_tmax}\n\n')
        
    criterion = nn.CrossEntropyLoss()
    # create dataset

    plot_train_acc=[]
    plot_test_acc=[]
    plot_train_loss=[]
    plot_lr=[]

    # for-loop 
        # train
    data_len = len(trainloader)
    for epoch in range(epoch_max):
        running_loss=0.0
        train_acc_sum,train_num=0.0,0.0

        # get the inputs; data is a list of [inputs, labels]
        for i,data in enumerate(trainloader,0):
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward
            outputs=model(inputs)

            # loss backward
            loss=criterion(outputs,labels)
            loss.backward()

            # optimize
            optimizer.step()

            running_loss+=loss.item()

            train_num +=labels.shape[0]
            train_acc_sum+=(outputs.argmax(1)==labels).sum()

        # adjust learning rate
        scheduler.step()
        # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
        save_path = f'{local_path}/{args.model}-{args.optimizer}-{args.scheduler}/ckpt{epoch+1}.pt'
        torch.save(model,save_path)


        epoch_train_acc = train_acc_sum/train_num
        epoch_loss = running_loss/data_len
        print(f'[Epoch:{epoch+1}] loss: {epoch_loss:.3f} train-acc:{epoch_train_acc:.2f}')
        # test
        epoch_test_acc = test(model,args,epoch)

        plot_train_acc.append(epoch_train_acc.cpu())
        plot_train_loss.append(epoch_loss)
        plot_test_acc.append(epoch_test_acc.cpu())
        plot_lr.append(scheduler.get_last_lr()[0])

        with open(f'{local_path}/{args.model}-{args.optimizer}-{args.scheduler}/log.txt','a',encoding='utf-8') as tmp_ouptut:
            tmp_ouptut.write(f'[Epoch:{epoch+1}] loss: {epoch_loss:.3f} train-acc:{epoch_train_acc:.2f}\n')
            tmp_ouptut.write(f'[Epoch:{epoch+1}] test-acc: {epoch_test_acc:.2f}\n')

        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Train-Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Test-Accuracy/train', epoch_test_acc, epoch)
        writer.add_scalar('Learning-Rate/train', scheduler.get_last_lr()[0], epoch)


    print("Finished Training")


    # plot 1- train acc & test acc
    plt.figure()
    plt.plot(plot_train_acc)
    plt.plot(plot_test_acc)
    plt.title(f'{args.model}-{args.optimizer}-{args.scheduler} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train-acc','test-acc'],loc='upper left')
    plt.savefig(f'{local_path}/{args.model}-{args.optimizer}-{args.scheduler}/{args.model}-{args.optimizer}-{args.scheduler}-Accuracy.png')

    # plot 2- train loss

    plt.figure()
    plt.plot(plot_train_loss)
    plt.title(f'{args.model}-{args.optimizer}-{args.scheduler}  Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train-loss'],loc='upper left')
    plt.savefig(f'{local_path}/{args.model}-{args.optimizer}-{args.scheduler}/{args.model}-{args.optimizer}-{args.scheduler}-TrainLoss.png')

    # plot 3-Learning Rate
    plt.figure()
    plt.plot(plot_lr)
    plt.title(f'{args.model}-{args.optimizer}-{args.scheduler}  Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend(['lr'],loc='upper left')
    plt.savefig(f'{local_path}/{args.model}-{args.optimizer}-{args.scheduler}/{args.model}-{args.optimizer}-{args.scheduler}-LearningRate.png')

def test(model, args,epoch=-1):
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
        print(f'test-acc: {test_acc:.2f}')
    else:
        print(f'[Epoch:{epoch+1}] test-acc: {test_acc:.2f}')

    return test_acc



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    args = parser.parse_args()


    writer = SummaryWriter(f"runs/{args.model}-{args.optimizer}-{args.scheduler}")
    local_path = 'model_saves'
    # hyperparameters

    # python main.py --run=train --model=fcnn --optimizer=adamw --scheduler=step
    epoch_max=30
    batch_size=64
    adamw_lr=2.8e-4

    sgd_lr=2.5e-2
    sgd_momentum=0.4

    steplr_gamma=0.6
    steplr_stepsize=epoch_max/6
    coslr_tmax=40


    device = torch.device("cuda:0"if torch.cuda.is_available() else"cpu") # GPU

    test_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    # create dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=1)

    # create model
    if args.model == 'linear':
        model = LinearClassifier(3*32*32,10)
    elif args.model == 'fcnn':
        model = FCNN(3*32*32,3,10)
    else: 
        raise AssertionError

    # create optimizer
    if args.optimizer == 'adamw':
        # create Adamw optimizer
        optimizer = torch.optim.AdamW(model.parameters(),lr=adamw_lr)
    elif args.optimizer == 'sgd':
        # create SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(),lr=sgd_lr,momentum=sgd_momentum)
    else:
        raise AssertionError
    
    # create scheduler
    if args.scheduler == 'step':
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,gamma=steplr_gamma,step_size=steplr_stepsize)
    elif args.scheduler == 'cosine':
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=coslr_tmax)
    else:
        raise AssertionError

    model=model.to(device=device)
    if args.run == 'train':
        train(model, optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, args)
    else: 
        raise AssertionError
    
    writer.flush()
    writer.close()
    
# You need to implement training and testing function that can choose model, optimizer, scheduler and so on by command, such as:
# python main.py --run=train --model=fcnn --optimizer=adamw --scheduler=step


