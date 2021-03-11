#! /usr/bin/env python

import torch
import torch.nn as nn
from torchvision import datasets ,models,transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision

# training parameters
data_path = '/home/austin/DataSet/ncsist_dataset/missile_angle/missile_angle_datasets'
batch_size = 40
n_class = 6
Epoch = 50

#load image from folder and set foldername as label

train_data = datasets.ImageFolder(
    data_path + '/train_data',
    transform = transforms.Compose([transforms.ToTensor()])                         
)

test_data = datasets.ImageFolder(
    data_path + '/test_data',
    transform = transforms.Compose([transforms.ToTensor()])                         
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle= True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True)

#CNN model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Sequential(              
            nn.Conv2d(
                in_channels=3,              
                out_channels=32,            
                kernel_size=4,              
                stride=1,                   
                padding=0,                  
            ),                                                 
            nn.MaxPool2d(kernel_size=2, stride=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=0,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=1,
                padding=1,
            ),                           
            nn.MaxPool2d(kernel_size=2, stride=2),                
        )
        self.fc1 = nn.Linear(34048, 200)
        self.fc2 = nn.Linear(200, n_class)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = CNN_Model().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
iTer = len(train_loader)
eval_iter = len(test_loader)

for epoch in range(Epoch): # loop over the dataset multiple times
    # Train mode
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the input
        train_inputs, train_labels = data

        # wrap time in Variable
        train_inputs, train_labels = Variable(train_inputs).cuda(), Variable(train_labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        train_outputs = net(train_inputs)
        loss = criterion(train_outputs, train_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Eval mode
    eval_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # get the input
            eval_inputs, eval_labels = data

            # wrap time in Variable
            eval_inputs, eval_labels = Variable(eval_inputs).cuda(), Variable(eval_labels).cuda()
            
            # prediction
            eval_outputs = net(eval_inputs)
            loss = criterion(eval_outputs, eval_labels)
            eval_loss += loss.item()

    if (epoch + 1)%10 == 0:
        torch.save(net.state_dict(),'/home/austin/CNN_epoch_%d_loss_%5f.pth' %(epoch + 1, train_loss / iTer))
        print('Model save')
    print('Epoch : %d || train_Loss : %.5f || eval_Loss : %.5f ' %(epoch + 1, train_loss / iTer, eval_loss / eval_iter))


print('Finished Training')