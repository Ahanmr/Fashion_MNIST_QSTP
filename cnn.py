# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:01:01 2018

@author: ahanmr
"""
import torch
import pandas as pd
import numpy as np

mnist_test=pd.read_csv("fashion-mnist_test.csv")

mnist_train=pd.read_csv("fashion-mnist_train.csv")

test_data=np.array(mnist_test, dtype=float)
train_data=np.array(mnist_train, dtype=float)

print(train_data.shape)
print(test_data.shape)

x_train=train_data[:,1:]/255
y_train=train_data[:,0]

x_test=test_data[:,1:]/255
y_test=test_data[:,0]

import matplotlib.pyplot as plt
image = x_train[16, :].reshape((28,28))
plt.imshow(image)
plt.show()

from sklearn.cross_validation import train_test_split
x_train, x_validate, y_train, y_validate = train_test_split(x_train,y_train, test_size=0.2, random_state=0)

'''
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

'''

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
#train_dataset=fashion(root='./data',train=True,transform=transforms.ToTensor(),download=True)
#test_dataset=fashion(root='./data',train=False,transform=transforms.ToTensor(),download=True)
batch_size=100
n_iters=18000
num_epochs=n_iters/(len(mnist_train)/batch_size)
num_epochs=int(num_epochs)

train_loader=torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,shuffle=True)

class CNNModule(nn.Module):
    def __init__(self):
        super (CNNModule,self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.relu1=nn.ELU()
        nn.init.xavier_uniform(self.cnn1.weight)


        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.relu2=nn.ELU()
        nn.init.xavier_uniform(self.cnn2.weight)


        self.maxpool2=nn.MaxPool2d(kernel_size=2)

        self.fcl=nn.Linear(32*7*7,10)
           
    def forward(self,x):
        out=self.cnn1(x)
        out=self.relu1(out)
        #print ("CNN1")
        #print (out.size())
        
        out=self.maxpool1(out)
        #print ("Maxpool1")
        #print (out.size())
        
        out=self.cnn2(out)
        out=self.relu2(out)
        #print ("CNN2")
        #print (out.size())
        out=self.maxpool2(out)
        #print ("Maxpool2")
        #print (out.size(0))
        
        out=out.view(out.size(0),-1)

        out=self.fcl(out)

        return out

