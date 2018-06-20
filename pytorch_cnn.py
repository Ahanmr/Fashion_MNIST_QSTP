#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 23:02:00 2018

@author: ahanmr
"""
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Create an empty numpy array to fill
        img_as_np = np.ones((28, 28), dtype='uint8')
        # Fill the numpy array with data from pandas df
        for i in range(1, self.data.shape[1]):
            row_pos = (i-1) // self.height
            col_pos = (i-1) % self.width
            img_as_np[row_pos][col_pos] = self.data.iloc[index][i]
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)

if __name__ == "__main__":

    transformations = transforms.Compose([transforms.ToTensor()])

    train_dataset = CustomDatasetFromCSV('fashion-mnist_train.csv',
                             28, 28,
                             transformations)
    test_dataset = CustomDatasetFromCSV('fashion-mnist_test.csv',
                             28, 28,
                             transformations)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=100,
                                                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=100,
                                                    shuffle=False)
    batch_size=100
    n_iters=18
    num_epochs=n_iters/(len(train_loader)/batch_size)
    num_epochs=int(num_epochs)

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc = nn.Linear(7*7*32, 10)
        
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
    model=CNN()
    criterion=nn.CrossEntropyLoss()
    learning_rate=0.015
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
    
    iter=0
    for epoch in range(num_epochs):
        for i,(images,labels) in enumerate (train_loader):
            images=Variable(images)
            labels=Variable(labels)
    
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            iter+=1
            if iter%500==0:
                correct=0
                total=0
                for images,labels in test_loader:
                    images=Variable(images)
    
                    outputs=model(images)
                    
                    _,predicted=torch.max(outputs.data,1)
                    total+=labels.size(0)
                    correct+=(predicted==labels).sum()
                accuracy= (100.0* correct)/(total)
                print("Iteration:"+str(iter)+"  Loss:"+str(loss)+"  Accuracy:"+str(accuracy))
    