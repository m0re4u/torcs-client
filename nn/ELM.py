import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import pickle

class ELM(nn.Module):
    def __init__(self, input_features, output_features, size):
        super(ELM, self).__init__()
        self.fc1 = nn.Linear(input_features, size)
        #self.bn = nn.BatchNorm1d(7000)
        self.fc2 = nn.Linear(size, output_features, bias=False) # ELM do not use bias in the output layer.

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        #x = self.bn(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x

    def forwardToHidden(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        #x = self.bn(x)
        x = F.leaky_relu(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def trainELM(train_file, cuda_enabled, params):
    data = ds.DriverDataset(train_file)
    train_loader = DataLoader(data, batch_size=params["batch"], shuffle=False, num_workers=1)

    H = params["hidden"]  # number of hidden neurons
    alpha = params["lr"]  # learning rate
    epochs = params["epochs"]

    D_in = 22  # number of inputs
    D_out = 3  # number of outputs

    print("Data has size: {}".format(len(data)))

    model = ELM(D_in, D_out, H)
    if cuda_enabled:
        model.cuda()

    print(model)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    for epoch in range(epochs):
        for batch_i, (batch_target, batch_data) in enumerate(train_loader):
            if cuda_enabled:
                batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
            x_batch = Variable(batch_data)
            y_batch = Variable(batch_target)

            # Forward pass
            hiddenOut = model.forwardToHidden(x_batch)

            # Compute and print loss
            loss = criterion(hiddenOut, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred = model(x_batch)

            loss = criterion(y_pred, y_batch)
            print(loss)

            # Zero gradients, perform a backward pass, and update the weights.

        print("Epoch: {:6d} - Loss: {}".format(epoch, loss.data[0]))

    torch.save(model.state_dict(), "NNdriverReservoir.pt")