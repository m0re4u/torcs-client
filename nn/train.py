import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import pandas as pd
import shutil


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h = self.linear1(x)
        h = F.tanh(h)
        y_pred = self.linear2(h)
        y_pred = F.softmax(y_pred)
        return y_pred


def read_file(filename):
    df = pd.read_csv(filename, skiprows=1)
    targets = df[df.columns[0:3]]
    targets.drop(targets.tail(1).index, inplace=True)
    target_matrix = targets.as_matrix()
    data = df[df.columns[4:]]
    data.drop(data.tail(1).index, inplace=True)
    data_matrix = data.as_matrix()
    return target_matrix, data_matrix


def main(train_file, test_file, cuda_enabled):
    targets, data = read_file(train_file)
    # test_target, test_data = read_file(test_file)

    H = 100  # number of hidden neurons
    alpha = 1e-07  # learning rate
    epochs = 3000
    D_in, D_out = data.shape[1], targets.shape[1]
    EXAMPLES = data.shape[0]
    if cuda_enabled:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    print("Data have size: {}".format(data.shape))
    print("Targets have size: {}".format(targets.shape))

    model = TwoLayerNet(D_in, H, D_out)
    if cuda_enabled:
        model.cuda()

    print(model)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=0.9)

    for epoch in range(epochs):
        x_batch = Variable(torch.Tensor(data).type(dtype), requires_grad=True)
        y_batch = Variable(torch.Tensor(targets).type(dtype), requires_grad=False)

        y_pred = model(x_batch)

        # Compute and print loss
        loss = criterion(y_pred, y_batch)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch: {:5d} - Loss: {}".format(epoch, loss.data[0]))

    torch.save(model.state_dict(), "NNdriver.pt")


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "-f", "--train_file", help="",
        default="../../train_data/aalborg.csv"
    )
    parser.add_argument(
        "-t", "--test_file", help="",
        default="../../train_data/alpine-1.csv"
    )
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    cuda_enabled = args.cuda and torch.cuda.is_available()
    if cuda_enabled:
        print("CUDA is enabled")
    else:
        print("CUDA is not enabled")
    main(args.train_file, args.test_file, cuda_enabled)
