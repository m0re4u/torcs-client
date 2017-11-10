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
    """
    Read a data file from TORCS and use it as labeled data
    """
    df = pd.read_csv(filename)
    # Drop speed column
    df1 = df.drop(['SPEED'], axis=1)
    # Drop last row
    df1.drop(df1.tail(1).index, inplace=True)

    # First three columns are targets, following 21 are input
    return df1.as_matrix()


def main(train_file, cuda_enabled, params):
    data = read_file(train_file)

    H = params["hidden"]  # number of hidden neurons
    alpha = params["lr"]  # learning rate
    epochs = params["epochs"]

    D_in = 21  # number of inputs
    D_out = 3  # number of outputs

    if cuda_enabled:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    print("Data has size: {}".format(data.shape))

    model = TwoLayerNet(D_in, H, D_out)
    if cuda_enabled:
        model.cuda()

    print(model)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=params["mom"])

    for epoch in range(epochs):
        np.random.shuffle(data)
        x_batch = Variable(torch.Tensor(data[:, 3:]).type(dtype), requires_grad=True)
        y_batch = Variable(torch.Tensor(data[:, :3]).type(dtype), requires_grad=False)
        y_pred = model(x_batch)

        # Compute and print loss
        loss = criterion(y_pred, y_batch)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch: {:6d} - Loss: {}".format(epoch, loss.data[0]))

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
        "-lr", "--learning_rate", help="Set the learning rate",
        default="0.000001", type=float
    )
    parser.add_argument(
        "-H", "--hidden", help="Set the number of hidden neurons",
        default="500", type=int
    )
    parser.add_argument(
        "-e", "--epochs", help="Set the number of epochs to run",
        default="10000", type=int
    )
    parser.add_argument(
        "-m", "--momentum", help="Set the momentum of the SGD",
        default="0.9", type=float
    )
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()
    cuda_enabled = args.cuda and torch.cuda.is_available()

    if cuda_enabled:
        print("CUDA is enabled")
    else:
        print("CUDA is not enabled")

    param_dict = {
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "mom": args.momentum
    }
    main(args.train_file, cuda_enabled, param_dict)
