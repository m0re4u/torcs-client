import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import pandas as pd


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
        y_pred = F.tanh(y_pred)
        return y_pred


def main(train_file, cuda_enabled, params):
    data = ds.DriverDataset(train_file)
    train_loader = DataLoader(data, batch_size=params["batch"], shuffle=True, num_workers=1)

    H = params["hidden"]  # number of hidden neurons
    alpha = params["lr"]  # learning rate
    epochs = params["epochs"]

    D_in = 22  # number of inputs
    D_out = 3  # number of outputs

    print("Data has size: {}".format(len(data)))

    model = TwoLayerNet(D_in, H, D_out)
    if cuda_enabled:
        model.cuda()

    print(model)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=params["mom"])

    for epoch in range(epochs):
        for batch_i, (batch_target, batch_data) in enumerate(train_loader):
            if cuda_enabled:
                batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
            x_batch = Variable(batch_data)
            y_batch = Variable(batch_target)

            # Forward pass
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
    import dataset as ds
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
        "-b", "--batch", help="Set the batch size",
        default="10", type=int
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
        "mom": args.momentum,
        "batch": args.batch
    }
    main(args.train_file, cuda_enabled, param_dict)
else:
    from . import dataset as ds
