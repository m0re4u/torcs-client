import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import pickle


# Echo State Network Reservoir module
class Reservoir(nn.Module):
    """
    Echo State Network Reservoir module
    """

    def __init__(self, input_features, output_features, size, bias=True, initial_state=None):
        """
        Constructor
        :param input_features: Number of input features.
        :param reservoir_features:  Reservoir's size.
        :param output_features: Number of outputs
        :param size: Reservoir size
        :param bias: Use bias?
        """
        super(Reservoir, self).__init__()

        # Params
        self.input_features = input_features
        self.output_features = output_features
        self.size = size
        self.bias = bias
        self.initial_state = initial_state

        # The learnable output weights
        self.weight = nn.Parameter(torch.Tensor(output_features))

        # Initialize reservoir vector
        if self.initial_state is not None:
            self.x = Variable(self.initial_state, requires_grad=False)
        else:
            self.x = Variable(torch.zeros(self.size), requires_grad=False)
        # end if

        # Initialize inout weights
        self.win = Variable((torch.rand(self.size, self.input_features) - 0.5) * 2.0, requires_grad=False)

        # Initialize reservoir weights randomly
        self.w = Variable((torch.rand(self.size, self.size) - 0.5) * 2.0, requires_grad=False)

        # Linear output
        self.ll = nn.Linear(self.size, self.output_features)

    # end __init__


    # Forward
    def forward(self, u, X):
        """
        Forward
        :param u: Input signal
        :return: I don't know.
        """
        # Batch size
        batch_size = u.size()[0]

        # States
        states = Variable(torch.zeros(batch_size, self.size), requires_grad=False)

        # Starting state
        x = X

        # For each state
        for index in range(batch_size):
            x = F.tanh(self.win.mv(u[index, :]) + self.w.mv(x))
            states[index, :] = x
        # end for

        # Linear output
        p = self.ll(states)

        return p, x
    # end forward

# end Reservoir


def main(train_file, cuda_enabled, params):
    data = ds.DriverDataset(train_file)
    train_loader = DataLoader(data, batch_size=params["batch"], shuffle=False, num_workers=1)

    H = params["hidden"]  # number of hidden neurons
    alpha = params["lr"]  # learning rate
    epochs = params["epochs"]

    D_in = 22  # number of inputs
    D_out = 3  # number of outputs

    print("Data has size: {}".format(len(data)))

    model = Reservoir(D_in, D_out, H, bias=False)
    if cuda_enabled:
        model.cuda()

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    initial_state = Variable(torch.zeros(H), requires_grad=False)

    for epoch in range(epochs):
        for batch_i, (batch_target, batch_data) in enumerate(train_loader):
            if cuda_enabled:
                batch_data, batch_target = batch_data.cuda(), batch_target.cuda()
            x_batch = Variable(batch_data)
            y_batch = Variable(batch_target)

            # Forward pass
            y_pred, initial_state = model(x_batch, initial_state)

            # Compute and print loss
            loss = criterion(y_pred, y_batch)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: {:6d} - Loss: {}".format(epoch, loss.data[0] / len(x_batch)))

    torch.save(model.state_dict(), "../models/NNdriverReservoir.pt")
    torch.save(initial_state, "../models/initial_state_reservoir.pt")


if __name__ == '__main__':
    import dataset as ds
    pd.options.mode.chained_assignment = None  # default='warn'

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "-f", "--train_file", help="",
        default="../data/test.csv"
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="Set the learning rate",
        default="1e-3", type=float
    )
    parser.add_argument(
        "-H", "--hidden", help="Set the number of hidden neurons",
        default="500", type=int
    )
    parser.add_argument(
        "-e", "--epochs", help="Set the number of epochs to run",
        default="100", type=int
    )
    parser.add_argument(
        "-b", "--batch", help="Set the batch size",
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
        "mom": args.momentum,
        "batch": args.batch
    }
    main(args.train_file, cuda_enabled, param_dict)
else:
    from . import dataset as ds
