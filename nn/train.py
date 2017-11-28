import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

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
        h = self.linear2(h)
        h = F.tanh(h)
        return h


class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self.linear1(x)
        h = F.tanh(h)
        h = self.linear2(h)
        h = F.tanh(h)
        h = self.linear3(h)
        h = F.tanh(h)
        return h


class FiveLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(FiveLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self.linear1(x)
        h = F.tanh(h)
        h = self.linear2(h)
        h = F.tanh(h)
        h = self.linear3(h)
        h = F.tanh(h)
        h = self.linear4(h)
        h = F.tanh(h)
        h = self.linear5(h)
        h = F.tanh(h)
        return h


def main(train_file, cuda_enabled, params):
    data = ds.DriverDataset(train_file, normalize=params["norm"])
    train_loader = DataLoader(
        data, batch_size=params["batch"], shuffle=False, num_workers=1)

    epochs = params["epochs"]   # number of training epochs
    H = params["hidden"]        # number of hidden neurons
    alpha = params["lr"]        # learning rate

    D_in = 22  # number of inputs
    D_out = 3  # number of outputs

    print("Data has size: {}".format(len(data)))

    if params["depth"] == 5:
        model = FiveLayerNet(D_in, H, D_out)
    elif params["depth"] == 3:
        model = ThreeLayerNet(D_in, H, D_out)
    else:
        model = TwoLayerNet(D_in, H, D_out)

    if cuda_enabled:
        model.cuda()

    print(model)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    try:
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

                # Zero gradients, perform a backward pass, and update the
                # weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch: {:6d} - Loss: {}".format(epoch, loss.data[0]))
    except KeyboardInterrupt:
        # Save file on KeyboardInterrupt
        filename = "models/NNdriver{}-{}.pt".format(params['depth'], H)
        print("Saved model to {}".format(filename))
        torch.save(model.state_dict(), filename)

    # Save model after training
    filename = "models/NNdriver{}-{}.pt".format(params['depth'], H)
    print("Saved model to {}".format(filename))
    torch.save(model.state_dict(), filename)


if __name__ == '__main__':
    import dataset as ds
    pd.options.mode.chained_assignment = None  # default='warn'

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "-f", "--train_file", help="CSV file with training data",
        default="csv_data/aalborg.csv"
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="Set the learning rate",
        default="1e-3", type=float
    )
    parser.add_argument(
        "-H", "--hidden", help="Set the number of hidden neurons",
        default="15", type=int
    )
    parser.add_argument(
        "-e", "--epochs", help="Set the number of epochs to run",
        default="20000", type=int
    )
    parser.add_argument(
        "-b", "--batch", help="Set the batch size",
        default="10000", type=int
    )
    parser.add_argument(
        "-d", "--depth", help="Set depth of model",
        default="2", type=int
    )
    parser.add_argument(
        "-n", "--norm", help="Normalize sensor values between 0 and 1",
        default=False, action='store_true'
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
        "batch": args.batch,
        "depth": args.depth,
        "norm": args.norm
    }
    main(args.train_file, cuda_enabled, param_dict)
