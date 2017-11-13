import torch
import numpy as np
from torch import autograd, nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import csv

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, target_size):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, target_size)

    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.tanh(x)
        return x


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


def getData(path):
    input_vectors = []
    target_vectors = []
    with open(path, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i, row in enumerate(data):
            if i != 0:
                float_row = [float(item) for item in row]
                input_vectors.append(float_row[3:])
                target_vectors.append(float_row[0:3])
        del input_vectors[-1]
        del target_vectors[-1]

    return input_vectors, target_vectors


def chunks(l, n):
    for i in range(0, len(l), n):
        return l[i:i + n]


def run_ff_net():
    input_vectors, target_vectors = getData('test.csv')

    # Remove last column of the data (TODO??)
    # for i, _ in enumerate(input_vectors):
    #     input_vectors[i] = input_vectors[i][:-1]

    input_size = len(input_vectors[0])
    hidden_size = 15
    target_size = len(target_vectors[0])
    learning_rate = 1e-04

    model = Model(input_size, hidden_size, target_size)
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    try:
        for epoch in range(15000):
            input = autograd.Variable(torch.FloatTensor(input_vectors))
            target = autograd.Variable(torch.Tensor(target_vectors))
            out = model(input)
            loss = criterion(out, target)
            model.zero_grad()
            loss.backward()
            opt.step()

            print('epoch: %i, loss: %f' % (epoch, loss.data[0]))
    except KeyboardInterrupt:
        print('out', out)
        print('target', target.view(1, -1))
        print('loss', loss.data[0])
        torch.save(model.state_dict(), "NNdriver.pt")

    print('out', out)
    print('target', target.view(1, -1))
    print('loss', loss.data[0])
    torch.save(model.state_dict(), "NNdriver.pt")


def run_reservoir():
    input_vectors, target_vectors = getData('test.csv')

    input_size = len(input_vectors[0])
    hidden_size = input_size
    reservoir_size = int(len(input_vectors)/10)
    target_size = len(target_vectors[0])
    learning_rate = 1e-04

    model = Reservoir(input_size, target_size, reservoir_size)
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(3000):
        input = autograd.Variable(torch.FloatTensor(input_vectors))
        target = autograd.Variable(torch.Tensor(target_vectors))
        initial_state = Variable(torch.zeros(reservoir_size), requires_grad=False)
        out = model(input, initial_state)
        criterion = nn.MSELoss()
        loss = criterion(out[0], target)
        model.zero_grad()
        loss.backward()
        opt.step()

        print('loss', loss.data[0])

    print('out', out)
    print('target', target.view(1, -1))
    print('loss', loss.data[0])
    torch.save(model.state_dict(), "NNdriver.pt")

if __name__ == '__main__':
    run_ff_net()
    # run_reservoir()
