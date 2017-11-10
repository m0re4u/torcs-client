from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import math
from nn.train import TwoLayerNet
from torch.autograd import Variable


DEGREE_PER_RADIANS = 180 / math.pi


class ANNDriver(Driver):
    def __init__(self):
        self.model = TwoLayerNet(21, 500, 3)
        self.model.load_state_dict(torch.load("/home/m0re/projects/uni/ci_vu/torcs-client/nn/NNdriver.pt", map_location=lambda storage, loc: storage))

    def drive(self, carstate: State) -> Command:
        sensors = [carstate.distance_from_center, carstate.angle / DEGREE_PER_RADIANS, *(carstate.distances_from_edge)]
        # Forward pass our model
        y = self.model(Variable(torch.Tensor(sensors)))

        # Create command from model output
        command = Command()
        # if y.data[0] > y.data[1]:
        #     command.accelerator = 1
        #     command.brake = 0
        # else:
        #     command.accelerator = 0
        #     command.brake = 1
        command.accelerator = y.data[0]
        command.brake = y.data[1]

        command.steering = y.data[2]

        # Naive switching of gear
        self.switch_gear(carstate, command)

        print("---------------------------")
        print(command)
        print(carstate.distances_from_edge)

        return command

    def switch_gear(self, carstate, command):
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
