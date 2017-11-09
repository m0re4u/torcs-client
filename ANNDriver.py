from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import math
from nn.train import TwoLayerNet
from torch.autograd import Variable


DEGREE_PER_RADIANS = 180 / math.pi


class ANNDriver(Driver):
    def drive(self, carstate: State) -> Command:
        model = TwoLayerNet(21, 100, 3)
        model.load_state_dict(torch.load("/home/m0re/projects/uni/ci_vu/torcs-client/nn/NNdriver.pt"))
        sensors = [carstate.distance_from_center, carstate.angle / DEGREE_PER_RADIANS, *carstate.distances_from_edge]
        y = model(Variable(torch.Tensor(sensors)))
        command = Command()
        command.accelerator = y.data[0]
        command.brake = y.data[1]
        command.steering = -y.data[2]
        self.switch_gear(carstate, command)
        print("---------------------------")
        print("Angle: {}".format(carstate.angle / DEGREE_PER_RADIANS))
        print(carstate.distances_from_edge)
        return command

    def switch_gear(self, carstate, command):
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
