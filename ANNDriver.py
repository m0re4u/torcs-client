from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import math
from nn.train import Model
from torch.autograd import Variable


DEGREE_PER_RADIANS = 180 / math.pi


class ANNDriver(Driver):
    model = Model(22, 15, 3)
    model.load_state_dict(
    torch.load('/home/parallels/Desktop/Parallels_Shared_Folders/CI2017/torcs-server/torcs-client/nn/NNdriver.pt'))

    def drive(self, carstate: State) -> Command:
        sensors = [carstate.speed_x, carstate.distance_from_center, carstate.angle, *carstate.distances_from_edge]
        y = self.model(Variable(torch.Tensor(sensors)))
        command = Command()

        accelerator, brake, steering = self.smooth_commands(y.data[0], y.data[1], y.data[2])

        command.accelerator = accelerator
        command.brake = brake
        command.steering = steering
        self.switch_gear(carstate, command)
        print("---------------------------")
        print("SENSOR:", sensors)
        print("Accelerate", command.accelerator)
        print("Brake", command.brake)
        print("Steering", -command.steering)
        return command

    def smooth_commands(self, accelerator, brake, steering):
        if accelerator < 0.0:
            accelerator = 0.0

        if brake < 0.0:
                brake = 0.0
        if accelerator > 0.9:
            accelerator = 1.0
            brake = 0.0

        return accelerator, brake, steering

    def switch_gear(self, carstate, command):
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
