from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import math
import logging
from nn import train
from torch.autograd import Variable


class ANNDriver(Driver):
    def __init__(self, model_file, H, record_train_file=None):
        super().__init__(False)

        # Load model
        self.model = train.TwoLayerNet(22, H, 3)
        self.model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

        # Check if we want to record the actuator & sensor data
        self.record = False
        if record_train_file is not None:
            self.record = True
            self.file_handler = open(record_train_file, 'w')
            self.file_handler.write("ACCELERATION,BRAKE,STEERING,SPEED,\
            TRACK_POSITION,ANGLE_TO_TRACK_AXIS,TRACK_EDGE_0,TRACK_EDGE_1,\
            TRACK_EDGE_2,TRACK_EDGE_3,TRACK_EDGE_4,TRACK_EDGE_5,TRACK_EDGE_6,\
            TRACK_EDGE_7,TRACK_EDGE_8,TRACK_EDGE_9,TRACK_EDGE_10,\
            TRACK_EDGE_11,TRACK_EDGE_12,TRACK_EDGE_13,TRACK_EDGE_14,\
            TRACK_EDGE_15,TRACK_EDGE_16,TRACK_EDGE_17,TRACK_EDGE_18")

    def __del__(self):
        if self.record and self.file_handler:
            self.file_handler.close()
            self.file_handler = None

    def drive(self, carstate: State) -> Command:
        sensors = [carstate.speed_x, carstate.distance_from_center, carstate.angle, *(carstate.distances_from_edge)]
        # Forward pass our model
        y = self.model(Variable(torch.Tensor(sensors)))

        # Create command from model output
        command = Command()
        command.accelerator = y.data[0]
        command.brake = y.data[1]
        command.steering = y.data[2]

        # Naive switching of gear
        self.switch_gear(carstate, command)

        print("---------------------------")
        print(command)
        print(carstate.distances_from_edge)
        print("Speed: {}".format(carstate.speed_x))
        print("Angle: {}".format(carstate.angle))

        if self.record is True:
            sensor_string = ",".join([str(x) for x in sensors]) + "\n"
            self.file_handler.write(str(y.data[0]) + "," + str(y.data[1]) + "," + str(y.data[2]) + "," + sensor_string)
        return command

    def switch_gear(self, carstate, command):
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
