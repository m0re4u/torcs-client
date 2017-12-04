from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import math
import numpy as np
import logging
from pytocl.driver import Driver
from nn import train
import random
from torch.autograd import Variable


class ANNDriver(Driver):
    def __init__(self, model_file, H, depth, record_train_file=None, normalize=False):
        super().__init__(False)
        if normalize:
            self.norm = True
        else:
            self.norm = False

        # Select right model
        if depth == 3:
            self.model = train.ThreeLayerNet(22, H, 3)
        elif depth == 5:
            self.model = train.FiveLayerNet(22, H, 3)
        else:
            print("Using depth=2")
            self.model = train.TwoLayerNet(22, H, 3)
        # Load model
        self.model.load_state_dict(torch.load(
            model_file, map_location=lambda storage, loc: storage))

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

        self.is_leader = True
        self.helper = StupidDriver(self.record)
        self.standard_driver = Driver()
        self.race_started = False
        self.recovers = 0
        self.recover_mode = False
        self.speeds = []
        self.number = 1

    def __del__(self):
        if self.record:
            self.file_handler.close()
            self.file_handler = None

    def drive(self, carstate: State) -> Command:
        if not self.race_started and carstate.distance_from_start < 3:
            self.race_started = True
            try:
                position_team = int(open("mjv_partner1.txt", 'r').read())
                open("mjv_partner2.txt", 'w').write(str(carstate.race_position))
                self.number = 2
                self.partner = 1
            except:
                open("mjv_partner1.txt", "w").write(str(carstate.race_position))
                self.number = 1
                self.partner = 2
        elif self.race_started:
            position_other = int(open("mjv_partner{}.txt".format(self.partner), 'r').read())
            open("mjv_partner{}.txt".format(self.number), "w").write(str(carstate.race_position))
            if carstate.race_position > position_other:
                self.leader = False
                print("Sukkel")
            else:
                self.leader = True
                print("Leader")

        # print("Distance: {}".format(carstate.distance_from_start))
        # Select the sensors we need for our model

        self.speeds.append(carstate.speed_x)

        command = Command()
        distances = list(carstate.distances_from_edge)
        sensors = [carstate.speed_x, carstate.distance_from_center,
                   carstate.angle, *(carstate.distances_from_edge)]
        # CRASHED
        off_road = all(distance == -1.0 for distance in distances)
        standing_still = self.recovers == 0 and all(abs(s) < 1.0 for s in self.speeds[-5:])
        if self.race_started and (off_road or self.recovers > 0):
            command = self.recover(carstate, command)
        # NOT CRASHED
        else:
            if carstate.gear == -1:
                carstate.gear = 2
            if self.norm:
                # Sensor normalization -> ?
                sensors = self.normalize_sensors(sensors)

            # Forward pass our model
            y = self.model(Variable(torch.Tensor(sensors)))
            # Create command from model output

            command.accelerator = np.clip(y.data[0], 0, 1)
            command.brake = np.clip(y.data[1], 0, 1)
            command.steering = np.clip(y.data[2], -1, 1)

            if self.race_started and not self.is_leader and carstate.distance_from_start > 50:
                command = self.helper.drive(command, distances, carstate)

            # Naive switching of gear
            self.switch_gear(carstate, command)

        # print("Accelerate: {}".format(command.accelerator))
        # print("Brake:      {}".format(command.brake))
        # print("Steer:      {}".format(command.steering))

        if self.record is True:
            sensor_string = ",".join([str(x) for x in sensors]) + "\n"
            self.file_handler.write(str(y.data[0]) + "," + str(y.data[1]) + "," + str(y.data[2]) + "," + sensor_string)
        return command

    def recover(self, carstate, command):
        self.recover_mode = True
        dist = abs(carstate.distance_from_center)
        front_right = carstate.angle < 0 and carstate.distance_from_center < 0
        back_right = carstate.angle > 0 and carstate.distance_from_center < 0

        front_left = carstate.angle > 0 and carstate.distance_from_center > 0
        back_left = carstate.angle < 0 and carstate.distance_from_center > 0

        if front_right or front_left:
            command.gear = 1
        else:
            command.gear = -1

        if front_right or back_left or (dist < 0.1 and self.recovers > 25):
            command.steering = 1
        else:
            command.steering = -1

        command.accelerator = 0.4

        self.recovers += 1
        if (dist < 0.1 and self.recovers > 25) or self.recovers > 50:
            self.recovers = 0
        else:
            self.recover_mode = True
        return command

    def switch_gear(self, carstate, command):
        """
        Naive switching of gears
        """
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

    def normalize_sensors(self, sensors):
        """
        Normalize all sensor values to be between 0 and 1
        """
        new_sensors = []

        for i, sensor in enumerate(sensors):
            # Speed -100 --> 300 km/hour
            if i == 0:
                new_sensors.append((sensor + 100) / 400)

            # Distance from centre -1 --> 1
            if i == 1:
                new_sensors.append(sensor)

            # Angle to track -1 --> 1
            if i == 2:
                new_sensors.append(sensor)

            # Track edges 0 --> 200 meters
            if i > 2:
                new_sensors.append(sensor / 200)

        return new_sensors


class StupidDriver(Driver):
    def __init__(self, record):
        super().__init__(False)
        self.last_command = None
        self.record = record

    def __del__(self):
        if self.record:
            self.file_handler.close()
            self.file_handler = None

    def drive(self, command, distances, carstate: State) -> Command:

        # Create command from model output
        command = self.next_move(carstate, command, distances)

        return command

    def next_move(self, carstate, command, distances):

        # decrease speed
        if self.last_command is None or carstate.speed_x > 50:
            command.accelerator = 0.0
            command.brake = 0.5
        # increase speed
        elif carstate.speed_x < 30:
            command.accelerator = 0.5
            command.brake = 0.0
        elif min(distances) > 3:
            command.accelerator = 0.5
            command.brake = 0.0
            command.steering = command.steering + random.uniform(-5, 5)
        elif min(distances) > 1:
            command.accelerator = 1.0
            command.brake = 0.0
            if carstate.angle < 0:
                command.steering = command.steering - 1
            else:
                command.steering = command.steering + 1



        self.last_command = command
        return command

    def calculate_steering(self, distances):
        x = 3
        dangers = sorted(distances)[:x]
        steering = 0
        for danger in dangers:
            index = distances.index(danger)
            steering += (2*(index - (len(distances)/2))) / len(distances)
        return steering / x

    def switch_gear(self, carstate, command):
        """
        Naive switching of gears
        """
        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
