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
import pickle
import os.path


class ANNDriverJasper(Driver):
    def __init__(self, model_file, H, depth, port, record_train_file=None, normalize=False):
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

        self.dict = {}
        self.dict["crashes"] = []
        self.crash_recorded = False
        self.dict_teammate = {}
        self.port = port
        self.partner = -1

        path = os.path.abspath(os.path.dirname(__file__))
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path, i)) and 'mjv_partner' in i:
                try:
                    os.remove(path + "/" + i)
                except:
                    pass

    def __del__(self):
        if self.record:
            self.file_handler.close()
            self.file_handler = None

    def drive(self, carstate: State) -> Command:
        if carstate.distance_raced < 3:
            try:
                self.dict["position"] = carstate.race_position
                with open("mjv_partner{}.txt".format(self.port), "wb") as handle:
                    pickle.dump(self.dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                pass
            path = os.path.abspath(os.path.dirname(__file__))
            for i in os.listdir(path):
                if os.path.isfile(os.path.join(path, i)) and 'mjv_partner' in i and not str(self.port) in i:
                    self.partner = i.strip('mjv_partner').strip('.txt')
        else:
            self.race_started = True

            try:
                with open("mjv_partner{}.txt".format(self.partner), 'rb') as handle:
                    self.dict_teammate = pickle.load(handle)

                self.dict["position"] = carstate.race_position
                with open("mjv_partner{}.txt".format(self.port), "wb") as handle:
                    pickle.dump(self.dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # print(self.port, self.dict_teammate["position"])
                # print("RESULT", carstate.race_position > int(self.dict_teammate["position"]))
                if carstate.race_position > int(self.dict_teammate["position"]):
                    self.is_leader = False
                else:
                    self.is_leader = True
            except:
                print("Not able to read port", self.port)
                pass

            # print(self.dict_teammate)
            # print(self.port, self.leader)

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

            if not self.crash_recorded:
                self.crash_recorded = True
                self.dict["crashes"].append(carstate.distance_raced)

        # NOT CRASHED
        else:
            self.crash_recorded = False

            if carstate.gear == -1:
                carstate.gear = 2
            if self.norm:
                # Sensor normalization -> ?
                sensors = self.normalize_sensors(sensors)

            # Forward pass our model
            y = self.model(Variable(torch.Tensor(sensors)))[0]
            # Create command from model output

            command.accelerator = np.clip(y.data[0], 0, 1)
            command.brake = np.clip(y.data[1], 0, 1)
            command.steering = np.clip(y.data[2], -1, 1)

            command = self.smooth_commands(command)

            # print(self.race_started and not self.is_leader)
            # print("LEADER", self.is_leader)
            if self.race_started and not self.is_leader and carstate.distance_from_start > 50:
                command = self.check_swarm(command, carstate)

            # Naive switching of gear
            self.switch_gear(carstate, command)

        # print("Accelerate: {}".format(command.accelerator))
        # print("Brake:      {}".format(command.brake))
        # print("Steer:      {}".format(command.steering))

        if self.record is True:
            sensor_string = ",".join([str(x) for x in sensors]) + "\n"
            self.file_handler.write(str(y.data[0]) + "," + str(y.data[1]) + "," + str(y.data[2]) + "," + sensor_string)
        return command

    def check_swarm(self, command, carstate):
        for crash in self.dict_teammate["crashes"]:
            if crash - 100 < carstate.distance_raced < crash - 30:
                if command.accelerator > 0.9:
                    command.accelerator = 0.5

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
        if (dist < 0.1 and self.recovers > 35) or self.recovers > 50:
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

    def smooth_commands(self, command):
        if command.accelerate > 0.85 and self.time < 1000:
            command.accelerate = 1.0
            command.brake = 0.0

        if command.accelerate > 0.98 and brake < 0.1:
            command.accelerate = 1.0
            command.brake = 0.0

        self.time += 1

        return command


class StupidDriver(Driver):
    def __init__(self, record):
        super().__init__(False)
        self.last_command = None
        self.record = record
        self.steering_switched = False
        self.steering_sign = -1

    def __del__(self):
        if self.record:
            self.file_handler.close()
            self.file_handler = None

    def drive(self, command, distances, carstate: State) -> Command:

        # Create command from model output
        command = self.next_move(carstate, command, distances)

        return command

    def next_move(self, carstate, command, distances):
        # if command.brake < 0.9:
        #     if carstate.speed_x > 60:
        #         command.brake = 1
        #         command.accelerator = 0.0
        #     else:
        #         command.brake = 0
        #         command.accelerator = 1.0
        #     if min(distances) < 4:
        #         if not self.steering_switched:
        #             self.steering_switched = True
        #             if self.steering_sign == -1:
        #                 self.steering_sign = 1
        #             else:
        #                 self.steering_sign = -1
        #
        #         command.steering = 0.7 * self.steering_sign
        #     elif abs(carstate.angle) < 0.1:
        #         command.steering = 0.1 * self.steering_sign
        #         self.steering_switched = False
        #     else:
        #         self.steering_switched = False
        #         command.steering = 0

        return command

    # def next_move(self, carstate, command, distances):
    #     print("ANGLE", carstate.angle)
    #     # decrease speed
    #     if self.last_command is None or carstate.speed_x > 50:
    #         command.accelerator = 0.0
    #         command.brake = 0.5
    #     # increase speed
    #     elif carstate.speed_x < 30:
    #         command.accelerator = 0.5
    #         command.brake = 0.0
    #     elif min(distances) > 3:
    #         command.accelerator = 0.5
    #         command.brake = 0.0
    #         command.steering = command.steering + random.uniform(-5, 5)
    #     elif min(distances) > 2:
    #         self.steering_timer += 1
    #
    #         command.accelerator = 1.0
    #         command.brake = 0.0
    #
    #         if self.steering_timer == 0:
    #             if carstate.angle < 0:
    #                 self.steering_sign = "MINUS"
    #             else:
    #                 self.steering_sign = "PLUS"
    #
    #         if self.steering_timer > 0:
    #             if self.steering_sign == "MINUS":
    #                 command.steering = -1
    #             else:
    #                 command.steering = 1
    #
    #     if min(distances) > 3:
    #         self.steering_timer = 0
    #
    #     self.last_command = command
    #     return command

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
