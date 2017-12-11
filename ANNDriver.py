from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import math
import numpy as np
import logging
from nn import train
from torch.autograd import Variable
import pickle
import os


class ANNDriver(Driver):
    def __init__(self, model_file, H, depth, port, record_train_file=None, normalize=False, opp=False):
        super().__init__(False)
        self.norm = normalize
        self.time = 0
        self.opp = opp

        # Swarm variables
        self.is_leader = True
        self.standard_driver = Driver()
        self.race_started = False
        self.recovers = 0
        self.recover_mode = False
        self.speeds = []

        self.swarm_info = {}
        self.swarm_info["crashes"] = []
        self.crash_recorded = False
        self.swarm_info_partner = {}
        self.port = port
        self.partner_port = -1

        # Check if opponent data is included in the input
        if self.opp:
            d_in = 22 + 36
        else:
            d_in = 22

        # Select right model
        if depth == 3:
            self.model = train.ThreeLayerNet(d_in, H, 3)
        elif depth == 5:
            self.model = train.FiveLayerNet(d_in, H, 3)
        else:
            self.model = train.TwoLayerNet(d_in, H, 3)

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

    def __del__(self):
        if self.record:
            self.file_handler.close()
            self.file_handler = None

    def drive(self, carstate: State) -> Command:
        self.swarm_communicate(carstate=carstate)

        # Select the sensors we need for our model
        if self.opp:
            sensors = [carstate.speed_x, carstate.distance_from_center,
                   carstate.angle, *(carstate.distances_from_edge), *(carstate.opponents)]
        else:
            sensors = [carstate.speed_x, carstate.distance_from_center,
                       carstate.angle, *(carstate.distances_from_edge)]

        if self.norm:
            sensors = self.normalize_sensors(sensors)

        command = Command()
        distances = list(carstate.distances_from_edge)

        # CRASHED
        off_road = all(distance == -1.0 for distance in distances)
        if self.race_started and (off_road or self.recovers > 0):
            command = self.recover(carstate, command)

            if not self.crash_recorded:
                self.crash_recorded = True
                self.dict["crashes"].append(carstate.distance_raced)
        # NOT CRASHED
        else:
            self.crash_recorded = False

            # Forward pass our model
            y = self.model(Variable(torch.Tensor(sensors)))[0]

            # Apply heuristics to the output of the neural network
            accelerate, brake, steer = self.apply_heuristics(y.data)

            # Create command from model output
            command.accelerator = np.clip(accelerate, 0, 1)
            command.brake = np.clip(brake, 0, 1)
            command.steering = np.clip(steer, -1, 1)

            # Naive switching of gear
            self.switch_gear(carstate, command)

            if self.race_started and not self.is_leader and carstate.distance_from_start > 50:
                command = self.check_swarm(command, carstate)

        if self.record is True:
            sensor_string = ",".join([str(x) for x in sensors]) + "\n"
            self.file_handler.write(str(y.data[0]) + "," + str(y.data[1]) + "," + str(y.data[2]) + "," + sensor_string)

        return command

    def switch_gear(self, carstate, command):
        """
        Naive switching of gears
        """
        if carstate.rpm > 8500:
            command.gear = carstate.gear + 1
        elif carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

    def apply_heuristics(self, data):
        accelerate = data[0]
        brake = data[1]
        steer = data[2]

        if accelerate > 0.85 and self.time < 1000:
            accelerate = 1.0
            brake = 0.0

        if accelerate > 0.98 and brake < 0.1:
            accelerate = 1.0
            brake = 0.0

        self.time += 1

        return accelerate, brake, steer

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

    def swarm_communicate(self, carstate):
        if carstate.distance_raced < 3:
            try:
                self.swarm_info["position"] = carstate.race_position
                with open("mjv_partner{}.txt".format(self.port), "wb") as handle:
                    pickle.dump(self.swarm_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                pass
            path = os.path.abspath(os.path.dirname(__file__))
            for i in os.listdir(path):
                if os.path.isfile(os.path.join(path, i)) and 'mjv_partner' in i and not str(self.port) in i:
                    self.partner_port = i.strip('mjv_partner').strip('.txt')
        else:
            self.race_started = True

            try:
                with open("mjv_partner{}.txt".format(self.partner_port), 'rb') as handle:
                    self.swarm_info_partner = pickle.load(handle)

                self.swarm_info["position"] = carstate.race_position
                with open("mjv_partner{}.txt".format(self.port), "wb") as handle:
                    pickle.dump(self.swarm_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

                if carstate.race_position > int(self.swarm_info_partner["position"]):
                    self.is_leader = False
                else:
                    self.is_leader = True
            except:
                print("Not able to read port", self.port)
                pass

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
