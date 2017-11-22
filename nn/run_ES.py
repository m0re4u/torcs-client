import numpy as np
import subprocess
import torch
import time
import train
import os
import sys
import signal
import glob
import logging


class Evolution:
    def __init__(self, filename):
        self.model = train.TwoLayerNet(22, 15, 3)
        self.model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        self.parameters = self.model.parameters()

    def get_parameter_sets(self, standard_dev, noise_vector):
        parameter_sets = []

        for i in range(len(noise_vector)):
            new_parameter_set = self.parameters
            noise = noise_vector[i]

            for parameter in new_parameter_set:
                parameter.data += standard_dev * noise

            parameter_sets.append(new_parameter_set)

        return parameter_sets

    def compute_rewards(self, parameter_sets):
        reward_vector = []

        # Remove old drivers:
        for filename in glob.glob("../models/evol_driver*"):
            os.remove(filename)

        # Start drivers
        procs = []
        try:
            for i, param in enumerate(range(2)):
                torch.save(self.model.state_dict(), "../models/evol_driver{}.pt".format(i))

                print("Child {}".format(i))
                cmd = [
                    "python3", "../run.py",
                    # "-f", "../models/evol_driver{}.pt".format(i),
                    "-f", "../models/NNdriver.pt",
                    "-r", "../logs/data.log",
                    "-H", "15",
                    "-p", "{}".format(i + 3001)
                ]
                proc = subprocess.Popen(cmd)
                procs.append(proc)
                print("Started child with PID {}".format(proc.pid))
        except KeyboardInterrupt:
            for proc in procs:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        # Start torcs
        start = time.time()
        print("Running torcs at {}".format(start))
        cmd = ["torcs -r ~/Desktop/Parallels_Shared_Folders/CI2017/torcs-server/torcs-client/race-config/training.xml"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        print(proc.communicate())
        end = time.time()
        print("Finished torcs at {}, took {}".format(end, end - start))

        # TODO: Get rewards from torcs

        return reward_vector

    def update_parameters(self, learning_rate, reward_vector, standard_dev, noise_vector):
        n = len(reward_vector)

        gradient = 0
        for i, reward in enumerate(reward_vector):
            gradient += (1 / (n * standard_dev)) * reward * noise_vector[i]

        for parameter in self.parameters:
            parameter.data += learning_rate * gradient

    def run(self, iterations=1, population_size=20, standard_dev=0.1, learning_rate=1e-3):
        for i in range(iterations):
            print("Iteration: {}".format(i))
            noise_vector = np.random.standard_normal(population_size)

            parameter_sets = self.get_parameter_sets(
                standard_dev, noise_vector)
            reward_vector = self.compute_rewards(parameter_sets)

            self.update_parameters(learning_rate, reward_vector,
                                   standard_dev, noise_vector)


def main(filename):
    ev = Evolution(filename)
    ev.run()


if __name__ == '__main__':
    main("../models/NNdriver.pt")
