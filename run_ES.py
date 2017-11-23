import numpy as np
import subprocess
import torch
import time
import nn.train
import os
import sys
import signal
import glob
import logging
import argparse
from bs4 import BeautifulSoup


class Evolution:

    def __init__(self, model_file, exec_params):
        # Get the root of our project folder
        self.torcspath = os.path.dirname(os.path.realpath(__file__))

        # Init model
        self.model = nn.train.TwoLayerNet(22, 15, 3)
        self.model.load_state_dict(torch.load(
            model_file, map_location=lambda storage, loc: storage))
        self.parameters = self.model.parameters()

        # Executable config
        self.headless = exec_params['headless']
        self.race_config = os.path.join(self.torcspath, exec_params['race_config'])

    def get_parameter_sets(self, standard_dev, noise_vector):
        parameter_sets = []
        for i in range(len(noise_vector)):
            new_parameter_set = self.parameters
            noise = noise_vector[i]

            for parameter in new_parameter_set:
                parameter.data += standard_dev * noise

            parameter_sets.append(new_parameter_set)

        return parameter_sets

    def get_results(self):
        result_path = os.path.expanduser("~/.torcs/results")
        out_dir = os.path.join(
            result_path,
            # remove head path and extension
            '.'.join(os.path.split(self.race_config)[1].split('.')[:-1])
        )
        out_base = sorted(os.listdir(out_dir))[-1]
        out_file = os.path.join(
            out_dir,
            out_base
        )
        print("Reading results from: {}".format(out_file))

        with open(out_file) as fd:
            soup = BeautifulSoup(fd, 'xml')
        result_soup = soup.find('section', attrs={'name': 'Results'})
        rank_soup = result_soup.find('section', attrs={'name': 'Rank'})
        ranks = [
            (
                int(section['name']),
                section.find('attstr', attrs={'name': 'name'})['val']
            )
            for section in rank_soup.findAll('section')
        ]
        times = [
            (
                int(section['name']),
                float(section.find('attnum', attrs={'name': 'time'})['val'])
            )
            for section in rank_soup.findAll('section')
        ]
        ranking = list(zip(*sorted(ranks)))
        # contains [(ordered list of driver ids, ordered list of driver names)]
        print(ranking)
        # contains [(driverid, time)]
        print(times)

    def compute_rewards(self, parameter_sets):
        reward_vector = []

        # Remove old drivers:
        for filename in glob.glob("models/evol_driver*"):
            os.remove(filename)

        # Start drivers
        procs = []
        try:
            for i, param in enumerate(parameter_sets):
                # torch.save(param, "models/evol_driver{}.pt".format(i))
                cmd = [
                    "python3", self.torcspath + "/run.py",
                    # "-f", self.modelspath + "/evol_driver{}.pt".format(i),
                    "-f", "models/NNdriver.pt",
                    "-H", "15",
                    "-p", "{}".format(i + 3001)
                ]
                proc = subprocess.Popen(cmd)
                procs.append(proc)
                print("Started child {} with PID {} on port {}".format(
                    i, proc.pid, 3001 + i))
        except KeyboardInterrupt:
            for proc in procs:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        # Start torcs
        start = time.time()
        print("Running torcs at {}".format(start))
        if self.headless:
            cmd = ["torcs -r " + self.race_config]
        else:
            cmd = ["torcs"]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
        res = proc.communicate()
        end = time.time()
        print("Finished torcs at {}, took {}".format(end, end - start))
        print("torcs output:")
        print("-------")
        for line in res[0].split("\n"):
            print("[OUTPUT] {}".format(line))
        print("-------")
        self.get_results()

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


def main(model_file, exec_params, es_params):
    ev = Evolution(model_file, exec_params)
    ev.run(
        iterations=es_params["iterations"],
        population_size=es_params["population_size"],
        standard_dev=es_params["standard_dev"],
        learning_rate=es_params["learning_rate"]
    )


if __name__ == '__main__':
    filepath = os.path.realpath(__file__)

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "-i", "--iterations", help="Number of iterations for the ES algorithm",
        default=10, type=int
    )
    parser.add_argument(
        "-p", "--population_size", help="Number of drivers for each iteration",
        default=20, type=int
    )
    parser.add_argument(
        "-s", "--standard_dev", help="",
        default=0.1, type=float
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="Learning rate of the ES algorithm",
        default=1e-3, type=float
    )
    parser.add_argument(
        "-c", "--race_config", help="race configuration file (xml)",
        default=os.path.dirname(filepath) + "/race-config/training.xml"
    )
    parser.add_argument(
        "-m", "--init_model", help="initial model (xml)",
        default=os.path.dirname(filepath) + "/models/NNdriver.pt"
    )
    parser.add_argument(
        "--no-headless", help="Run with graphical output",
        action="store_true"
    )

    args = parser.parse_args()
    # Parameters used in the ES algorithm
    ES_params = {
        'iterations': args.iterations,
        'population_size': args.population_size,
        'standard_dev': args.standard_dev,
        'learning_rate': args.learning_rate
    }

    # Parameters used in running torcs and the clients
    exec_params = {
        'race_config': args.race_config,
        'headless': not args.no_headless
    }
    main(args.init_model, exec_params, ES_params)
