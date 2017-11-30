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
import random
from copy import deepcopy
from bs4 import BeautifulSoup

HIDDEN_NEURONS = 100


class Evolution:

    def __init__(self, model_file, exec_params, es_params):
        # Get the root of our project folder
        self.torcspath = os.path.dirname(os.path.realpath(__file__))
        self.modelspath = os.path.join(self.torcspath, "models")

        # Init model
        self.model = nn.train.TwoLayerNet(22, HIDDEN_NEURONS, 3)
        self.model.load_state_dict(torch.load(
            model_file, map_location=lambda storage, loc: storage))
        # Copy the Tensors in the state_dict
        self.parameters = self.model.state_dict()

        # Executable config
        self.headless = exec_params['headless']
        self.race_config = os.path.join(self.torcspath, exec_params['race_config'])

        self.iterations = es_params['iterations']
        self.population_size = es_params['population_size']
        self.standard_dev = es_params['standard_dev']
        self.learning_rate = es_params['learning_rate']

    def noise_parameter_sets(self):
        parameter_sets = []
        noise_sets = []
        for i in range(self.population_size):
            new_parameter_set = deepcopy(self.parameters)
            new_noise = []
            for param_name, param_tensor in new_parameter_set.items():
                noise = torch.Tensor(np.random.normal(size=param_tensor.size()))
                param_tensor += self.standard_dev * noise
                new_noise.append(noise)

            parameter_sets.append(new_parameter_set)
            noise_sets.append(new_noise)

        return parameter_sets, noise_sets

    def get_results(self, race):
        result_path = os.path.expanduser("~/.torcs/results")
        out_dir = os.path.join(
            result_path,
            # remove head path and extension
            '.'.join(os.path.split(race)[1].split('.')[:-1])
        )
        out_base = sorted(os.listdir(out_dir))[-1]
        out_file = os.path.join(
            out_dir,
            out_base
        )
        print("Reading results from: {}".format(out_base))

        with open(out_file) as fd:
            soup = BeautifulSoup(fd, 'xml')
        result_soup = soup.find('section', attrs={'name': 'Results'})
        rank_soup = result_soup.find('section', attrs={'name': 'Rank'})
        results = [
            (
                # Final position
                int(section['name']),
                # Driver index
                int(section.find('attnum', attrs={'name': 'index'})['val']),
                # Driver name
                section.find('attstr', attrs={'name': 'name'})['val'],
                # Driver time
                float(section.find('attnum', attrs={'name': 'time'})['val']),
                # No. of laps completed
                int(section.find('attnum', attrs={'name': 'laps'})['val'])
            )
            for section in rank_soup.findAll('section')
        ]
        return results

    def init_drivers(self, index, params):
        # Save current parameter set for the client to read in
        torch.save(params, "models/evol_driver{}.pt".format(index))
        cmd = [
            "python3", self.torcspath + "/run.py",
            "-f", (self.modelspath + "/evol_driver{}.pt").format(index),
            "-H", str(HIDDEN_NEURONS),
            "-p", "{}".format(index + 3001)
        ]
        proc = subprocess.Popen(cmd)
        return proc.pid

    def run_torcs(self):
        start = time.time()
        if self.headless:
            # Pick a random config (random track)
            race = random.choice(os.listdir(self.race_config))
            cmd = ["torcs -r " + os.path.join(self.race_config, race)]
        else:
            race = input("Select race-config (default:\"quickrace\"):")
            cmd = ["torcs"]
        print("Running torcs with race: {} at {:04.3f}".format(race, start))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
        res = proc.communicate()
        end = time.time()
        print("Finished torcs at {:04.3f}, took {:04.3f} seconds".format(end, end - start))
        results = self.get_results(race)
        return results

    def combine_results(self, results):

        rewards = []
        for rank, driver_index, _, time, laps in results:
            result = 0

            # Did not complete all labs at Aalborg track
            if laps != 3:
                rewards.append(result)
            else:
                # Did complete the laps, hence calculate the score

                # Check how many cars in front crashed
                num_cars_crashed = 0
                for rank2, driver_index2, _, time2, laps2 in results:
                    if driver_index2 < driver_index and laps2 != 3:
                        num_cars_crashed += 1

                # Time component
                result += (50000 / time)

                # Overtaking component
                start_rank = driver_index + 1
                result += 50 * (start_rank - rank - num_cars_crashed)

                # Minimum of 0
                if result < 0:
                    result = 0

                rewards.append(result)

        return rewards

    def compute_rewards(self, parameter_sets):
        reward_vector = np.zeros(self.population_size)

        # Remove old drivers:
        for filename in glob.glob("models/evol_driver*"):
            os.remove(filename)

        # Start drivers
        procs = []
        try:
            for i, param in enumerate(parameter_sets):
                proc = self.init_drivers(i, param)
                procs.append(proc)

            # Start torcs and wait for it to finish
            results = self.run_torcs()
        except Exception as e:
            print("Error: {}".format(e))
            for proc in procs:
                os.killpg(os.getpgid(proc), signal.SIGTERM)

        print("Race results:\n {}".format("\n ".join(str(r) for r in results)))

        reward_vector = self.combine_results(results)
        print("Rewards:\n {}".format(reward_vector))
        return reward_vector

    def update_parameters(self, reward_vector, noise_sets):
        gradient = []
        for _, p in self.parameters.items():
            gradient.append(torch.zeros(p.size()))
        # Per reward, update the parameters with higher scoring reward having
        # a larger weight
        for i, reward in enumerate(reward_vector):
            # Multiple sets of parameters
            for j, noise in enumerate(noise_sets[i]):
                if self.standard_dev == 0:
                    update = (1 / self.population_size) * reward * noise
                else:
                    update = ((self.population_size * self.standard_dev)) * reward * noise

                gradient[j] += update

        for i, (param_name, param_tensor) in enumerate(self.parameters.items()):
            param_tensor += self.learning_rate * gradient[i]

    def run(self):
        for i in range(self.iterations):
            print("Iteration: {}".format(i))
            # Get noised parameter sets
            parameter_sets, noise_sets = self.noise_parameter_sets()
            # Compute reward based on a simulated race
            reward_vector = self.compute_rewards(parameter_sets)
            # Update parameters using the noised parameters and the race outcome
            self.update_parameters(reward_vector, noise_sets)
            torch.save(self.parameters, "models/output_gen_end.pt")


def main(model_file, exec_params, es_params):
    ev = Evolution(model_file, exec_params, es_params)
    print("Running with ES parameters:\n {}".format(es_params))
    ev.run()


if __name__ == '__main__':
    filepath = os.path.realpath(__file__)

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "-i", "--iterations", help="Number of iterations for the ES algorithm",
        default=10, type=int
    )
    parser.add_argument(
        "-s", "--standard_dev", help="Standard deviation for the noise imposed \
        during training",
        default=1e-06, type=float
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="Learning rate of the ES algorithm",
        default=1e-06, type=float
    )
    parser.add_argument(
        "-c", "--race_config", help="Race configuration file (xml) directory. \
        This will also choose the right population size (name of subdirectory)",
        default=os.path.dirname(filepath) + "/race-config/headless/2/"
    )
    parser.add_argument(
        "-m", "--init_model", help="initial model (for pytorch)",
        default=os.path.dirname(filepath) + "/models/NNdriver2-100-300.pt"
    )
    parser.add_argument(
        "--no-headless", help="Run with graphical output",
        action="store_true"
    )

    args = parser.parse_args()
    if os.path.isdir(args.race_config):
        # use folder name to find population size
        folder = os.path.basename(os.path.abspath(args.race_config))
        popsize = int(folder)
    else:
        exit("Error determining population size! \
        {} might not be a valid config folder".format(args.race_config))

    # Parameters used in the ES algorithm
    ES_params = {
        'iterations': args.iterations,
        'population_size': popsize,
        'standard_dev': args.standard_dev,
        'learning_rate': args.learning_rate
    }
    # Parameters used in running torcs and the clients
    exec_params = {
        'race_config': args.race_config,
        'headless': not args.no_headless
    }

    main(args.init_model, exec_params, ES_params)
