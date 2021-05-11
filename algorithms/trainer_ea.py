import os
from abc import ABCMeta, abstractmethod

import numpy as np
import ray
from ray.rllib.agents import Trainer, with_common_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.filter import MeanStdFilter

from algorithms.worker_es import ESWorker
from algorithms.worker_ga import GAWorker
from utils.pettingzooenv import PLAYER_1_ID, PLAYER_2_ID
from utils.policies import RandomPolicy


class EATrainer(Trainer):
    """ Class that includes some functionality that is used by both the
    Evolution Strategies and Genetic Algorithm Trainers. """
    __metaclass__ = ABCMeta

    @override(Trainer)
    def _init(self, config, env_creator):
        self.config = config
        self.env_creator = env_creator
        worker_class = GAWorker if config['algorithm'] == 'ga' else ESWorker
        self._workers = [
            worker_class.remote(config, env_creator, idx + 1)
            for idx in range(config["num_workers"])
        ]

        self.episodes_total = 0
        self.timesteps_total = 0
        self.generation = 0

    def collect_samples(self):
        """ Sample game frames from the environment by letting two random policies
        play against eachother. """
        env = self.env_creator(self.config)
        obs = env.reset()
        obs_filter = MeanStdFilter(obs[PLAYER_1_ID].shape)
        policy = RandomPolicy(self.config['number_actions'])
        samples = []
        for _ in range(500):
            obs, _, done, _ = env.step
            samples += [obs_filter(obs[PLAYER_1_ID]), obs_filter(obs[PLAYER_2_ID])]
            if done[PLAYER_1_ID]:
                env.reset()
        return samples

    @abstractmethod
    @override(Trainer)
    def step(self):
        """ Should be overwritten by child class. """
        pass

    def try_save_winner(self, winner_weights):
        """ Save the best weights to a file. """
        if not os.path.exists('results'):
            os.mkdir('results')
        filename = f'results/winner_weights_generation_{self.generation}.npy'
        np.save(filename, winner_weights)
        filename = f'/tmp/winner_weights_generation_{self.generation}.npy.mp4'
        with open(filename, 'wb+') as file:
            np.save(file, winner_weights)
        return filename

    def add_videos_to_summary(self, results, summary):
        """ Add videos to the summary dictionary s.t. they can be logged to the wandb
        framework. """
        for i, result in enumerate(results):
            video = result['video']
            if video:
                summary[f'train_video_{i}'] = results[i]['video']

    def evaluate_current_weights(self, best_mutation_weights):
        """ Send the weights to a number of workers and ask them to evaluate the weights. """
        evaluate_jobs = []
        for i in range(self.config['evaluation_games']):
            worker_id = i % self.config['num_workers']
            evaluate_jobs += [self._workers[worker_id].evaluate.remote(
                best_mutation_weights)]
        evaluate_results = ray.get(evaluate_jobs)
        return evaluate_results

    def increment_metrics(self, results):
        """ Increment the total timesteps, episodes and generations. """
        self.timesteps_total += sum([result['timesteps_total'] for result in results])
        self.episodes_total += len(results)
        self.generation += 1
