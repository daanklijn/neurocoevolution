from pathlib import Path

import wandb
import numpy as np
import ray
import yaml
from ray.rllib.agents import with_common_config

from algorithms.trainer_ea import EATrainer
from utils.chromosome import VBNChromosome

DEFAULT_CONFIG = with_common_config(yaml.load(
    Path('configs/config_es_default.yaml').read_text()
))


class ESTrainer(EATrainer):
    """ Trainer class for the Coevolutionary Evolution Strategies algorithm.
    This class distributes the mutation and evaluation workload over a number
    of workers and updates the network weights."""

    _name = "ES"
    _default_config = DEFAULT_CONFIG

    def _init(self, config, env_creator):
        super(ESTrainer, self)._init(config, env_creator)
        self.weights = VBNChromosome(number_actions=self.config['number_actions'],
                                     input_channels=self.config['input_channels']
                                     )
        self.weights.virtual_batch_norm(self.collect_samples())

    def step(self):
        """ Evolve one generation using the Evolution Strategies algorithm.
        This consists of four steps:
        1. Send the current weights to a number of workers and mutate and evaluate them.
        2. Communicate the mutated weights and their fitness back to the Trainer.
        3. Update the weights using the ES update rule.
        4. Evaluate the updated weights against a random policy and log the outcome.
        """
        worker_jobs = []
        for i in range(self.config['population_size']):
            worker_id = i % self.config['num_workers']
            record = i < self.config['num_train_videos']
            worker_jobs += [self._workers[worker_id].mutate.remote(
                self.weights.get_weights(), record)]

        results = ray.get(worker_jobs)
        rewards = [result['total_reward'] for result in results]
        noises = [result['noise'] for result in results]

        normalized_rewards = self.normalize_rewards(rewards)
        weight_update = self.compute_weight_update(noises, normalized_rewards)
        weights = self.weights.get_perturbable_weights()
        self.weights.set_perturbable_weights(weights + weight_update)
        winner_file = self.try_save_winner(self.weights.get_weights())

        evaluate_results = self.evaluate_current_weights(self.weights.get_weights())
        evaluate_rewards = [result['total_reward'] for result in evaluate_results]
        evaluate_videos = [result['video'] for result in evaluate_results]

        self.increment_metrics(results)

        summary = dict(
            timesteps_total=self.timesteps_total,
            episodes_total=self.episodes_total,
            train_reward_min=np.min(rewards),
            train_reward_mean=np.mean(rewards),
            train_reward_max=np.max(rewards),
            train_top_5_reward_avg=np.mean(np.sort(rewards)[-5:]),
            evaluate_reward_min=np.min(evaluate_rewards),
            evaluate_reward_mean=np.mean(evaluate_rewards),
            evaluate_reward_med=np.median(evaluate_rewards),
            evaluate_reward_max=np.max(evaluate_rewards),
            avg_timesteps_train=np.mean(
                [result['timesteps_total'] for result in results]),
            avg_timesteps_evaluate=np.mean(
                [result['timesteps_total'] for result in evaluate_results]),
            eval_max_video=evaluate_videos[np.argmax(evaluate_rewards)],
            eval_min_video=evaluate_videos[np.argmin(evaluate_rewards)],
            winner_file=wandb.Video(winner_file) if winner_file else None
        )
        self.add_videos_to_summary(results, summary)
        return summary

    def compute_weight_update(self, noises, normalized_rewards):
        """ Compute the weight update using the update rule from the OpenAI ES. """
        config = self.config
        factor = config['learning_rate'] / (
                config['population_size'] * config['mutation_power'])
        weight_update = factor * np.dot(np.array(noises).T, normalized_rewards)
        return weight_update

    def normalize_rewards(self, rewards):
        """ Normalize the rewards using z-normalization. """
        rewards = np.array(rewards)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        if reward_std == 0:
            return rewards - reward_mean
        else:
            return (rewards - reward_mean) / reward_std
