from pathlib import Path

import numpy as np
import ray
import yaml
from ray.rllib.agents import with_common_config

from algorithms.trainer_ea import EATrainer
from utils.chromosome import VBNChromosome

DEFAULT_CONFIG = with_common_config(yaml.load(
    Path('configs/config_ga_default.yaml').read_text()
))


class GATrainer(EATrainer):
    _name = "GA"
    _default_config = DEFAULT_CONFIG

    def _init(self, config, env_creator):
        """ Trainer class for the Coevolutionary Genetic Algorithm.
        This class distributes the mutation and evaluation workload over a number
        of workers and updates and maintains the population."""

        super(GATrainer, self)._init(config, env_creator)

        self.elites = [VBNChromosome(number_actions=self.config['number_actions'],
                                     input_channels=self.config['input_channels'])
                       for _ in range(config['number_elites'])]
        samples = self.collect_samples()
        for chrom in self.elites:
            chrom.virtual_batch_norm(samples)

        self.hof = [self.elites[i].get_weights() for i in
                    range(self.config['number_elites'])]
        self.winner = None

    @property
    def step(self):
        """ Evolve the next generation using the Genetic Algorithm. This process
        consists of three steps:
        1. Communicate the elites of the previous generation
        to the workers and let them mutate and evaluate them against individuals from
        the Hall of Fame. To include a form of Elitism, not all elites are mutated.
        2. Communicate the mutated weights and fitnesses back to the trainer and
        determine which of the individuals are the fittest. The fittest individuals
        will form the elites of the next population.
        3. Evaluate the fittest
        individual against a random policy and log the results. """

        # Evaluate mutations vs first hof
        worker_jobs = []
        for i in range(self.config['population_size']):
            worker_id = i % self.config['num_workers']
            elite_id = i % self.config['number_elites']
            should_mutate = (i > self.config['number_elites'])
            should_record = (i < self.config['num_train_videos'])
            worker_jobs += [
                self._workers[worker_id].evaluate_mutations.remote(self.hof[-1],
                                                                   self.elites[
                                                                       elite_id].get_weights(),
                                                                   record=should_record,
                                                                   mutate_oponent=should_mutate)]
        results = ray.get(worker_jobs)

        # Evaluate vs other hof
        worker_jobs = []
        for j in range(len(self.hof) - 1):
            for i in range(self.config['population_size']):
                worker_id = len(worker_jobs) % self.config['num_workers']
                worker_jobs += [
                    self._workers[worker_id].evaluate_mutations.remote(self.hof[-2 - j],
                                                                       results[i][
                                                                           'oponent_weights'],
                                                                       record=False,
                                                                       mutate_oponent=False)]

        results += ray.get(worker_jobs)
        rewards = []
        print(len(results))
        for i in range(self.config['population_size']):
            total_reward = 0
            for j in range(self.config['number_elites']):
                reward_index = self.config['population_size'] * j + i
                total_reward += results[reward_index]['score_vs_elite']
            rewards.append(total_reward)

        best_mutation_id = np.argmax(rewards)
        best_mutation_weights = results[best_mutation_id]['oponent_weights']
        print(f"Best mutation: {best_mutation_id} with reward {np.max(rewards)}")

        self.try_save_winner(best_mutation_weights)
        self.hof.append(best_mutation_weights)

        new_elite_ids = np.argsort(rewards)[-self.config['number_elites']:]
        print(f"TOP mutations: {new_elite_ids}")
        for i, elite in enumerate(self.elites):
            mutation_id = new_elite_ids[i]
            elite.set_weights(results[mutation_id]['oponent_weights'])

        # Evaluate best mutation vs random agent
        evaluate_results = self.evaluate_current_weights(best_mutation_weights)
        evaluate_rewards = [result['total_reward'] for result in evaluate_results]

        train_rewards = [result['score_vs_elite'] for result in results]
        evaluate_videos = [result['video'] for result in evaluate_results]

        self.increment_metrics(results)

        summary = dict(
            timesteps_total=self.timesteps_total,
            episodes_total=self.episodes_total,
            train_reward_min=np.min(train_rewards),
            train_reward_mean=np.mean(train_rewards),
            train_reward_med=np.median(train_rewards),
            train_reward_max=np.max(train_rewards),
            train_top_5_reward_avg=np.mean(np.sort(train_rewards)[-5:]),
            evaluate_reward_min=np.min(evaluate_rewards),
            evaluate_reward_mean=np.mean(evaluate_rewards),
            evaluate_reward_med=np.median(evaluate_rewards),
            evaluate_reward_max=np.max(evaluate_rewards),
            avg_timesteps_train=np.mean(
                [result['timesteps_total'] for result in results]),
            avg_timesteps_evaluate=np.mean(
                [result['timesteps_total'] for result in evaluate_results]),
            eval_max_video=evaluate_videos[np.argmax(evaluate_rewards)],
            eval_min_video=evaluate_videos[np.argmax(evaluate_rewards)],
        )

        self.add_videos_to_summary(results, summary)
        return summary
