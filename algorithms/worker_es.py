import ray
import wandb
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from ray.rllib.utils.filter import MeanStdFilter

from algorithms.worker_ea import EAWorker
from utils.chromosome import VBNChromosome
import numpy as np

from utils.pettingzooenv import PLAYER_1_ID, PLAYER_2_ID


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ESWorker(EAWorker):
    """ Worker class for the Coevolutionary Evolution Strategies.
    This class handles both the evaluation and mutation of individuals.
    After evaluation, the results are communicated back to the Trainer"""

    def __init__(self, config, env_creator, worker_index):
        super().__init__(config, env_creator, worker_index)

        self.elite = VBNChromosome(number_actions=self.config['number_actions'],
                                   input_channels=self.config['input_channels'])
        self.oponent = VBNChromosome(number_actions=self.config['number_actions'],
                                     input_channels=self.config['input_channels'])
        self.filter = MeanStdFilter(self.env.reset()[PLAYER_2_ID].shape)
        self.video_path = f'/tmp/video_worker_{worker_index}.mp4'
        self.video_path_eval = f'/tmp/video_worker_{worker_index}_eval.mp4'

    def mutate(self, weights, record):
        """ Mutate the inputted weights and evaluate its performance against the
        weights of the previous generation. """
        recorder = VideoRecorder(self.env, path=self.video_path) if record else None
        self.elite.set_weights(weights)
        self.oponent.set_weights(weights)
        perturbations = self.oponent.mutate(self.config['mutation_power'])

        _, oponent_reward1, ts1 = self.play_game(self.elite, self.oponent,
                                                 recorder=recorder)
        oponent_reward2, _, ts2 = self.play_game(self.oponent, self.elite,
                                                 recorder=recorder)

        if record:
            recorder.close()

        return {
            'total_reward': np.mean([oponent_reward1, oponent_reward2]),
            'timesteps_total': ts1 + ts2,
            'video': None if not record else wandb.Video(self.video_path),
            'noise': perturbations
        }
