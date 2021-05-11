from abc import abstractmethod, ABCMeta

import wandb
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import numpy as np

from utils.pettingzooenv import PLAYER_1_ID, PLAYER_2_ID
from utils.policies import RandomPolicy
import tensorflow as tf

tf.compat.v1.enable_eager_execution()


class EAWorker:
    """ Class that includes some functionality that is used by both the
    Evolution Strategies and Genetic Algorithm Workers. """

    __metaclass__ = ABCMeta

    def __init__(self,
                 config,
                 env_creator,
                 worker_index):

        self.config = config
        self.env = env_creator(config)
        print(f"Hello world from worker {worker_index}")

    def evaluate(self, weights):
        recorder = VideoRecorder(self.env, path=self.video_path_eval)
        self.elite.set_weights(weights)
        reward, _, ts = self.play_game(self.elite,
                                       RandomPolicy(self.config['number_actions']),
                                       recorder=recorder,
                                       eval=True)
        recorder.close()
        return {
            'total_reward': reward,
            'timesteps_total': ts,
            'video': wandb.Video(self.video_path_eval),
        }

    def play_game(self, player1, player2, recorder=None, eval=False):
        obs = self.env.reset()
        reward1 = 0
        reward2 = 0
        limit = self.config['max_evaluation_steps'] if eval else self.config[
            'max_timesteps_per_episode']
        for ts in range(limit):
            filtered_obs1 = np.array([self.filter(obs[PLAYER_1_ID])])
            filtered_obs2 = np.array([self.filter(obs[PLAYER_2_ID])])
            action1 = player1.determine_actions(filtered_obs1)
            action2 = player2.determine_actions(filtered_obs2)
            obs, reward, done, info = self.env.step({
                PLAYER_1_ID: action1[0],
                PLAYER_2_ID: action2[0]
            })
            if self.config['render']:
                self.env.render()
            if recorder:
                recorder.capture_frame()
            reward1 += reward[PLAYER_1_ID]
            reward2 += reward[PLAYER_2_ID]
            if done[PLAYER_1_ID] or done[PLAYER_2_ID]:
                break
        return reward1, reward2, ts
