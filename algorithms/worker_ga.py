import ray
import wandb
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from ray.rllib.utils.filter import MeanStdFilter

from algorithms.worker_ea import EAWorker
from utils.chromosome import VBNChromosome

from utils.pettingzooenv import PLAYER_1_ID, PLAYER_2_ID


@ray.remote(max_restarts=-1, max_task_retries=-1)
class GAWorker(EAWorker):
    """ Worker class for the Coevolutionary Genetic Algorithm.
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

    def evaluate_mutations(self, elite, oponent, record=False, mutate_oponent=True):
        recorder = VideoRecorder(self.env, path=self.video_path) if record else None
        self.elite.set_weights(elite)
        self.oponent.set_weights(oponent)
        if mutate_oponent:
            self.oponent.mutate(self.config['mutation_power'])
        elite_reward1, oponent_reward1, ts1 = self.play_game(
            self.elite, self.oponent, recorder=recorder)
        oponent_reward2, elite_reward2, ts2 = self.play_game(
            self.oponent, self.elite, recorder=recorder)
        total_elite = elite_reward1 + elite_reward2
        total_oponent = oponent_reward1 + oponent_reward2
        if record:
            recorder.close()
        return {
            'oponent_weights': self.oponent.get_weights(),
            'score_vs_elite': total_oponent,
            'timesteps_total': ts1 + ts2,
            'video': wandb.Video(self.video_path) if record else None,
        }
