from random import random
from ray.rllib import MultiAgentEnv
from supersuit import frame_stack_v1, resize_v0, frame_skip_v0, agent_indicator_v0
# from pettingzoo.atari import pong_v1, boxing_v1
from ray.tune import register_env

PLAYER_1_ID = 'first_0'
PLAYER_2_ID = 'second_0'


class ParallelPettingZooEnv(MultiAgentEnv):
    """ This class makes sure that pettingzoo's envs can be used in combination with
    ray. Some parts of this code are from one of PettingZoo's repositories. """

    def __init__(self, env, random_action=None, random_proba=0.05):
        self.par_env = env
        # agent idx list
        self.agents = self.par_env.possible_agents

        # Random actions to get the environment unstuck.
        self.random_action = random_action
        self.random_proba = random_proba

        # Get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.par_env.observation_spaces
        self.action_spaces = self.par_env.action_spaces

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.observation_spaces[self.agents[0]]

        # Get first action space, assuming all agents have equal space
        self.action_space = self.action_spaces[self.agents[0]]

        assert all(obs_space == self.observation_space
                   for obs_space
                   in self.par_env.observation_spaces.values()), \
            "Observation spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_observations wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_observations(env)`"

        assert all(act_space == self.action_space
                   for act_space in self.par_env.action_spaces.values()), \
            "Action spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_action_space wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.metadata = {"render.modes": ["rgb_array"]}
        self.reset()

    def reset(self):
        """ Reset the environment. """
        return self.par_env.reset()

    def step(self, action_dict):
        """ Take one step in the environment. """

        # Take `random_action` to get the environment unstuck.
        if self.random_action and random() < self.random_proba:
            action_dict = {
                PLAYER_1_ID: self.random_action,
                PLAYER_2_ID: self.random_action
            }
        for agent in self.agents:
            if agent not in action_dict:
                action_dict[agent] = self.action_space.sample()
        aobs, arew, adones, ainfo = self.par_env.step
        obss = {}
        rews = {}
        dones = {}
        infos = {}
        for agent in action_dict:
            # sometimes Joust does not include one of the obs
            if agent not in aobs:
                for agent in action_dict:
                    rews[agent] = 0
                    dones[agent] = True
                break

            obss[agent] = aobs[agent]
            rews[agent] = arew[agent]
            dones[agent] = adones[agent]
            infos[agent] = ainfo[agent]
        self.rewards = rews
        self.dones = dones
        self.infos = infos
        dones["__all__"] = all(adones.values())
        return obss, rews, dones, infos

    def close(self):
        """ Close the environment """
        self.par_env.close()

    def seed(self, seed=None):
        """ Seed the environment """
        self.par_env.seed(seed)

    def render(self, mode="human"):
        """ Render the game frame """
        return self.par_env.render(mode)


def register_pettingzoo_env(env_name):
    """ Register the Env including preprocessing pipeline, s.t. it can be easily
    imported using ray. """

    def get_env(config):
        name = env_name.replace('-', '_')
        env = __import__(f'pettingzoo.atari.{name}', fromlist=[None])
        env = env.parallel_env(obs_type='grayscale_image')
        env = frame_skip_v0(env, 4)
        env = resize_v0(env, 84, 84)
        env = frame_stack_v1(env, 4)
        env = agent_indicator_v0(env)
        return ParallelPettingZooEnv(env,
                                     random_action=config['random_action'],
                                     random_proba=config['random_action_probability'])

    print(f'Registering env with name {env_name}')
    register_env(env_name, lambda config: get_env(config))
