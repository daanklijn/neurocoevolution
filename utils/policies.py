from random import randint

class RandomPolicy(object):
    """ Policy that samples a random move from the number of actions that
    are available."""

    def __init__(self, number_actions):
        self.number_actions = number_actions

    def determine_actions(self, inputs):
        return [randint(0, self.number_actions - 1)]


