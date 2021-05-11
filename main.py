from pathlib import Path

import yaml
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from algorithms.trainer_es import ESTrainer
from algorithms.trainer_ga import GATrainer
from utils.pettingzooenv import register_pettingzoo_env

config = yaml.load(
    Path('configs/config_ga_test.yaml').read_text()
)
register_pettingzoo_env(config['env'])

tune.run(
    ESTrainer if config['algorithm'] == 'es' else GATrainer,
    name=config['name'],
    stop=config['stop_criteria'],
    loggers=[WandbLogger] if 'wandb' in config['logger_config'] else None,
    config=config
)
