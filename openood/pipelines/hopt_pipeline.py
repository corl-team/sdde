import wandb
import yaml
import random
import os

from .train_pipeline import TrainPipeline
from .test_ood_pipeline import TestOODPipeline
from ..utils import Config, merge_configs


class HoptPipeline:
    def __init__(self, config) -> None:
        self.config = config
        if config.recorder.name != 'wandb':
            raise ValueError('Expected "wandb" recorder for "hopt" pipeline.')
        self.sweep_id = config.recorder.sweep_id
        self.sweep_name = config.recorder.experiment
        self.output_dir = config.output_dir
        if not self.sweep_id:
            sweep_config = yaml.safe_load(str(config.hopt_params))
            sweep_config['parameters'] = flatten_hopt_config(sweep_config['parameters'])
            sweep_config['name'] = self.sweep_name
            self.sweep_id = wandb.sweep(sweep=sweep_config, project=config.recorder.project)

    def hopt_run(self):
        run_id = random.randint(0, 10**10)
        output_dir = self.output_dir + '-' + str(run_id)
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(dir=output_dir,
                   name=self.sweep_name + '-' + self.sweep_id + '-' + str(run_id),
                   group=self.config.recorder.group or None)
        config = merge_configs(self.config, Config(dict(wandb.config)))
        config.recorder.experiment = self.sweep_name + '-' + self.sweep_id + '-' + str(run_id)
        config.merge_option = 'merge'
        config.output_dir = output_dir
        evaluator = config.evaluator.name
        pipeline = config.pipeline.name

        train_pipeline = TrainPipeline(config)
        train_pipeline.run()

        config.evaluator.name = 'ood'
        config.pipeline.name = 'test_ood'
        config.postprocessor.postprocessor_args.checkpoint_root = config.output_dir
        config.output_dir += '-eval'

        test_pipeline = TestOODPipeline(config)
        test_pipeline.run()

        config.evaluator.name = evaluator
        config.pipeline.name = pipeline

    def run(self):
        wandb.agent(self.sweep_id, project=self.config.recorder.project,
                    function=self.hopt_run, count=self.config.num_hopt_trials)


def flatten_hopt_config(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                if a != 'values':
                    flatten(x[a], name + a + '.')
                else:
                    out[name[:-1]] = x
        else:
            out[name[:-1]] = x
    flatten(y)
    return out
