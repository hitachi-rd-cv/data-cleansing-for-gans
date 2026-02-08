# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
from __future__ import division

import datetime
import os
from typing import Mapping

import gokart
import luigi
import torch

from libs.mysacred import Experiment
from libs.scp import SCPException
from libs.waluigi.tools import PyTorchPickleFileProcessor, pull_output_via_scp
from dotenv import load_dotenv

load_dotenv()


class TaskBase(gokart.TaskOnKart):
    """TaskBase
    Base class inherited by most of Tasks

    Attributes:
        workspace_directory: root directory of the output it is insignificant for determining the name of output directly
    """
    workspace_directory: str = luigi.Parameter('./.processed', significant=False)
    db_name: str = luigi.Parameter(os.environ.get('MONGO_DB', "local"), significant=False)
    mongo_auth: str = luigi.Parameter(os.environ.get('MONGO_AUTH', None), significant=False)
    memo: str = luigi.Parameter('none', significant=False)
    fix_random_seed_value = luigi.IntParameter(0)
    fix_random_seed_methods = luigi.ListParameter([
        "random.seed",
        "numpy.random.seed",
        "torch.random.manual_seed",
        "torch.cuda.manual_seed_all",
    ])
    scp: bool = luigi.BoolParameter(significant=False)
    tags: list = luigi.ListParameter([], significant=False)
    _func_run: staticmethod

    @luigi.Task.event_handler(luigi.Event.START)
    def make_torch_deterministic(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def run_in_sacred_experiment(self, f, **kwargs):
        ex = Experiment(self.__class__.__name__, db_name=self.db_name, mongo_auth=self.mongo_auth, base_dir=os.path.abspath(os.path.curdir))
        ex.main(f)
        param_kwargs_sig = self.get_all_required_params(self)
        ex.add_config(param_kwargs_sig)
        ex.add_config({'seed': self.fix_random_seed_value})
        ex.add_config({'tags': self.tags})
        run = ex._create_run(bypassed_config=kwargs)
        return run()

    def output(self) -> object:
        '''
        do not overwrite this class in the child classes.
        this is executed in self.run to get unique output directly.
        Each combination of the tasks parameter leads its unique hash contained in self.tasks id.
        It enables output directory to be determined automatically and ensures the same tasks with different parameters are never overwritten.

        Returns: gokart.target.Target

        '''
        return self.make_target(f'{self.task_family}.pt', processor=PyTorchPickleFileProcessor())

    def make_and_get_temporary_directory(self):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        dir_tmp = os.path.join(self.workspace_directory, f'{self.task_family}_{self.task_unique_id}_{timestamp}')
        os.makedirs(dir_tmp, exist_ok=True)
        return dir_tmp

    def complete(self) -> bool:
        if self._rerun_state:
            for target in luigi.task.flatten(self.output()):
                target.remove()
            self._rerun_state = False
            return False

        are_exists = []
        for output in luigi.task.flatten(self.output()):
            if output.exists():
                are_exists.append(True)
            else:
                if self.scp:
                    os.makedirs(self.workspace_directory, exist_ok=True)
                    try:
                        pull_output_via_scp(output)
                    except SCPException as e:
                        print(e)
                        are_exists.append(False)
                    else:
                        are_exists.append(True)
                else:
                    are_exists.append(False)

        # is_completed = all([t.exists() for t in luigi.tasks.flatten(self.output())])
        is_completed = all(are_exists)

        if self.strict_check or self.modification_time_check:
            requirements = luigi.task.flatten(self.requires())
            inputs = luigi.task.flatten(self.input())
            is_completed = is_completed and all([task.complete() for task in requirements]) and all([i.exists() for i in inputs])

        if not self.modification_time_check or not is_completed or not self.input():
            return is_completed

        return self._check_modification_time()

    @classmethod
    def recursively_make_dict(cls, value):
        if isinstance(value, Mapping):
            return dict(((k, cls.recursively_make_dict(v)) for k, v in value.items()))
        return value

    def load(self, target=None):
        def _load(targets):
            if isinstance(targets, list) or isinstance(targets, tuple):
                return [_load(t) for t in targets]
            if isinstance(targets, dict):
                return {k: _load(t) for k, t in targets.items()}
            print(targets.path())
            return targets.load()

        return _load(self._get_input_targets(target))

    @classmethod
    def get_all_required_params(cls, task_instance):
        """
        Recursively get all the significant parameters and their values required by the given task instance, excluding parameters that are instances of `gokart.TaskInstanceParameter` or `gokart.ListTaskInstanceParameter`.
        """
        required_params = {}
        for required_task in luigi.task.flatten(task_instance.requires()):
            required_params.update(cls.get_all_required_params(required_task))
        for param_name, param in task_instance.get_params():
            if not isinstance(param, (gokart.TaskInstanceParameter, gokart.ListTaskInstanceParameter)) and param.significant:
                required_params[param_name] = getattr(task_instance, param_name)
        return required_params
