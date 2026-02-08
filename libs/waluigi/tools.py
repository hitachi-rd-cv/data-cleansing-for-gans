# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------
from __future__ import division

import logging
import os
import sys
import warnings
from collections import OrderedDict
from typing import Iterable, Any, Optional

import luigi
import torch
from gokart.build import _get_output, _reset_register, LoggerConfig, GokartBuildError
from gokart.file_processor import FileProcessor
from gokart.task import TaskOnKart
from luigi.task import flatten
from luigi.tools.deps_tree import bcolors
from paramiko import WarningPolicy, RSAKey, SSHClient

from libs.scp import SCPClient


def print_tree(task, indent='', last=True):
    '''
    Return a string representation of the tasks, their statuses/parameters in a dependency tree format
    '''
    # dont bother printing out warnings about tasks with no output
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Task .* without outputs has no custom complete\\(\\) method')
        is_task_complete = task.complete()
    is_complete = (bcolors.OKGREEN + 'COMPLETE' if is_task_complete else bcolors.OKBLUE + 'PENDING') + bcolors.ENDC
    name = f"{task.task_family}_{task.make_unique_id()}"
    params = task.to_str_params(only_significant=True)
    result = '\n' + indent
    if (last):
        result += '└─--'
        indent += '   '
    else:
        result += '|--'
        indent += '|  '
    result += '[{0}-{1} ({2})]'.format(name, params, is_complete)
    children = flatten(task.requires())
    for index, child in enumerate(children):
        result += print_tree(child, indent, (index + 1) == len(children))
    return result


def pop_cmndline_arg(key, flag=False, default=None):
    if flag:
        del sys.argv[sys.argv.index(key)]
        return
    else:
        arg_idx = sys.argv.index(key)
        try:
            del sys.argv[arg_idx]
            val = sys.argv.pop(arg_idx)
        except IndexError as e:
            if default is not None:
                val = default
            else:
                raise e
        return val


def get_downstream_tasks_recur(task, target_query=None, query_type='family'):
    target_tasks = OrderedDict()

    if query_type == 'family':
        is_target_task = task.task_family == target_query
    elif query_type == 'id':
        is_target_task = task.task_id == target_query
    elif query_type == 'any':
        assert target_query is None
        is_target_task = True
    else:
        raise ValueError(query_type)

    if is_target_task:
        target_tasks.update({task.task_id: task})

    required_tasks_tmp = task.requires()
    if required_tasks_tmp is None:
        return target_tasks

    else:
        if isinstance(required_tasks_tmp, (list, tuple, dict)):
            if isinstance(required_tasks_tmp, dict):
                required_tasks_tmp = list(required_tasks_tmp.values())
            required_tasks = normalize_list_recursively(required_tasks_tmp)
        else:
            required_tasks = [required_tasks_tmp]

        for required_task in required_tasks:
            target_tasks_child = get_downstream_tasks_recur(required_task, target_query, query_type)
            if target_tasks_child:
                target_tasks.update(target_tasks_child)

        if target_tasks:
            target_tasks.update({task.task_id: task})
            return target_tasks
        else:
            return target_tasks


class PyTorchPickleFileProcessor(FileProcessor):
    def format(self):
        return luigi.format.Nop

    def load(self, file):
        return torch.load(file)

    def dump(self, obj, file):
        torch.save(obj, file)


def print_scp_progress(filename, size, sent):
    print(f'filename: {filename}, size: {size}, sent: {sent}')


def pull_required_output_via_scp(task):
    targets = task.input()
    if not isinstance(targets, (list, tuple)):
        targets = [targets]

    for target in targets:
        if not target.exists():
            pull_output_via_scp(target)


def pull_output_via_scp(target):
    path = target.path()
    with SSHClient() as ssh:
        ssh.set_missing_host_key_policy(WarningPolicy())
        ssh.load_system_host_keys()
        key_filename = os.getenv('SCP_KEY_FILENAME')
        if key_filename:
            key = RSAKey(filename=key_filename)
            ssh.connect(hostname=os.getenv('SCP_SERVER_HOST'),
                        port=int(os.getenv('SCP_SERVER_PORT')),
                        username=os.getenv('SCP_SERVER_USER'),
                        pkey=key)
        else:
            ssh.connect(hostname=os.getenv('SCP_SERVER_HOST'),
                        port=int(os.getenv('SCP_SERVER_PORT')),
                        username=os.getenv('SCP_SERVER_USER'),
                        password=os.getenv('SCP_SERVER_PASS'))

        scp = SCPClient(ssh.get_transport(), progress=print_scp_progress)
        scp.get(os.path.join(os.getenv('SCP_SERVER_WORKDIR'), path), os.path.dirname(path))


def normalize_list_recursively(list_):
    '''
    get n-level nested list and returns un-nested list
    Args:
        list_:

    Returns:

    '''
    items = []
    for item in list_:
        if isinstance(item, Iterable):
            items.extend(normalize_list_recursively(item))
        else:
            items.append(item)
    return items


def build(task: TaskOnKart, return_value: bool = True, reset_register: bool = True, log_level: int = logging.ERROR, **env_params) -> Optional[Any]:
    """
    Run gokart task for local interpreter.
    Sharing the most of its parameters with luigi.build (see https://luigi.readthedocs.io/en/stable/api/luigi.html?highlight=build#luigi.build)
    """
    if reset_register:
        _reset_register()
    with LoggerConfig(level=log_level):
        result = luigi.build([task], detailed_summary=True, log_level=logging.getLevelName(log_level), **env_params)
        if result.status == luigi.LuigiStatusCode.FAILED:
            raise GokartBuildError(result.summary_text)
    return _get_output(task) if return_value else None
