# ------------------------------------------------------------------------
# Some classes or methods are made by modifying parts of Luigi (https://github.com/spotify/luigi), Copyright 2012-2019 Spotify AB.
# The portions of the following codes are licensed under the Apache License 2.0.
# The full license text is available at (https://github.com/spotify/luigi/blob/master/LICENSE).
# ------------------------------------------------------------------------

import json
import bz2

import luigi
import gokart
#
# class TaskInstanceParameter(gokart.TaskInstanceParameter):
#     def serialize(self, x):
#         params = bz2.compress(json.dumps(x.to_str_params(only_significant=True)).encode()).hex()[:16]
#         values = dict(type=x.get_task_family(), params=params)
#         return luigi.DictParameter().serialize(values)
#
#
# class _TaskInstanceEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, luigi.Task):
#             return TaskInstanceParameter().serialize(obj)
#         # Let the base class default method raise the TypeError
#         return json.JSONEncoder.default(self, obj)
#
# class ListTaskInstanceParameter(luigi.Parameter):
#     def serialize(self, x):
#         return json.dumps(x, cls=_TaskInstanceEncoder)


class MyListParameter(luigi.ListParameter):
    @classmethod
    def parse(cls, x):
        # for avoiding many escapes \ for string list on shell and adding []
        # assumes x is list of str without " and [] or list of number
        if isinstance(x, list):
            return x
        elif all([not xx.isdigit() for xx in x]):
            x = '["' + x.replace(',', '","') + '"]'
        else:
            x = '[' + x + ']'
        return super().parse(cls, x)
