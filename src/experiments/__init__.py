# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from src.experiments.stream_tuning import StreamTuningExperiment


def get_experiment_by_name(name):
    if name == 'stream_tuning':
        return StreamTuningExperiment
    raise NotImplementedError(name)


def init_experiment(_name, _config_from_id=None, **kwargs):
    return get_experiment_by_name(_name)(**kwargs)
