# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

VISDOM_CONF_PATH = '/home/TUE/s167139/Thesis/MNTDP/resources/visdom.yaml'
MONGO_CONF_PATH = '/home/TUE/s167139/Thesis/MNTDP/resources/mongo.yaml'
LOCAL_SAVE_PATH = '/home/TUE/s167139/local/veniat/lileb/runs'


def load_conf(path):
    _, ext = os.path.splitext(path)
    with open(path) as file:
        if ext == '.json':
            import json
            conf = json.load(file)
        elif ext in ['.yaml', '.yml']:
            import yaml
            conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf
