import copy
import torch
import time
import os
import random
import numpy as np

import aux_funcs  as af
import network_architectures as arcs

from architectures.CNNs.VGG import VGG

from train_networks import *

random_seed = af.get_random_seed()
af.set_random_seeds()
print('Random Seed: {}'.format(random_seed))
device = af.get_pytorch_device()
models_path = 'networks/{}'.format(af.get_random_seed())
af.create_path(models_path)
af.set_logger('outputs/train_models'.format(af.get_random_seed()))

tasks = ['cifar10']
cnns = []
sdns = []

for task in tasks:
        af.extend_lists(cnns, sdns, arcs.create_resnet56_rpf(models_path, task, save_type='cd'))

train(models_path, cnns, sdn=False, device=device)
