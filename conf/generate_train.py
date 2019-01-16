import _jsonnet
import random
import glob
import json
import sys
import os


# dan: bool,
# averaged: bool,
# num_filters: int,
# dropout: float,
# batch_size: int,
# clipping: float,
# lr: float,
# l2: float,
# dataset: str

learning_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
l2s = [0., 0.0001, 0.001, 0.01, 0.1]
dropouts = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
averageds = [True, False]
num_filters = [5, 10, 15, 20, 25, 30, 35, 40]
batch_sizes = [8, 12, 16]
clippings = [0.25, 1., 5.]

jsonnet_str = open('conf/train.jsonnet').read()

for dataset in glob.glob('datasets/*/'):
    for dan in [True, False]:
        learning_rate = random.sample(learning_rates, k=1)[0]
        l2 = random.sample(l2s, k=1)[0]
        dropout = random.sample(dropouts, k=1)[0]
        averaged = random.sample(averageds, k=1)[0]
        num_filter = random.sample(num_filters, k=1)[0]
        batch_size = random.sample(batch_sizes, k=1)[0]
        clipping = random.sample(clippings, k=1)[0]

        d = os.path.basename(os.path.normpath(dataset))

        with open(f'runs/{d}_dan-{dan}.json', 'w') as fp:
            json_str = _jsonnet.evaluate_snippet(
                "conf/train.jsonnet", jsonnet_str,
                ext_codes={
                    'dan': str(dan).lower(),
                    'averaged': str(averaged).lower(),
                    'num_filters': str(num_filter),
                    'dropout': str(dropout),
                    'batch_size': str(batch_size),
                    'clipping': str(clipping),
                    'lr': str(learning_rate),
                    'l2': str(l2),
                },
                ext_vars={
                    'dataset': dataset
                })
            fp.write(json_str)
