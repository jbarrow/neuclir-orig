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
# lang: str
# ranking_loss: bool
# use_attention: bool
# use_batch_norm: bool
# query_embeddings: str
# doc_embeddings: str

learning_rates = [0.001, 0.003, 0.01]
l2s = [0., 0.0001, 0.001, 0.01]
dropouts = [0., 0.1, 0.2, 0.3]
averageds = [True, False]
num_filters = [20, 25, 30, 35, 40]
batch_sizes = [8, 12]
clippings = [0.25, 1., 5.],
use_idfs = [True, False]
use_attentions = [True, False]
use_batch_norms = [True, False]
ranking_losses = [True, False]
dans = [True, False]

jsonnet_str = open('conf/train.jsonnet').read()

for dataset in glob.glob('datasets/*_irr/'):
    for eval_lang in ['tl', 'so']:
        for i in range(10):
            dan = random.sample(dans, k=1)[0]
            use_idf = random.sample(use_idfs, k=1)[0]
            use_attention = random.sample(use_attentions, k=1)[0]
            use_batch_norm = random.sample(use_batch_norms, k=1)[0]
            ranking_loss = random.sample(ranking_losses, k=1)[0]
            learning_rate = random.sample(learning_rates, k=1)[0]
            l2 = random.sample(l2s, k=1)[0]
            dropout = random.sample(dropouts, k=1)[0]
            averaged = random.sample(averageds, k=1)[0]
            num_filter = random.sample(num_filters, k=1)[0]
            batch_size = random.sample(batch_sizes, k=1)[0]
            clipping = random.sample(clippings, k=1)[0]
            query_averaged = random.sample(averageds, k=1)[0]

            d = os.path.basename(os.path.normpath(dataset))

            with open(f'runs/{d}_run-{i}-{eval_lang}.json', 'w') as fp:
                json_str = _jsonnet.evaluate_snippet(
                    "conf/train.jsonnet", jsonnet_str,
                    ext_codes={
                        'dan': str(dan).lower(),
                        'averaged': str(averaged).lower(),
                        'num_filters': str(num_filter),
                        'dropout': str(dropout),
                        'batch_size': str(batch_size),
                        'clipping': '5.0',
                        'lr': str(learning_rate),
                        'l2': str(l2),
                        'use_batch_norm': str(use_batch_norm).lower(),
                        'use_attention': str(use_attention).lower(),
                        'use_idfs': str(use_idf).lower(),
                        'doc_projection': str(False).lower(),
                        'ranking_loss': str(ranking_loss).lower(),
                        'query_averaged': str(query_averaged).lower()
                    },
                    ext_vars={
                        'dataset': dataset,
                        'lang': eval_lang,
                        'idf_weights': 'idf_weights/' + eval_lang + '_idf.txt',
                        'query_embeddings': "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
                        'doc_embeddings': "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt"
                    })
                fp.write(json_str)

for dataset in glob.glob('datasets/*_irr/'):
    for i in range(10):
        eval_lang = 'tl'
        dan = random.sample(dans, k=1)[0]
        use_idf = random.sample(use_idfs, k=1)[0]
        use_attention = random.sample(use_attentions, k=1)[0]
        use_batch_norm = random.sample(use_batch_norms, k=1)[0]
        ranking_loss = random.sample(ranking_losses, k=1)[0]
        learning_rate = random.sample(learning_rates, k=1)[0]
        l2 = random.sample(l2s, k=1)[0]
        dropout = random.sample(dropouts, k=1)[0]
        averaged = random.sample(averageds, k=1)[0]
        num_filter = random.sample(num_filters, k=1)[0]
        batch_size = random.sample(batch_sizes, k=1)[0]
        clipping = random.sample(clippings, k=1)[0]
        query_averaged = random.sample(averageds, k=1)[0]

        d = os.path.basename(os.path.normpath(dataset))

        with open(f'runs/{d}_run-{i}.json', 'w') as fp:
            json_str = _jsonnet.evaluate_snippet(
                "conf/train.jsonnet", jsonnet_str,
                ext_codes={
                    'dan': str(dan).lower(),
                    'averaged': str(averaged).lower(),
                    'num_filters': str(num_filter),
                    'dropout': str(dropout),
                    'batch_size': str(batch_size),
                    'clipping': '5.0',
                    'lr': str(learning_rate),
                    'l2': str(l2),
                    'use_batch_norm': str(use_batch_norm).lower(),
                    'use_attention': str(use_attention).lower(),
                    'use_idfs': str(use_idf).lower(),
                    'doc_projection': str(True).lower(),
                    'ranking_loss': str(ranking_loss).lower(),
                    'query_averaged': str(query_averaged).lower()
                },
                ext_vars={
                    'dataset': dataset,
                    'lang': eval_lang,
                    'idf_weights': 'idf_weights/' + eval_lang + '_idf.txt',
                    'query_embeddings': "/fs/clip-scratch/jdbarrow/neuclir/neuclir/data/cca.en.txt",
                    'doc_embeddings': "/fs/clip-scratch/jdbarrow/neuclir/neuclir/data/cca.tl.txt"
                })
            fp.write(json_str)

for dataset in glob.glob('datasets/*_irr_so/'):
    for i in range(10):
        eval_lang = 'so'
        dan = random.sample(dans, k=1)[0]
        use_idf = random.sample(use_idfs, k=1)[0]
        use_attention = random.sample(use_attentions, k=1)[0]
        use_batch_norm = random.sample(use_batch_norms, k=1)[0]
        ranking_loss = random.sample(ranking_losses, k=1)[0]
        learning_rate = random.sample(learning_rates, k=1)[0]
        l2 = random.sample(l2s, k=1)[0]
        dropout = random.sample(dropouts, k=1)[0]
        averaged = random.sample(averageds, k=1)[0]
        num_filter = random.sample(num_filters, k=1)[0]
        batch_size = random.sample(batch_sizes, k=1)[0]
        clipping = random.sample(clippings, k=1)[0]
        query_averaged = random.sample(averageds, k=1)[0]

        d = os.path.basename(os.path.normpath(dataset))

        with open(f'runs/{d}_run-{i}.json', 'w') as fp:
            json_str = _jsonnet.evaluate_snippet(
                "conf/train.jsonnet", jsonnet_str,
                ext_codes={
                    'dan': str(dan).lower(),
                    'averaged': str(averaged).lower(),
                    'num_filters': str(num_filter),
                    'dropout': str(dropout),
                    'batch_size': str(batch_size),
                    'clipping': '5.0',
                    'lr': str(learning_rate),
                    'l2': str(l2),
                    'use_batch_norm': str(use_batch_norm).lower(),
                    'use_attention': str(use_attention).lower(),
                    'use_idfs': str(use_idf).lower(),
                    'doc_projection': str(True).lower(),
                    'ranking_loss': str(ranking_loss).lower(),
                    'query_averaged': str(query_averaged).lower()
                },
                ext_vars={
                    'dataset': dataset,
                    'lang': eval_lang,
                    'idf_weights': 'idf_weights/' + eval_lang + '_idf.txt',
                    'query_embeddings': "/fs/clip-scratch/jdbarrow/neuclir/neuclir/data/cca.en.txt",
                    'doc_embeddings': "/fs/clip-scratch/jdbarrow/neuclir/neuclir/data/cca.so.txt"
                })
            fp.write(json_str)
