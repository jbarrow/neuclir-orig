settings = {
    'base_dir': 'data/sw',
    'scores': [
        'matchers/SW_Matcher_Q2Q3_All/umdNMT_stemmed',
        'matchers/SW_Matcher_Q1_All/umdNMT_stemmed'
    ],
    'judgements': [
        'annotations/dev',
        'annotations/analysis1',
        'annotations/analysis2'
    ],
    'queries': [
        'queries/query1',
        'queries/query2',
        'queries/query3',
    ],
    'docs': [
        'segmentations/sw/ANALYSIS1',
        'segmentations/sw/ANALYSIS2',
        'segmentations/sw/DEV',
        'segmentations/sw/EVAL1',
        'segmentations/sw/EVAL2',
        'segmentations/sw/EVAL3',
    ],
    'audio': [
        'docs/analysis1/audio',
        'docs/analysis2/audio',
        'docs/dev/audio',
        'docs/eval1/audio',
        'docs/eval2/audio',
        'docs/eval3/audio'
    ],
    'strategy': 'random',
    'logging': 'info',
    'n_irrelevant': 1,
    'splits': {
        'training': 0.7,
        'testing': 0.3
    },
    'output': 'data/sw'
}


# TODO: generate a csv file from the tlsw translations
# TODO: generate a csv with the query and all the relevant docs for each query
# TODO: write two wrappers for the same model: a training wrapper and an evaluation wrapper
# TODO: write a script that calls the AQWV evaluation script (in the mean time compute MRR)
# TODO: build out a basic model that relies on the dataset reader
