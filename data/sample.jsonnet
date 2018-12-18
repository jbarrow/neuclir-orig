{
  training: {
#    scores: [],
    judgements: [
      'annotations/eval1', 'annotations/eval2', 'annotations/eval3', 'annotations/analysis1', 'annotations/analysis2'
    ],
    queries: [
      'queries/query1', 'queries/query2', 'queries/query3'
    ],
    docs: [
      'segmentations/sw/ANALYSIS1', 'segmentations/sw/ANALYSIS2', 'segmentations/sw/DEV', 'segmentations/sw/EVAL1', 'segmentations/sw/EVAL2', 'segmentations/sw/EVAL3'
    ],
    audio: [
      'docs/analysis1/audio'
    ],
  },
  validation: {
#    scores: [],
    judgements: [
      'annotations/dev'
    ],
    queries: [],
    docs: [],
    audio: [],
  },
  test: {
#    scores: [],
    judgements: [],
    queries: [],
    docs: [],
    audio: [],
  },
  strategy: 'random',
  logging: 'info',
  n_irrelevant: 1,
  output: 'data/sw'
}



    /* 'base_dir': 'data/sw',
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
} */
