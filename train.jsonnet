local Embedder(path, dim, trainable=false) = {
  tokens: {
    type: 'embedding',
    pretrained_file: path,
    embedding_dim: dim,
    trainable: trainable
  }
};

local Scorer(embedding_dim, lexical_input=false) = {
  local lexical_dims = if lexical_input then 1 else 0,
  input_dim: embedding_dim * 2 + lexical_dims,
  num_layers: 2,
  hidden_dims: [40, 1],
  activations: ['relu', 'linear'],
  dropout: [0.2, 0.0]
};

local Pathify(relative_path) = '/storage/proj/joe/neuclir/' + relative_path;

{
  dataset_reader: { type: 'letor_train_reader', scores: false },
  validation_dataset_reader: { type: 'letor_train_reader', scores: false },
  train_data_path: Pathify('data/sw/train.csv'),
  validation_data_path: Pathify('data/sw/test.csv'),
  model: {
    type: 'letor_training',
    doc_field_embedder: Embedder(Pathify('data/cca.sw.txt'), 40),
    query_field_embedder: Embedder(Pathify('data/cca.en.txt'), 40),
    embedding_transformer: {
      input_dim: 40,
      num_layers: 1,
      hidden_dims: [40],
      activations: ['relu'],
      dropout: [0.2]
    },
    scorer: Scorer(40, false)
  },
  iterator: {
    type: 'basic',
    #sorting_keys: [['docs', 'num_tokens']],
    batch_size: 64
  },
  trainer: {
    num_epochs: 40,
    patience: 10,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: '+accuracy',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
}
