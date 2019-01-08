# Variables
local use_scores = true;
local embedding_dims = 50;

local random_seed = 2019;
local pytorch_seed = random_seed * 10;
local numpy_seed = pytorch_seed * 10;

local dataset = 'datasets/normalized/';

# Helper Functions
local Embedder(path, dim, trainable=false) = {
  tokens: {
    type: 'embedding',
    pretrained_file: path,
    embedding_dim: dim,
    trainable: trainable
  }
};

local EmbeddingTransformer(dim, dropout=0.5, activation='relu') = {
  input_dim: dim,
  num_layers: 1,
  hidden_dims: [dim],
  activations: [activation],
  dropout: [dropout]
};

local Scorer(embedding_dim, lexical_input=false) = {
  local lexical_dims = if lexical_input then 1 else 0,
  input_dim: embedding_dim * 2,# + lexical_dims,
  num_layers: 1,
  hidden_dims: [1],
  activations: ['linear'],
  dropout: [0.0]
};

local Pathify(relative_path) = '/storage3/proj/joe/neuclir/' + relative_path;

{
  random_seed: random_seed, pytorch_seed: pytorch_seed, numpy_seed: numpy_seed,
  dataset_reader: { type: 'paired_dataset_reader', scores: use_scores, lazy: false },
  #validation_dataset_reader: { type: 'paired_dataset_reader', scores: use_scores, lazy: false },
  validation_dataset_reader: { type: 'reranking_dataset_reader', scores: use_scores, lazy: true },
  #evaluate_on_test: true,
  train_data_path: Pathify(dataset + 'train.json'),
  #validation_data_path: Pathify('datasets/random/validation_paired.json'),
  validation_data_path: Pathify(dataset + 'validation.json'),
  model: {
    type: 'letor_training',
    aqwv_corrections: Pathify(dataset + 'validation_scoring.json'),

//    doc_field_embedder: Embedder(Pathify('data/embeddings/cca_soen.txt'), embedding_dims),
    doc_field_embedder: Embedder(Pathify('data/embeddings/glove.6B.50d.txt'), embedding_dims),
    doc_transformer: EmbeddingTransformer(embedding_dims),
//    doc_encoder:

//    query_field_embedder: Embedder(Pathify('data/embeddings/cca.en.txt'), embedding_dims),
    query_field_embedder: Embedder(Pathify('data/embeddings/glove.6B.50d.txt'), embedding_dims),
    query_transformer: EmbeddingTransformer(embedding_dims),
//    query_encoder:

    scorer: Scorer(embedding_dims, use_scores),
    total_scorer: {
      input_dim: 2,
      num_layers: 1,
      hidden_dims: [1],
      activations: ['linear'],
      dropout: [0.0]
    }
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['docs', 'list_num_tokens']],
    batch_size: 8
  },
  validation_iterator: {
    type: 'bucket',
    sorting_keys: [['docs', 'num_fields']],
    batch_size: 2
  },
  trainer: {
    num_epochs: 40,
    patience: 10,
    cuda_device: 0,
    grad_clipping: 5.0,
    validation_metric: '+aqwv_2',
    optimizer: {
      type: 'adam',
      lr: 0.001
    }
  }
}