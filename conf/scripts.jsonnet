// Variables we care about
local language = 'so';
local collections = ['DEV'];
local queries = ['QUERY1'];
local tags = 'PSQ';

// Helper variables
local base_settings = {
  data_structure_store: "data_store_structure.20190107_033748.txt",
  query_analyzer: "v10.3",
  asr: "v5.0",
  sentence_splitter: "v3.0",
  text_language_id: "v7.0",
  audio_language_id: "v1.0",
  morphology: "v4.1",
  edi_nmt: "v6.1",
  umd_nmt: "v4.1",
  umd_smt: "v2.4",
  post_edit: "2.0",
  domain_id: "4.0",
  stemmer: "1.0",
  indexer: "v4.2",
  matcher: "v11.1"
};

local settings = {
  sw: base_settings + { index: '1A' },
  tl: base_settings + { index: '1B' },
  so: base_settings + {
    index: '1S',
    asr: "v6.0",
    domain_id: "v6.0",
    data_structrue_store: "data_store_structure.20190107_031923.txt"
  }
}[language];


local pathify(paths) = std.join('/', paths);

local collection_paths = [pathify([settings['index'], 'IARPA_MATERIAL_BASE-'+settings['index'], collection]) for collection in collections];
local query_paths = [pathify(["/storage2", "data", "NIST-data", settings['index'], "IARPA_MATERIAL_BASE-" + settings['index'], "query_store", "query-analyzer-umd-" + settings['query_analyzer'], q]) + "/" for q in queries];

local asr_store="material-asr-"+language+"-"+settings['asr'];
local sent_split="sent-split-"+settings['sentence_splitter'];

// Helper functions
local MatcherConfiguration(name, type, index_type, params, mt="", indexes=[], format='tsv', cutoff=-1) = {
  local version="indexing-umd-"+settings['indexer'],
  local no_mt_systems = {"PSQ": false, "DBQT": false, "customPSQ": false},
  local use_mt = if std.objectHas(no_mt_systems, name) then false else true,

  local indexes =
    std.flattenArrays([
      if use_mt then
        [ pathify([collection, "text", "index_store", version, params, "text", "mt_store", mt+"_"+sent_split, "best", "indri_N"]) + "/",
          pathify([collection, "audio", "index_store", version, params, "audio", "mt_store", mt+"_"+asr_store, "best", "indri_N"]) + "/" ]
      else
        [ pathify([collection, "text", "index_store", version, params, "text", "src", "indri_N"]) + "/",
          pathify([collection, "audio", "index_store", version, params, "audio", "asr_store", asr_store, "indri_N"]) + "/" ]
      for collection in collection_paths]
    ),

  config_name: name,
  type: type,
  index_type: index_type,
  format: format,
  cutoff: cutoff,
  indexes: indexes
};

local QueryProcessor(query_paths) = {
  version: "query-analyzer-umd:"+settings['query_analyzer'],
  query_list_path: query_paths,
  target_language: language
};

// Central JSON structure

{
  submission_type: "contrastive",
  data_collection: {
    data_store_structure: [
       "data_structrue_store/" + settings['data_structure_store']
    ],
    collections: collection_paths
  },
  query_processor: QueryProcessor(query_paths),
  matcher: {
   version: "matching-umd:"+settings['matcher'],
    configurations: [
      MatcherConfiguration("DBQT", "indri", "DBQT", "indexing_params_v1.0"),
      MatcherConfiguration("PSQ", "indri", "UMDPSQPhraseBasedGVCCCutoff097", "indexing_params_v1.0"),
      MatcherConfiguration("UMDSMT", "indri", "words", "indexing_params_v1.0", mt="umd-smt-"+settings['umd_smt']),
      MatcherConfiguration("SMTsrc", "indri", "phrases-simple_part_flat_conjunction_075", "indexing_params_porter_stemmer_v2.0", mt="umd-smt-"+settings['umd_smt']),
      MatcherConfiguration("SMTsrp", "indri", "phrases-complex", "indexing_params_porter_stemmer_v2.0", mt="umd-smt-"+settings['umd_smt']),
      MatcherConfiguration("UMDNMT", "indri", "words", "indexing_params_v1.0", mt="umd-nmt-"+settings['umd_nmt']),
      MatcherConfiguration("UMDNMTsrf", "indri", "words_part_flat", "indexing_params_porter_stemmer_v2.0", mt="umd-nmt-"+settings['umd_nmt']),
      MatcherConfiguration("UMDNMTsrc", "indri", "words_conjunction_075", "indexing_params_porter_stemmer_v2.0", mt="umd-nmt-"+settings['umd_nmt']),
      MatcherConfiguration("EdiNMTsr", "indri", "words", "indexing_params_porter_stemmer_v2.0", mt="scriptsmt-systems-v6.1"),
      MatcherConfiguration("EdiNMTsrf", "indri", "words_part_flat", "indexing_params_porter_stemmer_v2.0", mt="scriptsmt-systems-v6.1")
    ] + if language == 'so' then [MatcherConfiguration("customPSQ", "bbn_text", "UMDPSQPhraseBasedGVCCCutoff097_part_flat", "indexing_params_v1.0")] else []
  },
  evidence_combination: {
    version: "evidence-combination:v11.2",
    cutoff_type: "fixed",
    cutoff: 2,
    score_type: "borda",
    filtering: ""
  },
  evaluator: {
    version: "evaluation:v8.0",
    relevance_judgments:"/storage2/data/NIST-data/relevance_judgments",
    mode: "light",
    beta: "40"
  },
  description: {
    tags: tags
  }
}
