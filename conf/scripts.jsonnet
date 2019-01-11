// Variables we care about
local language = 'tl';
local experiment_name = 'test';

local common = {
  collections: ['EVAL1', 'EVAL2', 'EVAL3'],
  queries: ['QUERY1'],
  tags: 'PSQ'
};

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


// UPDATE EXPERIMENTAL SETTINGS
local bm = { mt: "", cutoff: -1, format: "tsv", indexing_params: "indexing_params_v1" };
local matchers = [
  bm + { name: "DBQT", type: "indri", index_type: "DBQT" },
  bm + { name: "PSQ", type: "indri", index_type: "UMDPSQPhraseBasedGVCCCutoff097"},
  bm + {name: "UMDSMT", type: "indri", index_type: "words", mt: "umd-smt-"+settings['umd_smt']},
  bm + {name: "SMTsrc", type: "indri", index_type: "phrases-simple_part_flat_conjunction_075", indexing_params: "indexing_params_porter_stemmer_v2.0", mt: "umd-smt-"+settings['umd_smt']},
  bm + {name: "SMTsrp", type: "indri", index_type: "phrases-complex", indexing_params: "indexing_params_porter_stemmer_v2.0", mt: "umd-smt-"+settings['umd_smt'] },
  bm + {name: "UMDNMT", type: "indri", index_type: "words",  mt: "umd-nmt-"+settings['umd_nmt'] },
  bm + {name: "UMDNMTsrf", type: "indri", index_type: "words_part_flat", indexing_params: "indexing_params_porter_stemmer_v2.0", mt: "umd-nmt-"+settings['umd_nmt'] },
  bm + {name: "UMDNMTsrc", type: "indri", index_type: "words_conjunction_075", indexing_params: "indexing_params_porter_stemmer_v2.0", mt: "umd-nmt-"+settings['umd_nmt'] },
  bm + {name: "EdiNMTsr", type: "indri", index_type: "words", indexing_params: "indexing_params_porter_stemmer_v2.0", mt: "scriptsmt-systems-v6.1" },
  bm + {name: "EdiNMTsrf", type: "indri", index_type: "words_part_flat", indexing_params: "indexing_params_porter_stemmer_v2.0", mt: "scriptsmt-systems-v6.1" },
  bm + {name: "customPSQ", type: "bbn_text", index_type: "UMDPSQPhraseBasedGVCCCutoff097_part_flat" }
];

local experiments = {
  [experiment_name + '_' + matcher['name']]: common + {
    matchers: [matcher],
  } for matcher in matchers
};


local pathify(paths) = std.join('/', paths);

local collection_paths(collections) = [pathify([settings['index'], 'IARPA_MATERIAL_BASE-'+settings['index'], collection]) for collection in collections];
local query_paths(queries) = [pathify(["/storage2", "data", "NIST-data", settings['index'], "IARPA_MATERIAL_BASE-" + settings['index'], "query_store", "query-analyzer-umd-" + settings['query_analyzer'], q]) + "/" for q in queries];

local asr_store="material-asr-"+language+"-"+settings['asr'];
local sent_split="sent-split-"+settings['sentence_splitter'];

// Helper functions
local MatcherConfiguration(collection_paths, name, type, index_type, params, mt="", indexes=[], format='tsv', cutoff=-1) = {
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

local generate_matchers(collections, configs) = [MatcherConfiguration(
    collections, name=matcher['name'],
    type=matcher['type'],
    index_type=matcher['index_type'],
    params=matcher['indexing_params'],
    mt=matcher['mt'],
    format=matcher['format'],
    cutoff=matcher['cutoff']) for matcher in configs];

local QueryProcessor(query_paths) = {
  version: "query-analyzer-umd:"+settings['query_analyzer'],
  query_list_path: query_paths,
  target_language: language
};

// Central JSON structure

{
  [key + '.json']: {
    local collections = collection_paths(experiments[key]['collections']),
    submission_type: "contrastive",
    data_collection: {
      data_store_structure: [
         "data_structrue_store/" + settings['data_structure_store']
      ],
      collections: collections
    },
    query_processor: QueryProcessor(query_paths(experiments[key]['queries'])),
    matcher: {
     version: "matching-umd:"+settings['matcher'],
     configurations: generate_matchers(collections, experiments[key]['matchers'])
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
      tags: experiments[key]['tags']
    }
  } for key in std.objectFields(experiments)
}


/* configurations: [
  MatcherConfiguration(collections, name="DBQT", type="indri", index_type="DBQT", params="indexing_params_v1.0"),
  MatcherConfiguration(collections, "PSQ", "indri", "UMDPSQPhraseBasedGVCCCutoff097", "indexing_params_v1.0"),
  MatcherConfiguration(collections, "UMDSMT", "indri", "words", "indexing_params_v1.0", mt="umd-smt-"+settings['umd_smt']),
  MatcherConfiguration(collections, "SMTsrc", "indri", "phrases-simple_part_flat_conjunction_075", "indexing_params_porter_stemmer_v2.0", mt="umd-smt-"+settings['umd_smt']),
  MatcherConfiguration(collections, "SMTsrp", "indri", "phrases-complex", "indexing_params_porter_stemmer_v2.0", mt="umd-smt-"+settings['umd_smt']),
  MatcherConfiguration(collections, "UMDNMT", "indri", "words", "indexing_params_v1.0", mt="umd-nmt-"+settings['umd_nmt']),
  MatcherConfiguration(collections, "UMDNMTsrf", "indri", "words_part_flat", "indexing_params_porter_stemmer_v2.0", mt="umd-nmt-"+settings['umd_nmt']),
  MatcherConfiguration(collections, "UMDNMTsrc", "indri", "words_conjunction_075", "indexing_params_porter_stemmer_v2.0", mt="umd-nmt-"+settings['umd_nmt']),
  MatcherConfiguration(collections, "EdiNMTsr", "indri", "words", "indexing_params_porter_stemmer_v2.0", mt="scriptsmt-systems-v6.1"),
  MatcherConfiguration(collections, "EdiNMTsrf", "indri", "words_part_flat", "indexing_params_porter_stemmer_v2.0", mt="scriptsmt-systems-v6.1")
] + if language == 'so' then [MatcherConfiguration("customPSQ", "bbn_text", "UMDPSQPhraseBasedGVCCCutoff097_part_flat", "indexing_params_v1.0")] else [] */
