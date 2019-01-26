# options
local output = "datasets";
local filter_stopwords = true;
local tag = "zero_one_rand_1_irr";
# base path
local base = {
  sw: '/storage2/data/NIST-data/1A/IARPA_MATERIAL_BASE-1A',
  tl: '/storage2/data/NIST-data/1B/IARPA_MATERIAL_BASE-1B',
  so: '/storage2/data/NIST-data/1S/IARPA_MATERIAL_BASE-1S'
};
# score base path
local score_base = '/storage2/proj/han/EM/new_matchers_output/new_matcher_run/JoeRun2/new_run';
# local so_score_base = '/storage2/proj/petra/config/Experiments-January/runs-selection-Monday'
# functions to generate the necessary paths
local pathify(paths) = std.join('/', paths);
local n_irr = 1;

//
local sw_q1_da = "sw/sw_dev_analysis_q1/query-analyzer-umd-v10.2_matching-umd-v11.1_evidence-combination-v11.2/SW_QUERY1_DEV_ANALYSIS1_ANALYSIS2_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2";
local sw_q2q3_da = "sw/sw_dev_analysis_q2q3/query-analyzer-umd-v10.2_matching-umd-v11.1_evidence-combination-v11.2/SW_QUERY2QUERY3_DEV_ANALYSIS1_ANALYSIS2_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2";
local sw_q1_eval = "sw/sw_eval_q1/query-analyzer-umd-v10.2_matching-umd-v11.1_evidence-combination-v11.2/SW_QUERY1_EVAL1_EVAL2_EVAL3_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2";
local sw_q2q3_eval = "sw/sw_eval_q2q3/query-analyzer-umd-v10.2_matching-umd-v11.1_evidence-combination-v11.2/SW_QUERY2QUERY3_EVAL1_EVAL2_EVAL3_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2";

local so_q1_dev = "so/so_dev_q1/query-analyzer-umd-v10.3_matching-umd-v11.1_evidence-combination-v12.0/SO_QUERY1_DEV_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_customPSQ_Cutoff2";
local so_q1_ana = "so/so_analysis_q1/query-analyzer-umd-v10.3_matching-umd-v11.1_evidence-combination-v12.0/SO_QUERY1_ANALYSIS1_ANALYSIS2_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2";

local tl_q1_da = "tl/tl_dev_analysis_q1/query-analyzer-umd-v10.3_matching-umd-v11.1_evidence-combination-v12.0/TL_QUERY1_DEV_ANALYSIS1_ANALYSIS2_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2";
local tl_q2q3_da = "tl/tl_dev_analysis_q2q3/query-analyzer-umd-v10.3_matching-umd-v11.1_evidence-combination-v12.0/TL_QUERY2QUERY3_DEV_ANALYSIS1_ANALYSIS2_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2";

// system output directories
local matchers = {
  # system group
  dbqt: "UMD-CLIR-workQMDir-DBQT/",
  # system group
  #smt: "UMD-CLIR-workQMDir-UMDSMT/",
  smt_src: "UMD-CLIR-workQMDir-SMTsrc/",
  #smt_srp: "UMD-CLIR-workQMDir-SMTsrp/",
  # system group
  #umd_nmt: "UMD-CLIR-workQMDir-UMDNMT/",
  #umd_nmt_srf: "UMD-CLIR-workQMDir-UMDNMTsrf/",
  #umd_nmt_src: "UMD-CLIR-workQMDir-UMDNMTsrc/",
  # system group
  #edi_nmt_sr: "UMD-CLIR-workQMDir-EdiNMTsr/",
  edi_nmt_srf: "UMD-CLIR-workQMDir-EdiNMTsrf/",
  # system group
  psq: "UMD-CLIR-workQMDir-PSQ/"
};

{
  [matcher + '_' + tag + '.json']: {
    datasets: {
      train: {
        type: "paired",
        n_irrelevant: n_irr,
        strategy: "random",/*
        sample_system: matcher,
        p_difficult: 0.8,*/
        filter_stopwords: filter_stopwords,
        scores: {
          [matcher]: [
            pathify([score_base, sw_q1_da, matchers[matcher]]),
            pathify([score_base, sw_q2q3_da, matchers[matcher]]),
            pathify([score_base, sw_q1_eval, matchers[matcher]]),
            pathify([score_base, sw_q2q3_eval, matchers[matcher]])
          ]
        },
        judgements: [
          pathify([base['sw'], p, 'query_annotation.tsv']) for p in
            ['DEV_ANNOTATION', 'DEV_ANNOTATION1', 'ANALYSIS_ANNOTATION1', 'ANALYSIS_ANNOTATION2', 'ANALYSIS_ANNOTATION3', 'EVAL_ANNOTATION']
        ],
        queries: [
          pathify([base['sw'], 'query_store/query-analyzer-umd-v10.3', p, '*']) for p in
            ['QUERY1', 'QUERY2', 'QUERY3']
        ],
        docs: [
          pathify([base['sw'], ds, 'text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']) for ds in
            ['DEV', 'ANALYSIS1', 'ANALYSIS2', 'EVAL1', 'EVAL2', 'EVAL3']
        ] + [
          pathify([base['sw'], ds, 'audio/mt_store/umd-smt-v2.4_material-asr-sw-v5.0', '*.txt']) for ds in
            ['DEV', 'ANALYSIS1', 'ANALYSIS2', 'EVAL1', 'EVAL2', 'EVAL3']
        ]
      },
      validation_so: {
        type: "reranking",
        filter_stopwords: filter_stopwords,
        scores: {
          [matcher]: [
            pathify([score_base, so_q1_dev, matchers[matcher]])
          ]
        },
        judgements: [
          pathify([base['so'], 'DEV_ANNOTATION2', 'query_annotation.tsv']),
          pathify([base['so'], 'DEV_ANNOTATION1', 'query_annotation.tsv'])
        ],
        queries:    [pathify([base['so'], 'query_store/query-analyzer-umd-v10.3/QUERY1', '*'])],
        docs: [
          pathify([base['so'], 'DEV/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['so'], 'DEV/audio/mt_store/umd-smt-v2.4_material-asr-so-v7.0/', '*.txt'])
        ]
      },
      test_so: {
        type: "reranking",
        filter_stopwords: filter_stopwords,
        scores: {
          [matcher]: [
            pathify([score_base, so_q1_ana, matchers[matcher]])
          ]# for matcher in std.objectFields(matchers)
        },
        judgements: [pathify([base['so'], 'ANALYSIS_ANNOTATION2', 'query_annotation.tsv'])],
        queries:    [
          pathify([base['so'], 'query_store/query-analyzer-umd-v10.3/QUERY1', '*'])
        ],
        docs: [
          pathify([base['so'], 'ANALYSIS1/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['so'], 'ANALYSIS1/audio/mt_store/umd-smt-v2.4_material-asr-so-v7.0/', '*.txt']),
          pathify([base['so'], 'ANALYSIS2/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['so'], 'ANALYSIS2/audio/mt_store/umd-smt-v2.4_material-asr-so-v7.0/', '*.txt'])
        ]
      },
      validation_tl: {
        type: "reranking",
        filter_stopwords: filter_stopwords,
        scores: {
          [matcher]: [
            pathify([score_base, tl_q1_da, matchers[matcher]])
          ]# for matcher in std.objectFields(matchers)
        },
        judgements: [
          pathify([base['tl'], 'DEV_ANNOTATION1', 'query_annotation.tsv']),
          pathify([base['tl'], 'ANALYSIS_ANNOTATION1', 'query_annotation.tsv']),
          pathify([base['tl'], 'ANALYSIS_ANNOTATION2', 'query_annotation.tsv']),
          pathify([base['tl'], 'ANALYSIS_ANNOTATION3', 'query_annotation.tsv'])
        ],
        queries:    [pathify([base['tl'], 'query_store/query-analyzer-umd-v10.3/QUERY1', '*'])],
        docs: [
          pathify([base['tl'], 'DEV/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['tl'], 'DEV/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/', '*.txt']),
          pathify([base['tl'], 'ANALYSIS1/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['tl'], 'ANALYSIS1/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/', '*.txt']),
          pathify([base['tl'], 'ANALYSIS2/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['tl'], 'ANALYSIS2/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/', '*.txt'])
        ]
      },
      test_tl: {
        type: "reranking",
        filter_stopwords: filter_stopwords,
        scores: {
          [matcher]: [
            pathify([score_base, tl_q2q3_da, matchers[matcher]])
          ]# for matcher in std.objectFields(matchers)
        },
        judgements: [
          pathify([base['tl'], 'DEV_ANNOTATION1', 'query_annotation.tsv']),
          pathify([base['tl'], 'ANALYSIS_ANNOTATION1', 'query_annotation.tsv']),
          pathify([base['tl'], 'ANALYSIS_ANNOTATION2', 'query_annotation.tsv']),
          pathify([base['tl'], 'ANALYSIS_ANNOTATION3', 'query_annotation.tsv'])
        ],
        queries:    [
          pathify([base['tl'], 'query_store/query-analyzer-umd-v10.3/QUERY2', '*']),
          pathify([base['tl'], 'query_store/query-analyzer-umd-v10.3/QUERY3', '*'])
        ],
        docs: [
          pathify([base['tl'], 'DEV/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['tl'], 'DEV/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/', '*.txt']),
          pathify([base['tl'], 'ANALYSIS1/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['tl'], 'ANALYSIS1/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/', '*.txt']),
          pathify([base['tl'], 'ANALYSIS2/text/mt_store/umd-smt-v2.4_sent-split-v3.0', '*.txt']),
          pathify([base['tl'], 'ANALYSIS2/audio/mt_store/umd-smt-v2.4_material-asr-tl-v5.0/', '*.txt'])
        ]
      }
    },
    #systems: std.objectFields(matchers),
    systems: [matcher],
    output: pathify([output, matcher + '_' + tag]),
    logging: "info",
    normalization: 'zero_one',
  } for matcher in std.objectFields(matchers)
}
