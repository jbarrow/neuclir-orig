# base path
local base = "/storage3/proj/joe/neuclir/data/so/";
local scrs = "/storage3/proj/joe/";
# functions to generate the necessary paths
local Pathify(relative_path, base) = base + relative_path;

//
local sw_q1_da = "sw/sw_dev_analysis_q1/query-analyzer-umd-v10.2_matching-umd-v11.1_evidence-combination-v11.2/SW_QUERY1_DEV_ANALYSIS1_ANALYSIS2_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2/";
local sw_q2q3_da = "sw/sw_dev_analysis_q2q3/query-analyzer-umd-v10.2_matching-umd-v11.1_evidence-combination-v11.2/SW_QUERY2QUERY3_DEV_ANALYSIS1_ANALYSIS2_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2/";
local sw_q1_eval = "sw/sw_eval_q1/query-analyzer-umd-v10.2_matching-umd-v11.1_evidence-combination-v11.2/SW_QUERY1_EVAL1_EVAL2_EVAL3_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2/";
local sw_q2q3_eval = "sw/sw_eval_q2q3/query-analyzer-umd-v10.2_matching-umd-v11.1_evidence-combination-v11.2/SW_QUERY2QUERY3_EVAL1_EVAL2_EVAL3_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2/";

local so_q1_dev = "so/so_dev_q1/query-analyzer-umd-v10.3_matching-umd-v11.1_evidence-combination-v12.0/SO_QUERY1_DEV_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_customPSQ_Cutoff2/";
local so_q2_ana = "so/so_analysis_q1/query-analyzer-umd-v10.3_matching-umd-v11.1_evidence-combination-v12.0/SO_QUERY1_ANALYSIS1_ANALYSIS2_DBQT_PSQ_UMDSMT_SMTsrc_SMTsrp_UMDNMT_UMDNMTsrf_UMDNMTsrc_EdiNMTsr_EdiNMTsrf_Cutoff2/";

// system output directories
local matchers = {
  /* # system group
  dbqt: "UMD-CLIR-workQMDir-DBQT/",
  # system group
  umd_smt: "UMD-CLIR-workQMDir-UMDSMT/",
  umd_smt_src: "UMD-CLIR-workQMDir-SMTsrc/", */
  umd_smt_srp: "UMD-CLIR-workQMDir-SMTsrp/",
  /* # system group
  umd_nmt: "UMD-CLIR-workQMDir-UMDNMT/",
  umd_nmt_srf: "UMD-CLIR-workQMDir-UMDNMTsrf/",
  umd_nmt_src: "UMD-CLIR-workQMDir-UMDNMTsrc/",
  # system group
  edi_nmt_sr: "UMD-CLIR-workQMDir-EdiNMTsr/",
  edi_nmt_srf: "UMD-CLIR-workQMDir-EdiNMTsrf/",
  # system group */
  psq: "UMD-CLIR-workQMDir-PSQ/"
};

{
  datasets: {
    train: {
      type: "paired",
      n_irrelevant: 1,
      strategy: "random",
      scores: {
        [matcher]: [
          Pathify(sw_q1_da + matchers[matcher], scrs),
          Pathify(sw_q2q3_da + matchers[matcher], scrs),
          Pathify(sw_q1_eval + matchers[matcher], scrs),
          Pathify(sw_q2q3_eval + matchers[matcher], scrs)
        ] for matcher in std.objectFields(matchers)
      },
      judgements: [
        Pathify(p, base) + "/query_annotation.tsv" for p in
          ["swso/annotations/analysis_annotation1", "swso/annotations/analysis_annotation2",
           "swso/annotations/analysis_annotation3", "swso/annotations/eval_annotation",
           "swso/annotations/dev_annotation", "swso/annotations/dev_annotation1"]
      ],
      queries: [
        Pathify(p, base) for p in
          ["swso/queries/query1/*", "swso/queries/query2/*", "swso/queries/query3/*"]
      ],
      docs: [
        Pathify(ds, scrs + "neuclir/data/") for ds in
          ["en/smt/swen/audio/ANALYSIS1/*", "en/smt/swen/docs/ANALYSIS1/*",
           "en/smt/swen/audio/ANALYSIS2/*", "en/smt/swen/docs/ANALYSIS2/*",
           "en/smt/swen/audio/EVAL1/*", "en/smt/swen/docs/EVAL1/*",
           "en/smt/swen/audio/EVAL2/*", "en/smt/swen/docs/EVAL2/*",
           "en/smt/swen/audio/EVAL3/*", "en/smt/swen/docs/EVAL3/*",
           "en/smt/swen/audio/DEV/*", "en/smt/swen/docs/DEV/*"]
          /* ["swso/segmented/ANALYSIS1/audio/*", "swso/segmented/ANALYSIS1/text/*",
           "swso/segmented/ANALYSIS2/audio/*", "swso/segmented/ANALYSIS2/text/*",
           "swso/segmented/EVAL1/audio/*", "swso/segmented/EVAL1/text/*",
           "swso/segmented/EVAL2/audio/*", "swso/segmented/EVAL2/text/*",
           "swso/segmented/EVAL3/audio/*", "swso/segmented/EVAL3/text/*",
           "swso/segmented/DEV/audio/*", "swso/segmented/DEV/text/*"] */
      ]
    },
    /* validation_paired: {
      type: "paired",
      n_irrelevant: 1,
      strategy: "random",
      scores: {
        [matcher]: [
          Pathify(sw_q1_da + matchers[matcher], scrs),
          Pathify(sw_q2q3_da + matchers[matcher], scrs),
          Pathify(sw_q1_eval + matchers[matcher], scrs),
          Pathify(sw_q2q3_eval + matchers[matcher], scrs)
        ] for matcher in std.objectFields(matchers)
      },
      judgements: [ Pathify("so/annotations/dev_annotation1", base) + "/query_annotation.tsv" ],
      queries: [ Pathify("so/queries/query1/*", base) ],
      docs: [
        Pathify(ds, base) for ds in
          ["so/segmented/DEV/audio/*", "so/segmented/DEV/text/*"]
      ]
    }, */
    validation: {
      type: "reranking",
      scores: {
        [matcher]: [
          Pathify(so_q1_dev + matchers[matcher], scrs)
        ] for matcher in std.objectFields(matchers)
      },
      judgements: [ Pathify("so/annotations/dev_annotation2", base) + "/query_annotation.tsv" ],
      queries: [ Pathify("so/queries/query1/*", base) ],
      docs: [
        Pathify(ds, scrs + "neuclir/data/") for ds in
          ["en/smt/soen/audio/DEV/*", "en/smt/soen/docs/DEV/*"]
      ]
    },
    /* test: {
      type: "reranking",
      scores: {
        [matcher]: [
          Pathify(so_q1_analysis + matchers[matcher], scrs)
        ] for matcher in std.objectFields(matchers)
      },
      judgements: [ Pathify("so/annotations/analysis_annotation1", base) + "/query_annotation.tsv" ],
      queries: [ Pathify("so/queries/query1/*", base) ],
      docs: [
        Pathify(ds, base) for ds in
          ["so/segmented/ANALYSIS1/audio/*", "so/segmented/ANALYSIS1/text/*"]
      ]
    } */
  },
  systems: std.objectFields(matchers),
  output: "datasets/english_allscores",
  logging: "info"
}
