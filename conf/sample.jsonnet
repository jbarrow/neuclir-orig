# base path
local base = "/storage3/proj/joe/neuclir/data/so/";
# functions to generate the necessary paths
local Pathify(relative_path, base) = base + relative_path;

{

  datasets: {
    train: {
      type: "paired",
      n_irrelevant: 1,
      strategy: "random",
      scores: {
        smt: [ Pathify("swso/scores/q1_all/", base), Pathify("swso/scores/q2q3_all/", base) ]
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
        Pathify(ds, base) for ds in
          ["swso/segmented/ANALYSIS1/audio/*", "swso/segmented/ANALYSIS1/text/*",
           "swso/segmented/ANALYSIS2/audio/*", "swso/segmented/ANALYSIS2/text/*",
           "swso/segmented/EVAL1/audio/*", "swso/segmented/EVAL1/text/*",
           "swso/segmented/EVAL2/audio/*", "swso/segmented/EVAL2/text/*",
           "swso/segmented/EVAL3/audio/*", "swso/segmented/EVAL3/text/*",
           "swso/segmented/DEV/audio/*", "swso/segmented/DEV/text/*"]
      ]
    },
    validation_paired: {
      type: "paired",
      n_irrelevant: 1,
      strategy: "random",
      scores: {
        psq: Pathify("so/scores/q1_dev/systems/somali-customPSQ-Q1-DEV/query-analyzer-umd-v10.2_matching-umd-v11.0_evidence-combination-v11.2/SO_QUERY1_DEV_customPSQ_Cutoff2/UMD-CLIR-workQMDir-customPSQ/", base),
        smt: Pathify("so/scores/q1_dev/systems/somali-SMT-Q1-DEV/query-analyzer-umd-v10.2_matching-umd-v11.0_evidence-combination-v11.2/SO_QUERY1_DEV_SMT_Cutoff2/UMD-CLIR-workQMDir-SMT/", base)
      },
      judgements: [ Pathify("so/annotations/dev_annotation1", base) + "/query_annotation.tsv" ],
      queries: [ Pathify("so/queries/query1/*", base) ],
      docs: [
        Pathify(ds, base) for ds in
          ["so/segmented/DEV/audio/*", "so/segmented/DEV/text/*"]
      ]
    },
    validation: {
      type: "reranking",
      scores: {
        psq: Pathify("so/scores/q1_dev/systems/somali-customPSQ-Q1-DEV/query-analyzer-umd-v10.2_matching-umd-v11.0_evidence-combination-v11.2/SO_QUERY1_DEV_customPSQ_Cutoff2/UMD-CLIR-workQMDir-customPSQ/", base),
        smt: Pathify("so/scores/q1_dev/systems/somali-SMT-Q1-DEV/query-analyzer-umd-v10.2_matching-umd-v11.0_evidence-combination-v11.2/SO_QUERY1_DEV_SMT_Cutoff2/UMD-CLIR-workQMDir-SMT/", base)
      },
      judgements: [ Pathify("so/annotations/dev_annotation1", base) + "/query_annotation.tsv" ],
      queries: [ Pathify("so/queries/query1/*", base) ],
      docs: [
        Pathify(ds, base) for ds in
          ["so/segmented/DEV/audio/*", "so/segmented/DEV/text/*"]
      ]
    }
  },
  systems: ["smt"],
  output: "datasets/normalized",
  logging: "info"
}
