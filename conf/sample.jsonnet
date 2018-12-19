# base path
local base = "/storage3/proj/joe/neuclir/data/so/";
# functions to generate the necessary paths
local Pathify(relative_path, base) = base + relative_path;

{
  train: {
    scores: [ Pathify("swso/scores/q1_all/", base), Pathify("swso/scores/q2q3_all/", base) ],
    judgements: [
      Pathify(p, base) + "/query_annotation.tsv" for p in
        ["swso/annotations/analysis_annotation1", "swso/annotations/analysis_annotation2",
         "swso/annotations/analysis_annotation3", "swso/annotations/eval_annotation"]
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
         "swso/segmented/EVAL3/audio/*", "swso/segmented/EVAL3/text/*"]
    ]
  },
  validation: {
    scores: [ Pathify("so/scores/q1_dev/", base) ],
    judgements: [ Pathify("so/annotations/dev_annotation1", base) + "/query_annotation.tsv" ],
    queries: [ Pathify("so/queries/query1/*", base) ],
    docs: [
      Pathify(ds, base) for ds in
        ["so/segmented/DEV/audio/*", "so/segmented/DEV/text/*"]
    ],
  },
  strategy: "difficult",
  include_scores: true,
  n_irrelevant: 1,
  output: "datasets/random",
  logging: "info"
}
