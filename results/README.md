# Result files and their provenance

This release includes the **original saved outputs** supplied by the author.
The original model outputs were not recomputed to prepare this archive.

## `published_endpoint_stress_v2/`

Files from the executed `edge_repeated_endpoint_stress_v2.py` experiment,
including 10 outer partitions and 50 rows (five models per partition):

- `outer_split_audit.csv`
- `repeated_endpoint_stress_split_metadata.csv`
- `repeated_endpoint_stress_detailed.csv`
- `repeated_endpoint_stress_summary.csv`
- `repeated_endpoint_stress_publication_table.csv`
- `repeated_endpoint_stress_configuration.json`

The saved detailed results reproduce the endpoint-pair-disjoint quantities
in manuscript Table X, including balanced accuracy, ROC-AUC, average precision,
and thresholded confusion-matrix counts in Appendix Table C.VIII.

## `reproducibility/`

Files from the later executed reproducibility notebook (Step 146 and Step 148),
provided by the author:

- `endpoint_group_sizes.csv`: all 25,723 pairs and their record counts.
- `endpoint_group_class_composition.csv`: pair record counts and attack share.
- `outer_split_manifest.csv` and `.csv.gz`: full pair-level train/test
  assignments, all ten partitions, both sides of the boundary.
- `outer_split_row_manifest.csv.gz`: full row-index train/test assignments
  for all ten partitions, in the order of the deduplicated source table.
- `edge_configuration.json`: dataset, exclusions, split and preprocessing rules.
- `group_statistics.json`: aggregate structure **reconstructed from** the
  original pair/row CSVs; not an original notebook output.
- `published_table_C_VII_partition_composition.csv`: ten Table C.VII lines
  extracted verbatim from the executed notebook Step 148 output. That
  independent script replays the original partition search and train-only
  preprocessing without refitting models.

### Why two files have different feature counts

The earlier `published_endpoint_stress_v2/repeated_endpoint_stress_split_metadata.csv`
was retained **unaltered**. Its `feature_count` column differs from the final
paper Table C.VII in five splits (3, 5, 7, 8 and 9), due to different
preprocessing implementations/versions. The later executed independent Step 148
notebook reproduces the paper's Table C.VII feature counts: 28, 28, 33, 34,
27, 27, 33, 26, 32 and 26. Use
`reproducibility/published_table_C_VII_partition_composition.csv` when
verifying Table C.VII, and the original model-result CSV for Table X. Neither
source was silently rewritten or merged into the other.

### What was not supplied as a stored original output

Original standalone result CSVs for the placebo controls and the
transport-identity ablation were **not** present among the author-provided
files. Their results appear in the published manuscript, and code for these
arms is provided by `scripts/run_edge.py`, but no numerical replica of those
missing CSVs was synthesized or passed off as an original experiment output.

The datasets themselves are not bundled; obtain them from the original
providers to rerun all arms. No raw-record endpoint labels were used to infer
or reconstruct group-to-row relationships here; pair and row manifests were
copied from the executed notebook outputs.
