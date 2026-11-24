# Evaluation protocols

The historical comparison entry point is `python -m rase evaluate -- --help`. Provide the evaluation phase, all matching checkpoints, bank and cache explicitly. Report BTCV as validation, not test. Keep 512 and 256 evaluations distinct; distance units are pixels on the evaluation grid.

The archived `rase.evaluation.legacy.seg_metrics` implementation is preserved for compatibility, including its exact distance-transform behavior and NaN results when either mask is empty. It computes distances to the foreground mask rather than a newly substituted boundary-only distance map. Do not describe it as a different metric formula or silently replace NaNs with a finite diagonal penalty. This is different from later finite-penalty full-metric jobs.

For query-level Mean, average valid query values. For organ-level Macro, average per-organ means. mIoU is Macro IoU and is distinct from query-weighted Mean IoU. Empty-mask handling changes the valid denominators of distance metrics; publish per-metric counts. FEWSHOT pooling must identify both its organ set and its aggregation rule.

Select model weights and residual coefficients with the declared validation protocol. Requested epoch1 evaluations are additional analyses and must not overwrite the originally validation-selected run. Never attach old overlap metrics to newly computed distance metrics from another prediction pass.
