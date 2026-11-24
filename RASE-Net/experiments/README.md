# Internal component experiments

`internal_reference` preserves the recent matched BTCV source for scientific inspection: direct-fusion/name-only SCAR, random retrieval and static-gate EGRC, joint retrieval and complete state-aware EGRC. It uses original Base predictions for EGRC and a frozen historical SACF, without TRC or SCAR-refiner output. A source file is not evidence that the historical SACF was trained exclusively on training cases; verify that lineage separately.

These reference files depend on the original frozen recipe, source manifests and read-only prediction stores. They are not standalone runnable examples and are not used by the main CLI. No caches, server launchers or experiment-specific secrets are bundled.
