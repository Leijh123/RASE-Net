# Implementation and checkpoint compatibility

| Runtime | EGRC architecture | Intended use |
|---|---|---|
| `rase/_historical/egrc.py` | Learned token offset, organ gate | Historical pipeline and matching historical weights |
| `rase/models/egrc.py` | Coordinate projection, state MLP plus organ gate, validity mask; gate applied to attention, vote and residual | Recent matched internal component experiments and new integrations |

The CLI launches only the historical runtime. Importing the clean model package does not silently replace the historical implementation. New state-aware training should use a matching trainer/cache protocol; the reference source is retained under `experiments/internal_reference` and is not advertised as a standalone CLI.

The clean BTCV AAECA implementation comes from the later fixed-split dataflow code. Its checks are intentionally specific to 24/6 patients, 1766/407 slices, 13 classes and 512 grids. Do not use it for WORD or MRI without an explicit dataset adapter and corresponding verification. Historical preprocessing scripts remain in the backend for reference.

No historical numerical result is reassigned to another implementation. The provenance manifest records the source of each imported file and the portable path changes. The original servers and experiments are left unchanged.
