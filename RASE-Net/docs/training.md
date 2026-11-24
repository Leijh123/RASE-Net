# Training stages

## Dependencies

```text
Fixed training split → frequency statistics → AAECA + ISRFOS → TRC base
TRC base + training data → SACF → field-guided refiner
TRC base + training annotations → exemplar bank
TRC base + SACF + refiner → frozen prediction cache
Frozen cache + exemplar bank → EGRC
```

Freeze upstream models when training each refinement stage. Use `--help` to inspect the actual backend parameters. All commands run relative to the current working directory, so checkpoint, tokenizer and output paths should be absolute or relative to that directory.

```bash
python -m rase trc -- --help
python -m rase sacf -- --help
python -m rase refiner -- --help
python -m rase bank -- --help
python -m rase cache -- --help
python -m rase egrc -- --help
python -m rase evaluate -- --help
```

For AAECA, the clean BTCV implementation exposes `prepare_facts(cfg)` and `prepare_aug(cfg)` in `rase.trc.augmentation_btcv`. Provide `data_root`, `split`, `split_sha256`, `output`, `seed`, `augmentation_count` and `rare_names`. The resulting `data/aaeca/split.json` is supplied to TRC training with `--rare_boost` enabled for the complete configuration. AAECA generation does not start automatically.

`configs/btcv.example.json` demonstrates SACF20/refiner10/EGRC12 arguments. These budgets originate from the later requested component protocol; they are not asserted as the budgets of all historical full-framework runs. TRC example arguments must be filled from the intended locked run; no fabricated epoch budget is supplied.

The historical EGRC `--resume` option restores weights, not a verified complete optimizer/scheduler/RNG continuation. Do not use it to claim an exact continuation. The later internal reference includes complete checkpoints but has its own cache identities and dependencies.

EGRC without the SCAR refiner is not a universal protocol name: the recent matched Base+EGRC experiments still used a frozen SACF field, while the historical `--no_scar` path uses its own behavior. Keep these definitions separate.
