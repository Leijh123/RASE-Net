# Configuration files

Each top-level key names a launcher stage. Entries are the real argparse option names without `--`; booleans enable switches only when true. CLI options after `--` override earlier configuration options. The examples do not trigger sequential training automatically.

Copy the BTCV example and edit data roots, class definitions, checkpoint versions and stage budgets for another dataset. Do not copy the strict BTCV augmentation adapter to WORD/MRI unchanged. Evaluation options are deliberately not filled with historical checkpoints from the server.
