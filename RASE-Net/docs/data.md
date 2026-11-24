# Data preparation

Use fixed patient-level splits and keep the official partitions where applicable. No command in this release generates a random patient split.

| Dataset | Existing patient counts | Evaluation set |
|---|---|---|
| BTCV | 24 training / 6 validation | Validation; no independent test |
| WORD | 100 training / 20 validation / 30 test | Test |
| AMOS MRI | 32 training / 8 validation / 20 evaluation | Verify exact official-partition correspondence from the original case lists |

The AMOS MRI tables evaluate 13 abdominal structures. Do not describe the evaluated label count as 15 merely because the original AMOS taxonomy contains 15 labels. Verify effective labels and case membership for each local dataset release.

```text
data/BTCV/
  image/   # 512×512 PNG slices
  label/   # scipy sparse NPZ; filename records tensor shape
  imask/   # interactive-instance NPY masks
  split.json
```

Split JSON uses `training`, `validation`, optional `test`, and `labels`. Items provide `image`, `label` and `imask` paths relative to the data root; synthetic items may provide absolute paths and `patient_id`. Foreground label names and channel order must match the checkpoint. Background occupies label key `0`.

Only training cases contribute frequency counts, synthetic samples and the exemplar bank. Training retrieval excludes the query patient. Do not place validation/test annotations in the bank. Pretrained SAM-compatible IMIS weights and the text tokenizer are external prerequisites of the historical scripts, not random-initialization defaults.
