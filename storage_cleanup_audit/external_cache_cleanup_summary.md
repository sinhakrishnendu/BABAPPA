# External Cache Cleanup Summary

Generated: 2026-05-30

## Action Taken

Deleted only the generated BABAPPAlign embeddings cache:

`/Users/krishnendu/.cache/babappalign/embeddings`

The BABAPPAlign required model was preserved:

`/Users/krishnendu/.cache/babappalign/models/babappascore.pt`

## Disk Space

Before cleanup:

```text
Filesystem      Size    Used   Avail Capacity
/dev/disk3s5   1.8Ti   1.3Ti   548Gi    71%
```

After cleanup:

```text
Filesystem      Size    Used   Avail Capacity
/dev/disk3s5   1.8Ti   108Gi   1.7Ti     6%
```

Approximate space recovered: about `1.1 TiB`.

## Current Sizes

- `/Users/krishnendu/.cache`: `3.2G`
- `/Users/krishnendu/.cache/babappalign`: `47M`
- `/Users/krishnendu/Documents/GitHub/BABAPPA`: `210M`
- `/Users/krishnendu`: `69G`

## Validation

- `babappa check-aligners` confirms the BABAPPAlign model cache is present.
- `babappa validate-deployable-model-package` passed.
- `babappa validate-empirical-evidence-pack` passed for `WRKY_candidate_02_close`.

No source code, tests, docs, deployable model package, final reports, evidence packs, or real empirical CDS/tree inputs were deleted.
