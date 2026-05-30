# BABAPPA Git Push Readiness Report

Generated: 2026-05-30

## Release Identity

- Source version: `0.5.1-alpha`
- Release/archive label: `0.5.0-alpha`
- Package state: research-alpha, simulation-trained, conservative branch-site model family
- Scientific boundary: empirical discovery claims remain blocked pending codeml/HyPhy interpretation and simulation-matched null calibration.

## Validation Status

- Test suite: `336 passed, 58 skipped`
- Deployable package validation, working tree path: `ok`, `0 failures`, `0 warnings`
- Deployable package validation, Zenodo release path: `ok`, `0 failures`, `0 warnings`
- Manuscript compile: passed
- Manuscript PDF: `Manuscript/BABAPPA_method_paper_auxiliary_saturation.pdf`

## Cleanup Summary

- Repository size before cleanup: approximately `4.7G`
- Repository size after cleanup: approximately `207M`
- Primary deletion staging size: `4.6G`
- Manual-review deletion staging size: `13M`
- Remaining files larger than 50 MB: none found
- Cleanup manifest: `cleanup_reports/cleanup_decision_manifest.tsv`
- Deleted path manifest: `cleanup_reports/delete_paths.txt`
- Manual-review resolution: `cleanup_reports/manual_review_resolution.tsv`

Heavy raw/generated outputs were removed from the working tree or removed from the Git index when they were already tracked. Source, tests, documentation, examples, manuscript sources, the compiled manuscript PDF, cleanup reports, and release metadata were preserved.

## Zenodo Archive

- Tarball: `BABAPPA_0.5.0-alpha_release_zenodo_20260530.tar.xz`
- Tarball size: approximately `1.0M`
- SHA256:

```text
cc259617f19d9634fd6e11906903910498ab78d3797a10df1bb24b7db014dc30  BABAPPA_0.5.0-alpha_release_zenodo_20260530.tar.xz
```

- Checksum file: `BABAPPA_0.5.0-alpha_release_zenodo_SHA256SUMS.txt`
- Release folder: `release_artifacts/BABAPPA_0.5.0_alpha_clean`

The archive contains source, tests, examples, documentation, manuscript materials, deployable-model metadata/checkpoints, final validation reports, WRKY pilot evidence, cleanup reports, and release metadata. It excludes raw 10K/100K simulations, raw alignments, tensor shards, branch-site datasets, large logs, temporary directories, and raw empirical downloads.

## Git Status Summary

Current compact status after cleanup:

- `D`: 941 paths staged for removal from Git index. These are generated/heavy/ignored artifacts that should not remain in the GitHub repository.
- `M`: 1 source file modified: `src/babappa/branch/cycle39_report.py`

The source patch keeps final validation reporting truthful after intentionally pruned raw intermediates: missing raw 100K directories are classified as `pruned_after_completed_validation` when preserved summaries, truth audits, stage markers, model package, and cleanup manifests support the completed validation state.

## Recommended Git Commands

Review first:

```bash
git status
git diff --stat
find . -type f -size +50M -print -exec ls -lh {} \;
```

Stage release-ready source/docs/reports:

```bash
git add -u
git add .gitignore README.md pyproject.toml CITATION.cff LICENSE src tests docs examples Manuscript manuscript cleanup_reports GIT_PUSH_READINESS_REPORT.md
git status
```

Commit and tag:

```bash
git commit -m "Prepare BABAPPA 0.5.1-alpha research-alpha cleanup release"
git tag v0.5.1-alpha
git push origin main --tags
```

Zenodo upload should use:

```bash
BABAPPA_0.5.0-alpha_release_zenodo_20260530.tar.xz
BABAPPA_0.5.0-alpha_release_zenodo_SHA256SUMS.txt
```

Do not add `release_artifacts/` or the tarball to Git unless a separate release-asset policy is chosen.
