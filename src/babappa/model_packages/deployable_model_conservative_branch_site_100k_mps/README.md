# BABAPPA deployable conservative_branch_site model package

This package contains the retained tier checkpoints and calibration metadata from the completed 100K MPS validation.

Status: simulation-trained research-alpha; not final empirical branch-site inference.

Validate with:

```bash
babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps
```

Smoke-load with:

```bash
babappa smoke-load-deployable-model --package-dir deployable_model_conservative_branch_site_100k_mps --device auto --outdir deployable_model_load_smoke
```

Source run: `explicit_branch_truth_100k_mps`
