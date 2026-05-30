# Apple Silicon / MPS

BABAPPA supports Apple Silicon / Metal Performance Shaders as a research-alpha neural backend. This path is intended for the MacBook Pro M5 Max with 36 GB unified memory and for conservative explicit branch-truth validation planning.

## Environment

Recommended user-run exports before Python starts:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

Do not set `CUDA_VISIBLE_DEVICES` in Mac scripts. Do not use NVIDIA monitoring commands on macOS. `PYTORCH_MPS_HIGH_WATERMARK_RATIO` is intentionally left unset unless the user explicitly chooses a value.

Inspect the environment:

```bash
babappa check-neural-env
```

The report includes platform, machine, macOS version, Python executable, torch version, CUDA availability, MPS built/available, the recommended device, `PYTORCH_ENABLE_MPS_FALLBACK`, and `PYTORCH_MPS_HIGH_WATERMARK_RATIO`.

## Unified-Memory Guidance

36 GB unified memory is shared by the OS, Python, tensors, aligner processes, file cache, and the MPS backend. Use smaller batch sizes than CUDA, streamed output construction, capped branch-site datasets, and tier-by-tier resume checkpoints.

For the conservative 10K MPS path, start at batch size `128`. For 100K planning, use batch size `64` until 10K validates and the benchmark helper says a larger batch is safe.

## Smoke And Benchmark

Run the MPS smoke:

```bash
babappa smoke-mps-training --outdir mps_smoke --device auto --batch-size 32 --max-items 512
babappa validate-mps-smoke --smoke-dir mps_smoke
```

Run the lightweight benchmark:

```bash
babappa benchmark-apple-silicon --outdir apple_silicon_benchmark --device auto --batch-sizes 32,64,128 --max-items 1024
```

These commands are portability and sizing helpers only. Do not use them as scientific performance metrics.

## BABAPPAlign Model Cache

BABAPPAlign requires the BABAPPAScore model at `$HOME/.cache/babappalign/models/babappascore.pt`. Check the executable and model cache before running Mac plans:

```bash
babappa check-aligners --json-out aligner_status.json
babappa smoke-aligner --method babappalign --outdir aligner_smoke
```

If the model is missing, install it with:

```bash
mkdir -p "$HOME/.cache/babappalign/models"
curl -L "https://zenodo.org/record/18053201/files/babappascore.pt" -o "$HOME/.cache/babappalign/models/babappascore.pt"
```

The Mac 10K and 100K plan scripts stop before any tier starts if `babappalign` is requested and this model is absent. Use `--allow-missing-babappalign` only when intentionally testing a plan without BABAPPAlign.

## Conservative 10K MPS Plan

Generate the Mac-specific plan:

```bash
babappa plan-explicit-branch-truth-10k-mac --outdir explicit_branch_truth_10k_mps_plan --n-families-per-tier 2500 --tiers low,moderate,high,extreme --methods identity,mafft,babappalign,muscle --feature-policy conservative_branch_site --truth-mode explicit --negative-downsample-ratio 5 --max-output-rows-per-tier 1000000 --device mps --batch-size 128 --threads 8 --conda-env molevo
```

Then the user may run:

```bash
bash explicit_branch_truth_10k_mps_plan/run_explicit_branch_truth_10k_mps.sh
```

The generated scripts are marked user-run only. They run `babappa preflight-explicit-branch-truth-mps-plan` before the first simulation command, use macOS-safe conda initialization with nounset guarded around activation, use a mkdir-based lock with stale-lock instructions, run one tier at a time, run one stage at a time, use `.stage_complete_<stage>` markers, reuse validated existing outputs, build streamed/capped branch-site datasets, and avoid CUDA/NVIDIA commands.

Run the lightweight script validator:

```bash
babappa validate-mps-plan-script --plan-dir explicit_branch_truth_10k_mps_plan
```

Run the full fast preflight:

```bash
babappa preflight-explicit-branch-truth-mps-plan --plan-dir explicit_branch_truth_10k_mps_plan --scale 10k --require-babappalign true --require-mps true --conda-env molevo
```

Monitor:

```bash
bash explicit_branch_truth_10k_mps_plan/monitor_explicit_branch_truth_10k_mps.sh
```

Validate:

```bash
bash explicit_branch_truth_10k_mps_plan/validate_explicit_branch_truth_10k_mps.sh
```

## 100K Gate

Do not run 100K until conservative explicit branch-truth 10K MPS completes and validates. The 100K planner exists only to prepare a resumable path:

```bash
babappa plan-explicit-branch-truth-100k-mac --outdir explicit_branch_truth_100k_mps_plan --n-families-per-tier 25000 --tiers low,moderate,high,extreme --methods identity,mafft,babappalign,muscle --feature-policy conservative_branch_site --truth-mode explicit --negative-downsample-ratio 5 --max-output-rows-per-tier 2000000 --device mps --batch-size 64 --threads 8 --conda-env molevo
```

100K may require multiple days and large disk space. It must remain tier-resumable and never monolithic.

## If MPS Fails

First retry with `PYTORCH_ENABLE_MPS_FALLBACK=1` and a smaller `--batch-size`. If the failure persists, rerun the neural stage with `--device cpu`. Keep the same output naming and stage markers so validation can still explain exactly which tier and stage completed.
