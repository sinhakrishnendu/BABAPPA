# BABAPPA Model Spec

## Core objects
- `M0`: neutral model (`GY94`, fixed tree/branch lengths, fixed `kappa`, fixed codon frequencies, `omega=1`).
- `phi`: fixed representation map from site data to feature vectors.
- `E_theta`: frozen quadratic energy model trained on neutral simulations only.

## Validity contract (must not be violated)
1. Train energy once on neutral draws from M0.
2. Freeze energy parameters.
3. Calibrate only with frozen energy.
4. Compute p-values with rank-based Monte Carlo + add-one correction.
5. No retraining or adaptation on observed genes or calibration replicates.

## Test statistic
For gene of length `L`:
- per-site energy: `e_i = E_hat(phi(R_i))`
- statistic: `D = Var_i(e_i)`

## Calibration
- Generate `N` null genes under M0 with the same length `L`.
- Compute `D_1..D_N` under frozen model.
- `mu0_hat = mean(D_j)`, `sigma0_hat = sd(D_j)` for reporting.
- p-value from rank test with add-one correction and selected tail.

## Guardrails
- Warn if `L/n > 0.1`.
- Fail if `L/n > 0.25` unless `--override-scaling` is explicitly set.
- Default energy family for benchmark/paper runs: `QuadraticEnergy`.

## Leakage definition
Any use of observed genes to update `phi`, `theta`, or null calibration generator beyond fixed M0 violates finite-sample conditional validity.
