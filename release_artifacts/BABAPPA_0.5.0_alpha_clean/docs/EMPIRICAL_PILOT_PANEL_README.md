# Empirical Pilot Panel Template

Use `empirical_pilot_panel_template.tsv` to build a small, curated diagnostic panel before any broad empirical scan. BABAPPA remains simulation-trained research-alpha software and this workflow is not final empirical inference.

Suggested categories:

- known positive immune or detoxification gene families
- likely negative housekeeping genes
- alignment-sensitive families
- saturated families
- short or low-information families
- paralogy-risk families

Each row must provide a CDS FASTA, a matching tree, a foreground branch or taxon, an expected category, and the reference status for codeml/HyPhy-style comparison. Reference tools are external probes of behavior and failure modes; BABAPPA should not be tuned to mimic them mechanically.

Run:

```bash
babappa validate-empirical-pilot-panel --panel-manifest empirical_pilot_panel_template.tsv --outdir empirical_pilot_panel_validation
```

Scores are diagnostic until input QC, applicability/OOD checks, simulation-matched calibration, and reference interpretation are complete.
