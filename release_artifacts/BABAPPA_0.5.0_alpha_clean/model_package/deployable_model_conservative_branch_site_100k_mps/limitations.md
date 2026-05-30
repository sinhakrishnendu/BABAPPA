# Limitations

- Simulation-trained and simulation-supervised only.
- Explicit simulator branch-site truth was used for validation, not empirical inference.
- Conditional pass reflects pruned raw/intermediate artifacts after retained validation outputs were preserved.
- Foreground context columns remain a known leakage/OOD caution.
- Context-only shortcut risk remains a required empirical calibration caution.
- Final empirical branch-site inference claims are not supported by this package alone.
