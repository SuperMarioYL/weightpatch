# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-05

### Added

- `weightpatch diff`: compare two local safetensors checkpoints into a sparse
  tensor diff with per-tensor digests and changed slice ranges.
- `weightpatch apply` / `weightpatch revert`: write, verify, apply, stack and
  revert a `.wpatch` file with pre-image checks and slice-overlap conflicts.
- `weightpatch screen`: report kept, missing and reshaped edited tensors in a
  quantized export or a single-file GGUF.

[Unreleased]: https://github.com/SuperMarioYL/weightpatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SuperMarioYL/weightpatch/releases/tag/v0.1.0
