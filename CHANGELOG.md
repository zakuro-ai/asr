# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project governance files: CODEOWNERS, SECURITY.md, CONTRIBUTING.md, and issue/PR templates (#34)

### Fixed
- CI: add missing dependencies (`ascii_graph`, `tensorboard`) and skip GPU-only imports during test collection
- CI: skip `trainers.__main__` import test that requires the optional `sakura.asr_metrics` dependency

### Security
- Pin all GitHub Actions to commit SHAs to guard against supply-chain attacks (#42)

### Removed
- Dependabot configuration

## [0.4.2] - 2026-05-03

### Added
- Self-hosted runner CI workflow

### Changed
- Repository modernisation: consolidated test CI, repaired PyPI publish workflow, removed legacy files

### Removed
- Standalone runner test workflow (superseded by the modernised CI)

## [0.4.1] - 2026-04-20

### Added
- LibriSpeech ETL pipeline
- Sakura ASR model support
- Modern Python packaging via `pyproject.toml` (PEP 517/518)
- Auto-release workflow: automated version bump, GitHub release, and PyPI publish on every push to `master`

### Fixed
- Deprecated dependency upgrades across the board
- Requirements cleanup to reduce vulnerability surface

### Security
- Upgrade `protobuf` from 3.20.1 to 3.20.2 (Snyk — [#15](https://github.com/zakuro-ai/asr/pull/15))

## [0.1.0] - 2021-04-06

### Added
- Initial release with DeepSpeech-based ASR inference
- Docker support
- Zakuro (Japanese ASR) support
- Pretrained Japanese model trained on the JSUT dataset

[Unreleased]: https://github.com/zakuro-ai/asr/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/zakuro-ai/asr/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/zakuro-ai/asr/compare/0.1.0...v0.4.1
[0.1.0]: https://github.com/zakuro-ai/asr/releases/tag/0.1.0
