# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows
[Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-11-09

### Added

- `joblens rank` scores many postings, given as files or directories, against
  one resume, with text, Markdown, JSON and CSV reports.
- Summary of the required skills missing most often across all postings.
- `--top` and `--min-score` options for `rank`.
- `rank_jobs()`, `Ranking` and three more fictional example postings.

### Changed

- The embeddings backend encodes a repeated resume only once.

### Fixed

- Corrupted PDFs now produce a clear error instead of a traceback.

## [0.3.0] - 2026-11-08

### Added

- Optional `embeddings` similarity backend using sentence-transformers,
  installed with the `joblens[embeddings]` extra.
- `--similarity` and `--model` CLI options, and a `similarity` parameter on
  `match()` that accepts any backend.
- `MatchResult.similarity_method`, also shown in text and Markdown reports.
- `integration` pytest marker for tests that load a real model.

## [0.2.0] - 2026-11-07

### Added

- Skill implication graph (`data/implications.json`): specific skills imply
  broader ones, e.g. FAISS -> Vector Databases, PyTorch -> Deep Learning ->
  Machine Learning.
- `SkillExtractor.infer()` returns implied skills and the explicit skills they
  came from.
- `MatchResult.inferred_matches` and a "Credited via related" line in text and
  Markdown reports.
- `--no-inference` CLI flag and `infer_skills` parameter for strict matching.

## [0.1.0] - 2026-11-06

### Added

- Symbol-aware skill extraction over a curated tech-skill taxonomy.
- Required vs. preferred requirement parsing, including inline cues such as
  "is an asset".
- Explainable match scoring blending weighted skill coverage and TF-IDF
  similarity.
- CLI with text, Markdown and JSON output; `.txt`, `.md` and `.pdf` input.
- pytest suite and GitHub Actions CI on Python 3.10-3.12.

[0.4.0]: https://github.com/jonahernan99/joblens/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jonahernan99/joblens/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jonahernan99/joblens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jonahernan99/joblens/releases/tag/v0.1.0
