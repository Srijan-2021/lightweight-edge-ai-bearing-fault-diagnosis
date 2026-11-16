# Curated results

This folder is reserved for small, presentation-ready result summaries, figures and audit tables that can be understood without the raw vibration recordings.

Large intermediate feature matrices, raw signal exports, per-run caches, temporary predictions and model-training checkpoints should remain outside Git unless a specific artefact is selected for reproducibility and its size and provenance are reviewed.

The repository's `docs/experiment_results.md` and `docs/limitations.md` contain the current consolidated metrics and their evaluation caveats. In particular:

- Paderborn's 34-feature binary baseline is a separate packaged inference track.
- Paderborn's 22-feature sequence experiments are a separate V2 research track.
- CWRU is a supplementary selected-subset study and should not be described as independent physical-bearing generalisation.
