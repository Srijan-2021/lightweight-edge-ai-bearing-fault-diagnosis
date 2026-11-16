# Research notebooks

Run the notebooks in numerical order from a Jupyter session started inside this folder:

```bash
cd notebooks
python -m pip install -r ../requirements-research.txt
jupyter lab
```

The first six curated notebooks currently in this repository are:

| Notebook | Purpose |
|---|---|
| `01_dataset_verification.ipynb` | Check bearing folders and expected Paderborn MATLAB recordings |
| `02_metadata_preparation.ipynb` | Build bearing, operating-condition, measurement and label metadata |
| `03_feature_extraction.ipynb` | Extract time- and frequency-domain vibration features |
| `04_feature_analysis.ipynb` | Inspect distributions, relationships and feature redundancy |
| `05_data_split_and_sequences.ipynb` | Create bearing-disjoint partitions and chronological sequences |
| `06_classical_baselines.ipynb` | Evaluate classical baselines for the prepared tasks |

## Data paths

The notebooks expect the original data and generated intermediate files in the project layout described in [../data/README.md](../data/README.md). Raw MATLAB data is not stored in Git.

The early Paderborn notebooks use relative paths such as `../data/raw` and `../data/processed`, so start Jupyter with its working directory set to this `notebooks` folder.

## Notebook hygiene

Committed notebooks have cleared cell outputs to avoid embedding large tables, plots or run-specific machine paths. Run cells locally to regenerate outputs. The notebooks are research records; a successful JSON/syntax check is not the same as a complete end-to-end reproduction.

The sequence-model experiments, baseline-packaging pipeline, continuous-signal audit and CWRU supplementary study are documented separately in the project notes and are being curated into portable notebook files in a later pass. Do not assume they are all included in this six-notebook set.
