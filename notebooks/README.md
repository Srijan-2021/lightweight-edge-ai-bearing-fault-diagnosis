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
| `07_teacher_models.ipynb` | Train and evaluate LSTM, GRU and Transformer teacher models |
| `09_baseline_model_packaging.ipynb` | Package and verify the separate 34-feature binary baseline |
| `10_continuous_signal_pipeline.ipynb` | Inspect signal timing, package inference and audit continuous-recording predictions |
| `11_cwru_dataset_inspection.ipynb` | Run the supplementary CWRU 48 kHz feature and evaluation study |

## Module 2 — Dataset and metadata

Read [Dataset and Metadata](../docs/dataset_and_metadata.md) alongside notebooks 01 and 02. It explains the expected bearing/recording layout, filename structure, nested MATLAB channels, the two label schemes, and why bearing IDs must be preserved.

## Module 3 — Vibration signals

Read [Understanding Vibration Signals](../docs/vibration_signals.md) alongside notebook 03. It explains how the sensor samples become a waveform, what amplitude and frequency mean, how the FFT view differs from the time view, and why the plotted dominant frequency is not itself a fault diagnosis.

## Module 4 — Time-domain features

Read [Mean, Standard Deviation and RMS](../docs/time_domain_features.md) alongside notebook 03. It explains the first three signal statistics with a worked example and the actual Paderborn example values.

## Data paths

The notebooks expect the original data and generated intermediate files in the project layout described in [../data/README.md](../data/README.md). Raw MATLAB data is not stored in Git.

The early Paderborn notebooks use relative paths such as `../data/raw` and `../data/processed`, so start Jupyter with its working directory set to this `notebooks` folder.

## Notebook hygiene

Committed notebooks have cleared cell outputs to avoid embedding large tables, plots or run-specific machine paths. Run cells locally to regenerate outputs. The notebooks are research records; a successful JSON/syntax check is not the same as a complete end-to-end reproduction.

Notebook 07 contains the sequence teacher-model experiments, while notebook 09 covers packaging the separate 34-feature binary baseline. The large combined binary-detection notebook (08) remains excluded because it duplicates parts of the staged workflow and needs a more substantial split and review.
