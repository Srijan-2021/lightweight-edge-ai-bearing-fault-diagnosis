# Lightweight Edge AI for Bearing Fault Diagnosis

A research prototype for vibration-based bearing fault diagnosis using signal processing, engineered features, compact machine-learning models, and an offline edge-inference simulation.

> **Project status:** Experimental evaluation is complete for the current Paderborn baseline and the selected CWRU supplementary subset. The Streamlit presentation demo, standalone Paderborn inference source and compact baseline artefact are included; the wider research notebooks and additional model artefacts are being curated separately. This is an academic prototype, not a validated industrial predictive-maintenance product.

## Overview

Rolling-bearing faults can produce characteristic changes in vibration signals. This project investigates a lightweight diagnosis pipeline that converts vibration measurements into engineered features and uses a compact classifier to distinguish healthy from damaged bearings.

~~~text
Vibration signal
      |
      v
Signal validation and segmentation
      |
      v
Time / frequency / envelope feature extraction
      |
      v
Feature scaling
      |
      v
Compact classifier
      |
      v
Predicted condition + class probability
~~~

The implementation also includes a local offline replay path that feeds a saved inference pipeline with successive signal chunks. The replay is a software simulation; it is **not** a live sensor stream or a real-time hardware benchmark.

## Research objectives

- Develop a compact, reproducible vibration-based bearing-fault diagnosis workflow.
- Compare engineered time-, frequency- and envelope-domain representations.
- Evaluate a simple, lightweight machine-learning baseline before introducing more complex models.
- Investigate performance under different operating conditions and data-splitting strategies.
- Package inference so that the model can be exercised locally and inspected independently of the training notebook.

## Datasets

### Paderborn University Bearing DataCenter — primary dataset

The Paderborn dataset is the primary dataset for model development. The project uses vibration recordings and bearing-condition information to construct a binary healthy/damaged diagnosis task.

Dataset source: [Paderborn University Bearing DataCenter](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter)

The current baseline uses 34 engineered features: 22 general signal features plus bearing-geometry-informed envelope features around BPFO and BPFI harmonics. The implementation uses measured rotational speed where available.

### Case Western Reserve University (CWRU) — supplementary dataset

A selected 48 kHz drive-end subset is used for supplementary four-class experiments:

- Healthy
- Inner-race fault
- Ball fault
- Outer-race fault

The selected subset contains 16 recordings across four motor loads (0, 1, 2 and 3 HP), segmented into 140 non-overlapping one-second windows. A 37-feature table combines 9 time-domain, 13 frequency-domain and 15 envelope-domain features.

Dataset source: [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)

**Dataset note:** Raw dataset files are not included in this repository. Obtain data from the original providers and follow their access, attribution and usage conditions. The CWRU subset used here contains one selected fault-condition series per fault class across loads; the evaluation does not establish generalisation to entirely unseen physical bearing specimens.

## Models and experimental results

### Paderborn baseline

The principal compact baseline is a scikit-learn pipeline:

- StandardScaler
- LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)

The saved inference artefact was approximately 3.25 KiB in the project environment, excluding Python and library dependencies.

In the project's reported 320-sample evaluation, the binary baseline achieved:

| Metric | Result |
|---|---:|
| Accuracy | 99.06% |
| Balanced accuracy | 99.38% |
| F1-score | 99.37% |
| ROC AUC | 1.00 |

These are results from the current project experiment, **not** a claim of universal bearing-fault accuracy. The evaluation involved four physical bearings, and the test subset was reused during iterative project experimentation. It should therefore not be described as a pristine, untouched external test.

### CWRU supplementary experiments

A Logistic Regression baseline was evaluated with leave-one-motor-load-out (LOLO) splits and recording-grouped cross-validation.

| Feature configuration | Features | LOLO pooled accuracy | Recording-grouped test accuracy |
|---|---:|---:|---:|
| Frequency-only | 13 | 100% | 100% |
| Envelope-only | 15 | 100% | 100% |
| All features | 37 | 100% | 100% |

The broader ablation also evaluated time-only and pairwise feature combinations. Time-only achieved 97.86% pooled LOLO accuracy.

**Interpretation limitation:** Recording-grouped splitting prevents windows from the same file appearing in both training and test sets, but recordings at different loads can belong to the same selected fault-condition series. Therefore, these results demonstrate consistency across the selected recordings and loads, not independent physical-bearing generalisation. The small selected subset also limits the strength of conclusions.

## Inference and edge-simulation scope

The project includes a saved Paderborn inference pipeline and a local signal-replay utility. The replay feeds successive chunks (currently 4,096 samples per chunk) to the inference workflow.

Run the offline replay from the repository root after installing the dependencies and placing a valid Paderborn MAT file on your machine:

```bash
python src/edge_replay.py --mat-file path/to/recording.mat --chunk-size 4096
```

The script checks that chunk reconstruction exactly matches the original signal, performs prediction after the full signal has been buffered, prints timing breakdowns and saves a JSON report under `results/edge_replay/`. It is an accelerated offline simulation, not true real-time streaming.

The scope is deliberately described accurately:

- Local/offline replay of recorded signals.
- Model inference using saved preprocessing and classifier artefacts.
- No physical Raspberry Pi or other edge device was used for benchmarking.
- No live sensor acquisition, latency guarantee, power measurement or real-time deployment claim is made.

A future hardware study should measure end-to-end latency, memory use, energy consumption and robustness on the target device.

## Research notebooks

The repository includes a cleaned starter sequence of six Paderborn research notebooks. See [notebooks/README.md](notebooks/README.md) for their order and how to run them. Install the separate research environment with `requirements-research.txt`; the Streamlit app continues to use the smaller `requirements.txt`.

Raw Paderborn/CWRU recordings and generated intermediate datasets are not committed. See [data/README.md](data/README.md) for the local folder layout.

## Streamlit presentation demo

The repository includes an interactive Streamlit app for presentation and demonstration. It accepts a Paderborn `.mat` recording, extracts the packaged 34-feature representation, and predicts Healthy or Damaged using the saved StandardScaler + Logistic Regression baseline.

Run locally with Python 3.11:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

For public hosting, merge the app files into `main`, then select this repository, the `main` branch and `streamlit_app.py` as the entrypoint in Streamlit Community Cloud. See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md).

**Scope:** the hosted application is a cloud-based presentation demonstrator. Its interactive predictor is the standalone Paderborn 34-feature binary baseline, not the V2 sequence-model Stage 1/Stage 2 system. It is not a physical Raspberry Pi deployment or an edge-device benchmark. Raw MAT recordings are not committed.

## Documentation

- [Paderborn dataset and metadata — Module 2](docs/dataset_and_metadata.md)

- [System architecture](docs/architecture.md)
- [Experimental results summary](docs/experiment_results.md)
- [Literature review and research positioning](docs/literature_review.md)
- [Evaluation, overfitting and deployment limitations](docs/limitations.md)

## Repository contents

The intended organisation is:

~~~text
src/                 Inference and replay code
models/              Small trained model artefacts and metadata
notebooks/           Selected, cleaned experimental notebooks
results/selected/    Curated tables and presentation figures
docs/                Methodology, architecture, literature review and limitations
~~~

The repository now includes the Streamlit presentation app, the standalone Paderborn inference module, and its compact model artefact. Additional research notebooks, sequence-model artefacts and curated outputs are being added only after review. Raw vibration datasets, virtual environments, caches and temporary outputs should remain outside version control.

## Reproducibility

The Streamlit app includes a pinned Python 3.11 dependency file and a model-loading smoke test. The broader research notebooks still require their own documented data paths and execution sequence; the app's smoke test does not validate accuracy on new recordings.

Curated results and version-control guidance are documented in [results/README.md](results/README.md).

## Limitations

1. Results are dataset- and split-specific; metrics from different papers are not directly comparable unless their datasets, class definitions and split protocols match.
2. The selected CWRU subset does not provide a sufficient independent-specimen test.
3. The Paderborn baseline's reported test subset was reused during iterative experimentation.
4. Offline replay is not equivalent to live streaming or deployment on physical edge hardware.
5. Feature-extraction and inference costs have not been benchmarked end-to-end on a target edge device.

See [docs/limitations.md](docs/limitations.md) for the full interpretation and future-validation requirements.

## Acknowledgements and data sources

- [Paderborn University Bearing DataCenter](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter)
- [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter)

Please consult each dataset provider's terms before downloading, redistributing or using the data.
