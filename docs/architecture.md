# System Architecture

## 1. System overview

The project contains two related but distinct experimental tracks:

- **Paderborn:** the primary binary healthy/damaged diagnosis pipeline and the saved local inference/replay workflow.
- **CWRU:** a supplementary four-class feature-engineering and evaluation study. It is not the deployed Paderborn model and uses a different feature schema and label set.

Keeping these tracks separate prevents accidental mixing of their datasets, features, labels, and metrics.

## 2. Paderborn primary inference path

The currently documented baseline has 34 engineered inputs and a scikit-learn pipeline consisting of **StandardScaler** followed by class-balanced **Logistic Regression**.

~~~mermaid
flowchart TD
    A[Recorded Paderborn vibration signal] --> B[Input validation]
    B --> C[Signal buffering / offline replay]
    C --> D[Time-domain and spectral feature extraction]
    D --> E[Envelope features using bearing characteristic frequencies]
    E --> F[34-feature vector]
    F --> G[Saved StandardScaler]
    G --> H[Saved Logistic Regression classifier]
    H --> I[Healthy / Damaged prediction]
    H --> J[Class probability output]
~~~

### Feature representation

The Paderborn feature schema contains 34 inputs:

- 22 general signal features.
- 6 envelope-related BPFO features.
- 6 envelope-related BPFI features.

The envelope calculations use bearing geometry and rotational speed information according to the project's feature configuration. Measured RPM is used where available. The inference code and saved configuration are the source of truth for the exact feature names, order, filtering and numerical settings; this diagram is a conceptual overview.

### Inference and replay scope

The local replay utility processes an existing MAT recording in successive chunks (currently documented as 4,096 samples per chunk) and passes accumulated signal data through the inference workflow. This is an offline software simulation.

It is not:
- a live vibration sensor connection;
- a physical Raspberry Pi deployment;
- a hardware latency, power or memory benchmark;
- proof of hard real-time behaviour.

## 3. CWRU supplementary analysis path

The selected 48 kHz drive-end subset follows a separate four-class pipeline:

~~~mermaid
flowchart TD
    A[CWRU MAT recordings] --> B[48 kHz drive-end signal extraction]
    B --> C[Non-overlapping 1-second windows]
    C --> D[9 time-domain features]
    C --> E[13 frequency-domain features]
    C --> F[15 envelope-domain features]
    D --> G[Combined 37-feature table]
    E --> G
    F --> G
    G --> H[StandardScaler + Logistic Regression]
    H --> I[LOLO evaluation]
    H --> J[Recording-grouped evaluation]
    G --> K[Feature-group ablation]
~~~

### CWRU evaluation design

The selected subset contains 16 recordings: four class-condition series (Healthy, Inner, Ball and Outer) at 0, 1, 2 and 3 HP. The recordings were segmented into 140 non-overlapping one-second windows.

Two evaluation schemes were used:

1. **Leave-one-motor-load-out (LOLO):** all windows at one motor load are held out together.
2. **Recording-grouped cross-validation:** all windows from a given MAT filename are kept in one fold.

The second split prevents same-file windows from being split across train and test. However, filenames at different loads share the same selected fault-condition key (for example, IR007 across the four loads). Thus the recording-grouped result is not independent physical-bearing validation.

### CWRU feature groups

| Group | Count |
|---|---:|
| Time-domain | 9 |
| Frequency-domain | 13 |
| Envelope-domain | 15 |
| Combined | 37 |

The ablation evaluated seven combinations. Frequency-only (13 features) and Envelope-only (15 features) each achieved 100% in the selected LOLO and recording-grouped tests; these are dataset/subset-specific results, not a generalisation guarantee.

## 4. Model artefacts

The Paderborn inference artefacts are expected to include:

- paderborn_baseline_pipeline.joblib: fitted preprocessing and classifier pipeline.
- feature_config.json: feature names/order and extraction settings.
- metadata.json: model/dataset metadata.
- paderborn_inference.py: reusable inference implementation.
- edge_replay.py: offline recorded-signal replay utility.

The actual committed files and their versions will determine the final repository inventory. No placeholder file should be treated as a working model.

## 5. Reproducibility boundary

A new user should be able to distinguish:
- source code from experimental notebooks;
- fitted model artefacts from the raw datasets;
- Paderborn model inference from CWRU supplementary evaluation;
- offline replay from real hardware deployment.

Exact software versions, file manifests, dataset acquisition instructions and commands will be documented after the local project files are reviewed and the dependency specification is captured.
