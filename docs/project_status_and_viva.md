# Project Status and Viva Showcase — 16 November 2026

This document is a presentation-ready inventory of what is implemented, what has been demonstrated locally, what the existing experiments report, and what remains research work. It deliberately separates the Paderborn 34-feature packaged baseline, the Paderborn V2 22-feature sequence experiments, and the supplementary CWRU study.

## 1. Project in one paragraph

The project investigates vibration-based rolling-bearing fault diagnosis using the Paderborn University Bearing Dataset as its primary source. MATLAB vibration recordings are inspected and the named `vibration_1` channel is extracted. The initial workflow converts each recording into statistical and spectral features, studies feature redundancy, and evaluates classical machine-learning and sequence-model approaches. The V2 methodology keeps measurements from each physical bearing together when splitting data, because the intended research question is whether a model can recognise a bearing it has not seen before. A separate compact 34-feature StandardScaler + Logistic Regression baseline has been packaged behind a reusable inference module and a Streamlit presentation demo. The longer-term objective is to validate generalisation robustly and then investigate genuinely lightweight edge-style inference.

## 2. Problem and motivation

Rolling bearings support rotating shafts in machines. Damage to the inner race, outer race, rolling elements or combinations of these can change the measured vibration. Unexpected bearing failure can lead to downtime and maintenance cost.

The project asks whether vibration measurements can be converted into useful numerical features and used to identify a bearing's condition. The eventual research direction is a compact model suitable for edge-style inference, but physical edge deployment has not yet been demonstrated.

## 3. Datasets

### Primary: Paderborn

- 32 physical bearing IDs in four condition groups: 6 Healthy, 12 Outer, 11 Inner and 3 Both.
- Four operating-condition identifiers, with 20 approximately four-second recordings expected for each bearing/condition combination.
- Expected source files: 32 × 4 × 20 = 2,560 MAT files.
- The later V2 feature-building run produced 2,559 usable feature rows because `N15_M01_F10_KA08_2.mat` could not be read (`TypeError: Expecting matrix here`). The failure was logged rather than stopping the whole batch.
- The files contain nested MATLAB structures and named channels. The Paderborn loader searches for `vibration_1` for vibration and `speed` for measured RPM.
- The project uses a nominal 64 kHz sampling-rate assumption. The packaged inference code does not perform timestamp correction or resampling.

### Supplementary: CWRU

- A selected 48 kHz drive-end subset: 16 recordings across four selected conditions (Healthy, Inner, Ball and Outer) and four motor loads (0–3 HP).
- The selected recordings were segmented into 140 non-overlapping one-second windows.
- The feature table contains 37 features: 9 time-domain, 13 frequency-domain and 15 envelope-domain.
- The study is supplementary. Recording-grouped splits prevent windows from the same file crossing partitions, but recordings across motor loads can belong to the same selected fault-condition series. It does not prove generalisation to independent physical bearing specimens.

### External validation dataset

UORED-VAFCLS has been identified as a candidate for a later external test. It has not yet been downloaded, inspected or integrated into the code. A separate loader/adapter is required because its MAT structure, 42 kHz rate, sensor calibration, bearing geometry and labels differ from Paderborn. HUST is another candidate for later investigation.

## 4. Main technical workflow

1. Verify bearing folders and expected MATLAB recordings.
2. Prepare metadata: filename, physical bearing ID, operating-condition ID, measurement number and fault label.
3. Load the nested MAT structure and extract the named vibration channel.
4. Inspect waveform and frequency content.
5. Extract time-, frequency- and envelope-domain features.
6. Analyse feature distributions, relationships and redundancy.
7. Split by physical bearing, fit preprocessing only on training data, and create chronological five-recording sequences for V2 models.
8. Compare classical machine-learning and sequence-model approaches.
9. Package the separate binary baseline for interactive inference.

## 5. V1 and V2 — why the experimental design changed

### V1

V1 established the initial signal-processing and modelling workflow. The major methodological concern was sample/recording-level random splitting: measurements from the same physical bearing could appear in both training and test partitions. This makes testing easier because the model may learn bearing-specific characteristics.

### V2

V2 makes the physical bearing the grouping unit. Its report describes a split of 22 training bearings, 5 validation bearings and 5 final test bearings. The feature scaler and feature ranking are fitted using training data only. Five consecutive recordings are combined into a sequence for LSTM/GRU/Transformer experiments.

The V2 report describes the five final test bearings as excluded from model selection. Before presenting a strict untouched-test claim, audit the complete experiment history and saved predictions. This audit is especially important because different experimental tracks and saved result versions exist.

## 6. Feature engineering

### Initial 26-feature Paderborn representation

The initial Notebook 03 function calculates 26 general time- and frequency-domain features. Examples include mean, standard deviation, RMS, min, max, peak-to-peak, kurtosis, skewness, crest factor, dominant frequency, dominant magnitude, spectral centroid, spectral bandwidth, spectral RMS, spectral entropy and frequency-band powers.

### V2 reduced representation

Four features were removed during redundancy analysis: RMS, spectral RMS, peak-to-peak and spectral entropy. The V2 sequence experiments retain 22 general features. ANOVA feature ranking is performed on training data only, with Top12, Top15 and All22 comparisons.

### Separate packaged baseline representation

The interactive Paderborn baseline uses 34 features: 22 general statistical/spectral features plus 6 BPFO and 6 BPFI envelope-energy features. It is a separate binary Healthy/Damaged pipeline. Do not describe it as the V2 22-feature sequence model.

## 7. Models and reported experimental results

### Paderborn V2 Stage 1 — Healthy vs Damaged

The V2 report records classical validation accuracy of 94.36% (Top12), 95.49% (Top15) and 96.18% (All22). It reports 100% test accuracy for the four tested classical models on the selected V2 test sequences. Sequence-model results include LSTM and GRU at 100%, Transformer at 94.375% and a compact Stage 1 CNN at 100% in the reported run.

These are project-reported results for the described experiment. Do not present them as universal real-world accuracy. Confirm the exact notebook/run, test-use history, bearing IDs and prediction file before making a strict untouched-test claim.

### Paderborn V2 Stage 2 — Outer vs Inner vs Both

Stage 2 is materially harder. One reported comparison lists LSTM accuracy 43.75% with macro-F1 28.34%, Transformer accuracy 45.70% with macro-F1 35.52%, and compact CNN accuracy 53.52% with macro-F1 41.57%. The project contains other Stage 2 runs with different metrics; always identify the exact run and metric file when quoting a number.

The Both class is data-limited because only three physical Paderborn bearings belong to that class. Good pooled accuracy can hide weak performance on an individual class. The next planned research step is bearing-level cross-validation on the 27 development bearings while keeping the five final bearings outside model selection.

### Separate packaged Paderborn 34-feature binary baseline

- Model: StandardScaler + Logistic Regression (`class_weight='balanced'`, `max_iter=2000`, `random_state=42`).
- Training metadata: 1,999 recordings from 25 bearings.
- Reported 320-row evaluation: 99.06% accuracy, 99.38% balanced accuracy, 99.37% F1 and ROC AUC 1.00.
- Important limitation: the evaluation involved four physical bearings and the test subset was reused during iterative experimentation. It is not a pristine untouched external test. Do not merge these metrics with V2 sequence-model results.

### Supplementary CWRU

The selected CWRU study reports 100% pooled leave-one-motor-load-out accuracy for frequency-only, envelope-only and all 37 features; time-only reports 97.86%. These results describe the selected recordings and load splits, not generalisation to independent physical bearing specimens.

## 8. Software and local demonstration

### Reusable inference module

`src/paderborn_inference.py` loads the Paderborn MAT structure, extracts `vibration_1` and measured speed, calculates the 34 features in the model's required order, loads the saved preprocessing/classifier pipeline and returns Healthy/Damaged probabilities.

### Streamlit presentation application

`streamlit_app.py` provides Overview, Run diagnosis, Experimental results and Methodology pages. The interactive predictor accepts one Paderborn MAT file and displays the binary prediction, probabilities, waveform, spectrum and extracted feature vector.

On 16 November 2026, the application was launched locally on Windows. The sample `N09_M07_F10_K001_1.mat` returned Healthy with 99.80% displayed Healthy probability and 0.20% Damaged probability; the UI displayed 3,031.1 ms processing time. This verifies that the demo pipeline executed on that sample. It is not a new accuracy evaluation, an unseen-bearing test or an edge-device benchmark.

### Offline chunk replay

`src/edge_replay.py` simulates input arriving in 4,096-sample chunks, buffers/reconstructs the full recording and performs prediction after the complete signal is available. It is an offline software replay, not live streaming and not a Raspberry Pi deployment.

## 9. What is implemented vs what remains

| Status | Work |
|---|---|
| Completed | Paderborn and selected CWRU exploratory pipelines and reported experiments |
| Completed | Reusable Paderborn 34-feature binary inference module and saved model artefact |
| Completed | Local Streamlit presentation demo and successful sample prediction |
| Completed | Offline chunk-replay utility (software simulation) |
| Completed | Ten curated notebooks: 01–07 and 09–11; Module 2–4 study notes; Windows setup documentation |
| Completed locally | Python 3.11.9; Streamlit environment and app; separate notebook virtual environment and core dependencies installed |
| Next setup step | Register the notebook kernel and launch JupyterLab; place Paderborn raw data under `data/raw/` before running data-dependent cells |
| Planned | Audit experiment history, rebuild Stage 2 sequences with explicit bearing IDs, and run bearing-level development cross-validation |
| Planned | Freeze the selected model and run an audited final Paderborn test |
| Planned | Inspect UORED and write a separate loader/feature adapter before external validation |
| Planned | TCN student, knowledge distillation, Simulated Annealing, structured pruning and INT8 quantisation |
| Not demonstrated | Physical Raspberry Pi performance, live sensor streaming, energy consumption or thermal behaviour |

## 10. A presentation-ready 90-second summary

> My project focuses on vibration-based bearing fault diagnosis with the longer-term objective of lightweight edge-oriented inference. I used the Paderborn University Bearing Dataset as the primary dataset. It contains recordings from healthy and damaged bearings under different operating conditions. I first verified the dataset and prepared metadata so that every recording remained associated with its physical bearing. I then extracted the vibration channel from the MATLAB files and calculated statistical and frequency-domain features. The initial representation contained 26 features; after redundancy analysis, 22 were retained for the V2 sequence experiments. A key improvement from V1 to V2 was splitting by physical bearing rather than randomly splitting individual measurements, because the goal is to assess performance on bearings not used for training. I compared Logistic Regression, KNN, SVM, Random Forest and sequence models such as LSTM, GRU and Transformer. Healthy-versus-damaged detection was much easier than distinguishing Outer, Inner and Both faults, where generalisation remains a challenge. Separately, I packaged a 34-feature StandardScaler and Logistic Regression baseline in a reusable inference module and built a Streamlit interface that accepts a Paderborn MAT recording and displays a prediction and signal plots. The Streamlit app is currently a local/cloud presentation demo; physical edge deployment and the compression pipeline remain future work.

## 11. Likely viva questions

### Project motivation and dataset

**Q: What problem are you solving?**

A: Identifying bearing condition from vibration measurements, first as Healthy vs Damaged and then, in the more difficult task, identifying Outer, Inner or Both damage.

**Q: Why use Paderborn?**

A: It provides measurements from multiple physical bearings, fault conditions and operating conditions, allowing the project to study bearing-level generalisation rather than only random sample-level prediction.

**Q: Why preserve the bearing ID?**

A: Because several recordings come from the same physical bearing. The ID allows every recording from one bearing to remain in a single data partition and supports per-bearing evaluation.

**Q: What is inside a Paderborn MAT file?**

A: A nested MATLAB measurement structure containing multiple named channels. The loader inspects the channel names and extracts `vibration_1` rather than assuming a fixed array position. It also reads `speed` for measured RPM.

**Q: Why did you get 2,559 rows instead of 2,560?**

A: All 2,560 expected MAT files were present, but one file caused a MATLAB matrix-reading error during feature construction. The code logged that failure and continued with the remaining valid recordings.

### Signal processing and features

**Q: What is a vibration signal?**

A: A time-ordered sequence of numerical samples from the vibration sensor. Each sample is a recorded amplitude at a particular time.

**Q: What does 64 kHz mean?**

A: The code assumes 64,000 samples per second. The packaged inference module retains this nominal rate and does not correct timestamps or resample.

**Q: What is the difference between time and frequency domain?**

A: The time domain shows how amplitude changes over time. The frequency domain shows which oscillation frequencies are present and their spectral magnitudes.

**Q: What does FFT do?**

A: It transforms a sampled time signal into frequency components. In the code, `rfft` computes the real-signal FFT and `rfftfreq` provides the frequency coordinates.

**Q: What is mean, standard deviation and RMS?**

A: Mean is the average level; standard deviation measures spread around that average; RMS is the square root of the average squared signal and describes magnitude relative to zero.

**Q: Why were four features removed?**

A: RMS, spectral RMS, peak-to-peak and spectral entropy were removed during the project's redundancy analysis to create the fixed 22-feature V2 representation. This is a project-specific feature-selection choice, not a universal rule.

### Modelling and evaluation

**Q: What is data leakage in this project?**

A: If measurements from the same physical bearing enter both train and test, the model can exploit bearing-specific patterns. V2 addresses this by grouping all measurements from one bearing into one partition.

**Q: Why scale features?**

A: Features have different numerical ranges. StandardScaler transforms them to comparable scales. It must be fitted only on training data, then applied to validation and test.

**Q: Why sequence length five?**

A: The V2 sequence experiments group five consecutive recording-feature rows, so sequence models receive a short chronological context rather than one isolated feature row. Sequences must not cross bearing boundaries or bridge missing indices.

**Q: Why is Stage 2 harder than Stage 1?**

A: Healthy/Damaged is a binary separation, while Stage 2 must distinguish multiple damage locations, including combined Both faults. The Both class has only three physical bearings in this dataset, so class support and bearing diversity are limited.

**Q: Why not claim 99% as the final system accuracy?**

A: That 99.06% figure belongs to the separate packaged 34-feature binary baseline's 320-row evaluation, which involved four physical bearings and was reused during iteration. It is not an untouched external test and must not be mixed with V2 sequence-model results.

### Deployment and future work

**Q: Is this deployed on Raspberry Pi?**

A: No. The current Streamlit app is a presentation demo running locally/on a cloud host. The offline replay simulates chunked input but buffers the full signal before prediction. No physical Raspberry Pi performance has been measured.

**Q: What does the Streamlit app demonstrate?**

A: It loads the packaged Paderborn binary baseline, accepts a compatible MAT recording, extracts the same 34 features, predicts Healthy/Damaged and visualises the waveform and spectrum.

**Q: What are the next steps?**

A: First audit experiment history and perform bearing-level Stage 2 cross-validation with per-class and per-bearing metrics. Then freeze the model, run a defensible final test, inspect an external dataset such as UORED, and only then develop and benchmark a compact TCN/distillation/compression pipeline.

## 12. Statements to avoid

- Do not say that the Streamlit app is a Raspberry Pi deployment.
- Do not call the offline 4,096-sample replay live or real-time streaming.
- Do not combine the 34-feature baseline's 99.06% result with the V2 22-feature sequence-model results.
- Do not describe CWRU recording/load splits as independent physical-bearing generalisation.
- Do not claim the TCN, distillation, Simulated Annealing, pruning or INT8 quantisation is complete.
- Do not claim UORED has already been integrated or evaluated.
- Do not call the V2 final test strictly untouched until the complete experiment history has been audited.

## 13. Current local Windows launch commands

Project root used in the current local setup:

`C:\Game\lightweight-edge-ai-bearing-fault-diagnosis-add-portable-research-notebooks`

To relaunch the Streamlit app tomorrow, open PowerShell in the project root and run:

```powershell
C:\.venv-streamlit\Scripts\python.exe -m streamlit run streamlit_app.py
```

To launch JupyterLab after registering the notebook kernel, run from the project root:

```powershell
.\.venv-notebooks\Scripts\python.exe -m ipykernel install --user --name bearing-notebooks --display-name "Python 3.11 (Bearing Notebooks)"
cd notebooks
..\.venv-notebooks\Scripts\python.exe -m jupyter lab
```

Keep the relevant PowerShell window open while using the application. Press `Ctrl+C` in that window to stop the running server.