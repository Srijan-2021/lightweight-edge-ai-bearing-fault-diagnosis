# Validation Plan — Unseen Bearings and Edge-Device Readiness

This plan separates two different research claims. They need different experiments and must not be treated as one result:

1. **Bearing generalisation:** can the model recognise faults on a physical bearing specimen that was not used during training, feature selection or model tuning?
2. **Edge-device readiness:** can the complete inference pipeline run within the latency, memory and energy limits of a named target device?

Neither claim is established by the Streamlit app running on a Windows laptop. The Streamlit app is a presentation demonstrator for the separately packaged 34-feature Paderborn Healthy/Damaged baseline.

## A. What the existing Paderborn experiment already gives us

The Paderborn University Bearing DataCenter documents 32 bearing experiments, four operating conditions and 20 measurements of four seconds for each bearing/condition setting. The project verification therefore expected 32 × 4 × 20 = 2,560 MAT recordings. The later V2 feature-building process produced 2,559 usable feature rows because one MAT file could not be read.

V2's report specifies a physical-bearing split:

| Partition | Bearings | Count |
|---|---|---:|
| Training | K001, K002, K003, K004, KA01, KA04, KA05, KA06, KA07, KA09, KA15, KA16, KA30, KI01, KI04, KI05, KI07, KI08, KI14, KI16, KI17, KB23 | 22 |
| Validation | K005, KA03, KA08, KI03, KB24 | 5 |
| Final test | K006, KA22, KB27, KI18, KI21 | 5 |

The V2 report states that all recordings from one physical bearing stay in one partition and describes the final five test bearings as excluded from model selection. It also says the next Stage 2 cross-validation step uses only the 27 development bearings (22 train + 5 validation) and leaves those five final test bearings aside.

**Audit before making a final claim:** the project contains multiple experimental tracks. The 34-feature packaged Logistic Regression baseline is separate from V2's 22-feature sequence models. The packaged baseline's reported 320-row evaluation involved four physical bearings and was reused during iterative experimentation. Do not transfer that caveat automatically to the V2 22/5/5 sequence split, or transfer the V2 report's untouched-test statement automatically to the packaged 34-feature baseline. Check notebook versions, experiment logs and saved prediction files for each track separately.

## B. The correct unseen-bearing evaluation

### 1. Keep groups by physical bearing

Use the bearing ID as the grouping unit. All measurements and derived windows/sequences from one bearing must remain together. Do not split individual windows or recordings randomly if they originate from the same physical bearing.

### 2. Use cross-validation only on development bearings

For V2 Stage 2, rebuild sequences with three aligned arrays/fields:

- `X`: sequence features
- `y`: fault label
- `bearing_id`: physical bearing that produced the sequence

Then create grouped folds on the 27 development bearings. For each fold:

1. Assign complete bearings to fold-training or fold-validation.
2. Fit the scaler and any data-driven feature selection using fold-training bearings only.
3. Transform fold-validation bearings using the fold-training scaler.
4. Create sequences without crossing bearing boundaries or bridging missing measurement indices.
5. Train and evaluate the model.
6. Record the held-out bearing IDs and class support with the metrics.

Do not fit a scaler once on the complete development table before cross-validation. Do not select features, tune thresholds or choose model settings using the final five test bearings.

### 3. Freeze the final model before testing

After choosing the model using only the development folds, freeze the feature list, preprocessing, architecture, weights, thresholds and inference code. Run the locked final test once. Save predictions and a manifest containing the model version, data-file IDs, bearing IDs, preprocessing version and timestamp.

### 4. Report more than accuracy

Report the following for each fold and for the final test:

- Accuracy and balanced accuracy.
- Macro-F1 and per-class precision/recall/F1.
- Confusion matrix and class support.
- Sensitivity and specificity for Healthy/Damaged Stage 1.
- For Stage 2, per-class results for Outer, Inner and Both, including which physical bearings contributed.
- Mean and variation across held-out bearings/folds; where practical, confidence intervals resampled at the **bearing** level, not by treating thousands of windows as independent specimens.

Include a per-bearing table. A model that performs well on many windows from one bearing but fails on another should not be hidden by pooled window accuracy.

## C. Current data limitation

The Paderborn inventory contains only 32 physical bearing IDs. Its four project categories contain 6 Healthy, 12 Outer, 11 Inner and 3 Both bearing IDs. Thus 2,559 feature rows are not 2,559 independent physical specimens.

The `Both` category is particularly data-limited: there are only three physical bearings in that category in the complete Paderborn inventory. No split can create five independent Both-bearing test specimens from only three available bearings. More windows from those same three bearings do not solve this limitation.

As a **planning target, not a universal statistical rule**, aim for at least five independent physical test bearings per class and preferably around ten or more where feasible. Also retain enough separate bearings for training. The exact sample-size requirement depends on the class variability and the precision desired for the final confidence intervals. The current Paderborn data alone cannot meet this target for Healthy and Both.

## D. External-dataset validation options

External validation means training and freezing the model on Paderborn, then testing it on a different dataset without fitting the scaler, selecting features or tuning thresholds on that external test set.

### Candidate 1 — University of Ottawa UORED-VAFCLS

The published dataset description reports 20 physical bearings, with healthy, developing-fault and faulty states for each bearing (60 sets total). It includes five bearings in each of four fault types (inner race, outer race, ball and cage), uses a 42 kHz sampling rate and provides 10-second recordings in MAT/CSV/Excel formats. This is a useful independent collection to investigate for an external Healthy/Damaged test.

Important caveats: the class taxonomy differs from Paderborn (it includes cage faults rather than the Paderborn Both label), and the sampling rate and sensor/setup differ. All states from one physical bearing must stay in the same evaluation group. Inspect the downloaded metadata for exact sensor units, speed, bearing geometry and file-to-bearing mapping before adapting any features.

Dataset: https://data.mendeley.com/datasets/y2px5tg92h/5

### Candidate 2 — HUST bearing

The published HUST dataset description reports 99 raw vibration signals, six defect configurations, five bearing types (6204–6208) and three working conditions at 51.2 kHz. It can provide an additional cross-dataset and cross-bearing-type stress test.

Important caveats: the bearing types and sampling rate differ from Paderborn. The current 34-feature baseline assumes the Paderborn 6203 geometry and nominal 64 kHz rate when calculating BPFO/BPFI features. Do not apply that 34-feature extractor unchanged. A geometry-aware and sampling-rate-aware adapter is required, and the external-test protocol must be frozen before examining results.

Dataset: https://data.mendeley.com/datasets/cbv7jyx4p9/3

### External test rules

- First map the labels without looking at model predictions. A binary experiment may map Healthy to Healthy and documented developing/faulty states to Damaged, while retaining the original sublabels for analysis.
- Do not silently equate UORED cage faults or HUST combined crack labels with Paderborn's Both class.
- Record differences in sample rate, sensor sensitivity/units, bearing dimensions, speed, load and acquisition setup.
- Keep the Paderborn-trained model and its scaler fixed for a true cross-dataset test. If retraining or refitting on the external dataset is required, label that as a separate dataset-specific experiment, not zero-shot external validation.
- Report failures and domain shift, not just the best metric.

## E. Edge-device readiness is a separate experiment

A laptop prediction time or a cloud-hosted Streamlit app does not establish Raspberry Pi performance. A virtual ARM/Raspberry Pi environment can check that software and dependencies run, but it cannot prove the latency, memory behaviour, thermal behaviour or energy consumption of a physical Raspberry Pi.

### 1. Name the target

Before benchmarking, specify the exact target (for example, Raspberry Pi 4 Model B with RAM size, OS, CPU architecture and Python/library versions). Do not report a generic 'edge latency' without naming the device and runtime.

### 2. Measure the whole pipeline

Measure separately:

- MAT/file reading and channel extraction.
- Feature extraction and signal preprocessing.
- Model-only prediction.
- Full end-to-end time from a ready input signal to output.
- Model artefact size and installed runtime size.
- Peak RAM / resident memory and CPU use.
- Temperature/thermal throttling during repeated runs.
- Energy per inference and average power, if a suitable power meter or device telemetry is available.

### 3. Use a repeatable benchmark

Use a fixed collection of representative recordings across healthy/damaged states and operating conditions. Warm up the pipeline, then run repeated predictions (for example, 100 repetitions per selected recording), separating cold-start model loading from warm inference. Report median (p50), p95 and maximum latency, memory, and the number/identity of files. For power claims, measure on the physical target; do not estimate energy from laptop CPU time.

### 4. If no hardware is available

- The current Windows laptop can be used for a development benchmark, clearly labelled as laptop results.
- An ARM virtual machine or QEMU can be used for software-compatibility checks, clearly labelled as emulation.
- A borrowed university/lab Raspberry Pi can be used for actual device latency and memory measurements.
- Without access to a physical target, make no claim about Raspberry Pi latency, energy or thermal performance. It is still valid to report model size, software compatibility and laptop inference time as preliminary evidence.

## F. Practical schedule

These are planning estimates, assuming the raw datasets are available and the notebooks run successfully. They are not guaranteed completion times.

| Work package | Approximate effort | Output |
|---|---:|---|
| Audit bearing IDs, labels, split files and experiment history | 0.5–1 day | Reproducible data/split manifest and a clear statement about which test sets were or were not used for model selection |
| Rebuild Stage 2 sequences with bearing IDs and run grouped development CV | 2–4 days | Fold-wise, bearing-wise results and rare-class support |
| Freeze and evaluate the final Paderborn split, after audit | 0.5–1 day | Locked final results and per-bearing predictions |
| Download/inspect UORED or HUST and build a compatible external adapter | 2–4 days | Parsed files, label map, sample-rate/geometry audit |
| Run and document external validation | 1–2 days | Cross-dataset results and domain-shift analysis |
| Edge software benchmark on laptop/emulator | 0.5–1 day | Preliminary software compatibility and laptop timing, not physical edge proof |
| Physical edge benchmark, if a device can be borrowed | 1–3 days | Device-specific latency/memory and possibly energy results |

A careful no-hardware research validation is likely around **one to two focused weeks** if data access and execution are straightforward. Stage 2 may take longer because Both has only three Paderborn physical bearings. A real physical-device benchmark adds time depending on access to a borrowed device.

## G. What we can honestly claim at each stage

| Evidence achieved | Defensible claim |
|---|---|
| Existing random/window split | Performance on the sampled recordings only |
| Bearing-grouped CV with fold-wise preprocessing | Internal generalisation estimate to held-out Paderborn bearings |
| One-time locked Paderborn test with verified audit trail | Performance on the specified held-out Paderborn bearings |
| Frozen Paderborn model tested on a compatible independent external dataset | Cross-dataset transfer result for that dataset/protocol |
| Laptop/emulator benchmark | Laptop speed or software-compatibility result |
| Repeated benchmark on a named physical edge device | Device-specific latency/memory result; add power meter/telemetry for energy claim |

**Bottom line:** there is no single experiment that proves universal real-world generalisation. Build evidence in layers, keep physical bearings separate, lock test sets before use, and measure hardware claims on the hardware being claimed.