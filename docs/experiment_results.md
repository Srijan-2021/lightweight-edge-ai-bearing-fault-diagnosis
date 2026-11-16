# Experimental Results Summary

**Status:** Consolidated from the project experiment notes and the reported notebook outputs. This document records the evaluated protocols and the limits of interpretation; it is not an independent reproduction.

## 1. Paderborn primary track

### 1.1 Compact binary baseline

- **Task:** Healthy vs. Damaged.
- **Features:** 34 engineered features (22 general features plus BPFO/BPFI envelope features).
- **Model:** StandardScaler + Logistic Regression with class_weight="balanced", max_iter=2000 and random_state=42.
- **Held-out evaluation size:** 320 windows.
- **Confusion matrix (Healthy, Damaged):** [[80, 0], [3, 237]].
- **Accuracy:** 99.0625%.
- **Balanced accuracy:** 99.375%.
- **F1-score:** 99.371%.
- **ROC AUC:** 1.00.

The fitted model pipeline file was approximately 3.25 KiB in the project environment, excluding the Python runtime and dependencies.

**Qualification:** The evaluation involved four physical bearings, and the test subset was reused during iterative project experimentation. It must not be described as a pristine untouched external test.

### 1.2 Five-recording sequence experiments

The Paderborn sequence experiments used five recordings per sequence, each representing an independent approximately four-second measurement. They are not a continuous raw 64 kHz time series. Adjacent sequences overlap four of five recording positions and are correlated.

For the binary Stage 1 sequence experiment, the split was bearing-disjoint:
- Training: 22 bearings, 1,408 sequences.
- Validation: 5 bearings, 318 sequences.
- Test: 5 bearings, 320 sequences.

Reported Stage 1 results:

| Model | Validation accuracy | Test accuracy |
|---|---:|---:|
| LSTM | 92.14% | 100% |
| GRU | 100% | 100% |
| Transformer | 99.06% | 94.375% |
| Compact 1D CNN | 100% | 100% |

The compact Stage 1 CNN had 7,713 parameters.

**Qualification:** These results are retrospective project experiments. The test data had been seen during earlier project experiments, so these figures are not an untouched final external estimate.

### 1.3 Stage 2 fault-type sequence experiments

Stage 2 distinguishes Outer, Inner and Both fault conditions. The Both category had only one physical bearing in each of the training, validation and test partitions.

| Model | Test accuracy | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|---:|
| LSTM | 43.75% | 33.33% | 28.34% |
| Transformer | 45.70% | 44.53% | 35.52% |
| Compact 1D CNN | 53.52% | 50.26% | 41.57% |

The Stage 2 CNN had 7,139 parameters. The Both class was not predicted correctly by the evaluated Stage 2 sequence models.

**Interpretation:** The limited physical-bearing support, especially for the Both category, is a major constraint. These multiclass results should not be conflated with the binary baseline metrics.

## 2. CWRU supplementary track

### 2.1 Selected subset and feature table

- 16 MAT recordings: four selected condition series (Healthy, Inner, Ball, Outer) under 0, 1, 2 and 3 HP.
- Sampling rate: 48 kHz, drive-end channel.
- Windowing: 1-second, non-overlapping windows.
- Total: 140 windows, 35 per class.
- Feature table: 37 columns, comprising 9 time-domain, 13 frequency-domain and 15 envelope-domain features.

### 2.2 LOLO baseline and ablation

The four-fold leave-one-motor-load-out (LOLO) Logistic Regression baseline using all 37 features achieved 100% in all four held-out motor-load folds.

Pooled feature ablation results:

| Configuration | Features | Mean fold accuracy | Pooled accuracy | Pooled macro-F1 |
|---|---:|---:|---:|---:|
| Time | 9 | 96.25% | 97.86% | 97.85% |
| Frequency | 13 | 100% | 100% | 100% |
| Envelope | 15 | 100% | 100% | 100% |
| Time + Frequency | 22 | 100% | 100% | 100% |
| Time + Envelope | 24 | 100% | 100% | 100% |
| Frequency + Envelope | 28 | 100% | 100% | 100% |
| All 37 Features | 37 | 100% | 100% | 100% |

Time-only accuracy on the held-out 0 HP fold was 85%; its other three folds achieved 100%. The 13-feature Frequency-only and 15-feature Envelope-only configurations matched the full 37-feature configuration on this selected evaluation.

The mean fold metric and pooled metric differ because the four held-out folds contain unequal numbers of windows (20 in the 0 HP fold and 40 in each of the other folds).

### 2.3 Recording-grouped cross-validation and gap audit

A four-fold StratifiedGroupKFold split used MAT filename as the group. Each fold held out four complete recordings and trained on twelve; there was no filename overlap between training and test.

Frequency-only, Envelope-only and all 37 features each produced 100% training and 100% test accuracy in all four folds, with a zero observed accuracy gap.

**Critical qualification:** This establishes recording-file separation, not independent physical-bearing separation. The four load recordings for each class share one filename-derived condition key (Normal, IR007, B007 or OR007@6). Other load recordings from the same selected condition series can therefore be represented in training when one load recording is tested. The selected CWRU subset does not establish generalisation to entirely unseen physical bearing specimens.

## 3. What can be concluded

1. The Paderborn compact binary baseline is a small, executable inference model with strong results on the project's selected evaluation, but the test subset was reused during iterative experiments.
2. The Paderborn sequence experiments show that a small CNN can perform well on the binary task, while the limited support for the multiclass Both category produces weak Stage 2 generalisation.
3. On the selected CWRU subset, frequency-only and envelope-only features were sufficient to achieve the observed perfect scores under LOLO and recording-grouped splits.
4. No conventional train–test performance gap was observed in the audited CWRU folds.
5. Neither the CWRU recording-grouped evaluation nor the selected Paderborn test results prove universal or industrial physical-bearing generalisation.
6. The inference/replay workflow is an offline software simulation. No physical edge device has been benchmarked.

Use this page together with [Limitations and Interpretation](limitations.md) when presenting the results.
