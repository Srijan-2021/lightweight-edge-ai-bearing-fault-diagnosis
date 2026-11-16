# Limitations and Interpretation

This page records the limitations that must accompany the project's reported results. It is intended to prevent benchmark numbers from being interpreted more broadly than the experiments support.

## 1. Paderborn primary pipeline

- The reported binary Logistic Regression result is based on the project's current selected evaluation. The held-out subset involved a small number of physical bearings and was reused during iterative project experimentation.
- The reported result should not be described as a pristine, untouched external test or as proof of universal bearing-level generalisation.
- Feature parity between the offline inference implementation and the reference computation establishes computational consistency for the tested cases; it does not by itself validate all physical timing assumptions in every recording.
- Signal timestamp diagnostics identified intervals with deviations from nominal sampling intervals in a substantial subset of recordings. The code should not silently resample or reinterpret those recordings without a justified physical-time correction.

## 2. CWRU supplementary subset

The current selected subset contains 16 recordings:

- Normal_0 through Normal_3;
- IR007_0 through IR007_3;
- B007_0 through B007_3;
- OR007@6_0 through OR007@6_3.

These represent four selected condition series measured at four motor loads. The filename-derived condition keys are Normal, IR007, B007, and OR007@6, one key per class.

### What the evaluations establish

- **LOLO:** tests whether the model classifies the selected subset at a motor load excluded from training.
- **Recording-grouped CV:** prevents windows from the same MAT recording appearing in both training and test sets.
- **Feature ablation:** shows which feature families were sufficient for the selected class/load subset under the tested protocol.

### What they do not establish

- Recording independence is not necessarily physical-bearing independence.
- Different load files sharing one condition key can represent the same seeded fault-condition series under different operating loads.
- The selected CWRU data do not establish generalisation to new physical specimens, other defect severities, other bearing types, other machines or field data.
- Zero exact duplicate feature vectors does not rule out correlated measurements or specimen-specific signatures.
- 100% accuracy on this small, selected subset is not evidence that the model will achieve 100% in an industrial setting.

The CWRU results are therefore reported as supplementary, selected-subset results.

## 3. Training/test gap and overfitting

The audited Logistic Regression experiments showed 100% training and test accuracy (zero observed accuracy gap) across the four LOLO folds. The three configurations also achieved 100% test accuracy in recording-grouped cross-validation.

This means conventional train–test overfitting was not observed under these specific splits. It does **not** prove that leakage/dependence or specimen-specific signatures are absent. The grouping variable was the MAT filename, not a verified independent physical-bearing serial number.

## 4. Edge-deployment scope

The project contains a local saved inference pipeline and offline replay of recorded signals.

It does not currently demonstrate:
- physical Raspberry Pi or Jetson deployment;
- a live sensor stream;
- measured edge-device latency or throughput;
- peak RAM, energy use or power consumption;
- hard real-time guarantees;
- end-to-end latency including acquisition and feature extraction on target hardware.

Model file size alone is not an edge-performance benchmark. Future work should measure the complete pipeline on a clearly specified target device.

## 5. Comparison with literature

Reported results from different papers are not directly comparable unless the following align:

- dataset and exact subset;
- bearing identities and class definitions;
- fault severity and sensor location;
- sampling rate and signal window;
- train/validation/test split;
- overlap and leakage controls;
- target-label access (supervised, unsupervised or transductive);
- metric definition and aggregation;
- hardware and inference measurement conditions.

The literature review records these differences where the source provides the information.

## 6. Appropriate wording

Prefer:

> The model achieved the reported performance on the selected Paderborn/CWRU evaluation protocol.

> The recording-grouped CWRU evaluation held out complete MAT recordings, but it did not establish physical-bearing-independent generalisation.

> The saved inference pipeline was exercised through offline replay of recorded signals.

Avoid:

- "100% accurate in real-world bearing diagnosis";
- "generalizes to all unseen bearings";
- "deployed on Raspberry Pi";
- "real-time edge inference" without a corresponding hardware experiment.

## 7. Future validation

1. Obtain additional physical bearing specimens per class and severity.
2. Build a specimen-identity manifest and split by physical bearing before segmentation.
3. Use nested validation for model selection and keep a final test set untouched.
4. Evaluate cross-dataset transfer with harmonized labels and signal preprocessing.
5. Measure feature extraction plus inference on a specified edge device.
6. Record latency distributions, memory, energy, model storage and uncertainty behaviour.
