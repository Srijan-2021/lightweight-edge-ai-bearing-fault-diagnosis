# Paderborn Dataset and Metadata — Module 2

This note accompanies [Notebook 01](../notebooks/01_dataset_verification.ipynb) and [Notebook 02](../notebooks/02_metadata_preparation.ipynb).

## 1. How the dataset is organised

The project organises the primary Paderborn data by physical bearing. The 32 bearing IDs are grouped as:

| Category | Bearing count |
|---|---:|
| Healthy | 6 |
| Outer-race damage | 12 |
| Inner-race damage | 11 |
| Both (combined) damage | 3 |
| **Total** | **32** |

Each bearing is expected to have measurements from four operating-condition identifiers. Notebook 02 lists those identifiers explicitly:

- `N15_M07_F10`
- `N09_M07_F10`
- `N15_M01_F10`
- `N15_M07_F04`

The notebook expects 20 recordings for each bearing-condition combination. Thus the expected file count is 32 × 4 × 20 = 2,560 MAT files.

## 2. What the filename tells us

Example:

`N15_M07_F10_KA01_12.mat`

| Part | Project interpretation |
|---|---|
| `N15_M07_F10` | Operating-condition identifier |
| `KA01` | Physical bearing ID |
| `12` | Measurement number |
| `.mat` | MATLAB data file |

The filename is useful because the measurement must remain linked to the physical bearing from which it came. A bearing's identity is not an ordinary feature supplied to the classifier; it is metadata used to organise, audit and split the experiment.

## 3. What is stored inside the MAT file?

The original Paderborn files contain nested MATLAB structures and multiple named measurement channels. The reusable inference module reads the MAT structure, inspects its channel list, finds the channel named `vibration_1`, and extracts that channel's numerical `Data`. It also retrieves the `speed` channel.

The inference code uses the mean measured speed as RPM for its bearing-frequency calculations. Its nominal sampling-rate constant is 64,000 samples per second (64 kHz). The module explicitly preserves the original assumption: it does not correct the timestamp sequence or resample a recording.

## 4. Fault labels in Notebook 02

The notebook creates two target-label views from the bearing ID:

| Fault location | Numeric `fault_class` | Binary `binary_label` |
|---|---:|---:|
| Healthy | 0 | 0 |
| Outer | 1 | 1 |
| Inner | 2 | 1 |
| Both | 3 | 1 |

The four-class label preserves the detailed bearing condition. The binary label combines all damaged categories into one class. This supports the project's two questions:

- Stage 1: Is the bearing Healthy or Damaged?
- Stage 2: For damaged bearings, is the fault Outer, Inner or Both?

## 5. The unreadable file

During the later feature-table construction, `N15_M01_F10_KA08_2.mat` produced a matrix-reading error. The file was recorded in a failure log, and processing continued with the other files. The V2 project report records 2,559 feature rows from the expected 2,560 recordings. This 2,559 value is the later processing result, not an output of the folder-verification notebook alone.

## 6. Why this matters for data leakage

Imagine that 20 files belong to bearing `KA01`. Those are 20 measurements, but they do not represent 20 different physical bearings. They share the same physical source.

When evaluating generalisation to unseen bearings, all measurements from `KA01` must remain in only one split. Metadata preparation preserves the bearing ID so Notebook 05 can enforce that rule.

## 7. Project source and implementation

- Folder and file verification: `notebooks/01_dataset_verification.ipynb`
- Bearing labels, operating-condition checks and metadata CSV: `notebooks/02_metadata_preparation.ipynb`
- Named MATLAB-channel extraction and 64 kHz inference assumption: `src/paderborn_inference.py`
- Processed metadata output: `data/processed/bearing_metadata.csv`

Raw vibration recordings are intentionally not included in the public repository.