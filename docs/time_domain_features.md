# Module 4 — Time-Domain Features: Mean, Standard Deviation and RMS

This guide accompanies [Notebook 03 — Vibration Feature Extraction](../notebooks/03_feature_extraction.ipynb). It explains the first three statistics calculated by `extract_vibration_features()`.

## 1. Why calculate features?

A vibration recording contains hundreds of thousands of individual samples. A classical machine-learning pipeline can work with a much shorter description of that recording. Feature extraction calculates descriptive numerical values from the signal.

The initial Paderborn feature-extraction notebook calculates 26 values. This module begins with three simple time-domain statistics.

## 2. Mean — average signal level

For samples `x₁, x₂, …, xₙ`:

`mean = (x₁ + x₂ + … + xₙ) / n`

The mean is the ordinary average. It describes the average level or offset of the recorded values.

Python implementation:

```python
mean_value = np.mean(signal)
```

A mean close to zero means positive and negative samples average out around zero. It does **not** mean that the signal has no vibration or that its amplitude is small.

## 3. Standard deviation — spread around the mean

Standard deviation describes how spread out the readings are around their average:

`std = sqrt(mean((x − mean)²))`

- Small standard deviation: readings tend to stay closer to the mean.
- Larger standard deviation: readings are more spread out around the mean.

Notebook implementation:

```python
std_value = np.std(signal)
```

NumPy's default is `ddof=0`, so this code calculates the population standard deviation. It does not request the sample-standard-deviation variant (`ddof=1`).

## 4. RMS — overall magnitude

RMS means Root Mean Square:

`RMS = sqrt(mean(x²))`

Steps:
1. Square each sample.
2. Calculate the average of the squared values.
3. Take the square root.

Notebook implementation:

```python
rms_value = np.sqrt(np.mean(signal ** 2))
```

Squaring means that negative readings also contribute positively. RMS describes the overall signal magnitude relative to zero. Unlike standard deviation, it includes any non-zero average offset.

## 5. A small numerical example

For the toy signal `[1, 2, 3]`:

| Calculation | Result |
|---|---:|
| Mean | 2.0000 |
| Population standard deviation | 0.8165 |
| RMS | 2.1602 |

The calculations are:

- Mean: `(1 + 2 + 3) / 3 = 2`
- Standard deviation: `sqrt(((1−2)² + (2−2)² + (3−2)²) / 3) = sqrt(2/3)`
- RMS: `sqrt((1² + 2² + 3²) / 3) = sqrt(14/3)`

These are not interchangeable. Mean describes the average level, standard deviation describes spread around the mean, and RMS describes magnitude relative to zero.

## 6. Values from the actual Paderborn example

The original Notebook 03 example run for `N09_M07_F10_K001_1.mat` reported:

| Feature | Value |
|---|---:|
| Mean | -0.015454 |
| Standard deviation | 0.367295 |
| RMS | 0.367620 |

Because the mean is close to zero, RMS and standard deviation are close in this recording. They still describe different quantities. These values are for one example recording in its recorded units; they are not universal healthy/damaged thresholds.

## 7. Why was RMS later removed?

The V2 project report describes a feature-reduction step after correlation and redundancy analysis. Four of the original 26 features were removed:

- RMS
- Spectral RMS
- Peak-to-peak
- Spectral entropy

That left 22 retained features for the V2 sequence-model experiments. This is not because RMS is mathematically wrong or useless in every bearing problem. It was removed from this project's fixed 22-feature representation as part of the project's redundancy-reduction decision.

The separate packaged 34-feature binary baseline also uses a 22-feature statistical/spectral subset plus 12 BPFO/BPFI envelope-energy features. Its feature schema excludes RMS and peak-to-peak, so the exact feature order must be preserved at inference.

## 8. Viva-ready explanation

> "We extracted time-domain statistics from each vibration recording. Mean describes its average level, standard deviation describes how much the samples vary around the mean, and RMS describes the overall signal magnitude. In Python, we used `np.mean`, `np.std` and the square root of the mean squared signal. We initially included RMS in the 26-feature set, but it was later removed with three other redundant features to form the 22-feature V2 representation."

The values are descriptive features, not standalone fault diagnoses.