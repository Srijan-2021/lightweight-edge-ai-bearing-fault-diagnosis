# Module 3 — Understanding Vibration Signals

This guide accompanies [Notebook 03 — Vibration Feature Extraction](../notebooks/03_feature_extraction.ipynb). It explains the raw signal before introducing detailed feature mathematics.

## 1. What is a vibration signal?

While the test rig is operating, the vibration sensor records a changing physical motion. A computer does not receive a picture or a verbal description such as “the bearing is damaged”. It receives numerical readings:

`x = [x₀, x₁, x₂, x₃, …]`

Each number is one sensor sample. The value is the vibration amplitude recorded at that instant.

The `vibration_1` channel from the Paderborn MATLAB structure is the primary input used by the feature-extraction notebook. In the example file, the notebook inspected seven named Y channels: `force`, `phase_current_1`, `phase_current_2`, `speed`, `temp_2_bearing_module`, `torque`, and `vibration_1`. The example `vibration_1` data array had 256,823 samples.

## 2. How to read a waveform

A waveform is a graph of the signal:

- Horizontal axis: time.
- Vertical axis: vibration amplitude in the units stored in the source file.
- Each plotted point corresponds to a sampled reading.

The amplitude can be positive or negative because the signal varies around a reference level. The current notebook does not state a physical unit for the `vibration_1` values, so describe it as recorded amplitude rather than assuming it is acceleration in `m/s²`.

A larger excursion means a larger recorded value. It may be useful when describing a signal, but a peak alone does not prove that a bearing is damaged or tell us which fault is present.

A real bearing vibration signal is a mixture of different motions, machine effects and noise. It therefore need not look like a clean repeating sine wave.

## 3. Sampling rate and time

The Paderborn feature-extraction and packaged inference code use a nominal sampling rate:

`FS = 64000`

That means 64,000 readings are assumed per second. The nominal time gap between adjacent samples is:

`Δt = 1 / 64000 = 0.000015625 seconds = 15.625 microseconds`

Given a sample index `n`, the nominal time location is `t = n / FS`. Notebook 03 uses this idea to plot the first few milliseconds of the extracted recording.

The example `N09_M07_F10_K001_1.mat` contains 256,823 vibration samples. At the nominal 64 kHz rate this gives:

`256823 / 64000 ≈ 4.013 seconds`

This is an estimate based on the nominal rate. The packaged inference code explicitly does not correct the recording timestamps or resample the signal; do not claim that timestamp irregularities have been physically corrected.

## 4. What is frequency?

Frequency is how many cycles of a repeating pattern occur in one second. Its unit is Hertz (Hz).

- 1 Hz means one cycle per second.
- 100 Hz means one hundred cycles per second.
- 1,000 Hz is 1 kHz.

If more cycles fit into the same time interval, the oscillation is faster and has a higher frequency.

## 5. Time view versus frequency view

The two plots answer different questions:

| View | Horizontal axis | What it helps us see |
|---|---|---|
| Time waveform | Time | How the measured amplitude changes over the recording |
| Frequency spectrum | Frequency (Hz) | Which oscillation rates have stronger components |

Notebook 03 uses the real-valued Fast Fourier Transform (`np.fft.rfft`) to move from the sampled time signal to frequency components. It then calculates the corresponding frequency coordinates with `np.fft.rfftfreq`.

The feature function finds the strongest non-zero frequency bin and calls it `dominant_frequency`. In the original example notebook output, that feature was approximately 7,847.8 Hz. This is the strongest component reported by that particular FFT calculation; it is **not automatically a BPFO or BPFI fault frequency** and should not be presented as a diagnosis by itself.

## 6. What your example signal looked like numerically

For the original notebook example file `N09_M07_F10_K001_1.mat`, the recorded output was:

| Signal statistic | Value |
|---|---:|
| Samples | 256,823 |
| Minimum | -4.8798 |
| Maximum | 4.6906 |
| Mean | -0.01545 |
| Standard deviation | 0.3673 |
| Dominant frequency feature | 7,847.8 Hz |

The amplitude values are expressed in the recorded signal's units; the notebook output does not establish a physical unit.

## 7. Why do we extract features?

A raw signal contains many thousands of measurements. Instead of supplying only a large array to a classical machine-learning model, this project calculates a smaller set of descriptive numbers.

The initial notebook extracts 26 general time-domain and frequency-domain features. Later V2 experiments removed four redundant features and retained 22 for sequence modelling. A separate packaged Paderborn binary baseline uses 34 inputs, adding BPFO/BPFI envelope-energy features. These are separate pipelines.

The next module will explain how statistics such as mean, standard deviation, RMS and peak-to-peak describe a waveform, and how the FFT creates the frequency-domain features.