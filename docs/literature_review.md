# Literature Review: Lightweight and Reliable Bearing Fault Diagnosis

**Review date:** 16 November 2026  
**Scope:** Vibration-based rolling-bearing fault diagnosis, engineered signal features, compact ML/DL models, cross-operating-condition evaluation, data leakage, and edge deployment.

> This is a focused project literature review, not a claim that the listed works exhaust the field. Reported metrics are reproduced with their original task/protocol context. They must not be ranked directly when datasets, class definitions, splits, sample construction, or target-label access differ.

## 1. Executive synthesis

Recent bearing-diagnosis research has moved beyond maximizing benchmark accuracy. Four recurring questions are particularly relevant to this project:

1. **What is held out?** Segment-wise and condition-wise splits can allow the same physical bearing to contribute to both training and test sets. Bearing-wise separation is needed when the claim concerns generalisation to unseen bearings.
2. **Which representation is useful?** Time-domain statistics, frequency spectra, envelope spectra, wavelets, order-domain descriptors, and learned representations encode different physical information.
3. **How small is the deployed system?** Parameter count and model file size do not by themselves establish edge feasibility. End-to-end latency, memory, energy, feature-extraction cost, and measurements on the intended hardware matter.
4. **Does performance survive distribution shift?** A model that performs well under a random benchmark split may not remain reliable across speed, load, machine, sensor, or bearing changes.

The project's defensible position is an engineering-oriented study of an interpretable signal-processing pipeline, a compact classical baseline, comparative modelling, and an executable local/offline inference workflow. The work does not claim a new neural architecture or a physical Raspberry Pi deployment.

## 2. Selected research studies

### 2.1 Evaluation protocols and data leakage

#### 1. Vieira et al. (2026) — realistic evaluation and bearing-wise separation

**Reference:** João Paulo Vieira, Victor Afonso Bauler, Rodrigo Kobashikawa Rosa, and Danilo Silva, “Towards a more realistic evaluation of machine learning models for bearing fault diagnosis,” *Mechanical Systems and Signal Processing*, Volume 258, Article 114640 (15 August 2026).  
**DOI:** https://doi.org/10.1016/j.ymssp.2026.114640  
**Source:** https://github.com/gama-ufsc/bearing-data-leakage

- **Problem addressed:** Inflated test performance caused by data leakage in bearing-fault diagnosis.
- **Method/protocol:** Separates physical bearings between training and test; studies segment-wise and condition-wise partitioning; also proposes a multi-label formulation.
- **Datasets:** CWRU, Paderborn University, UORED-VAFCLS, and HUST bearing datasets.
- **Main findings:** Segment-wise and condition-wise strategies can produce spurious correlations. Performance can decline when test bearings are genuinely unseen. In two of the four datasets (CWRU and HUST), some classical feature-based methods outperformed the deep models considered.
- **Why it matters here:** This is the key reference for interpreting our CWRU experiments. Holding out a file or motor-load recording is not equivalent to holding out the physical bearing. Our selected CWRU subset contains one selected fault-condition series per fault class across loads; grouped-by-file results therefore cannot establish unseen-specimen generalisation.
- **Comparison caveat:** The paper studies multiple datasets and a bearing-wise protocol; its metrics should not be directly compared with our small, selected CWRU subset.

### 2.2 Lightweight models and edge deployment

#### 2. Hao Tian et al. (2026) — Frequency-Aware Lightweight CNN (FAL-CNN)

**Reference:** Hao Tian, Shitong Hua, Wenbiao Gui, Sijie Zou, and Chenglong Wang, “Frequency-Aware Lightweight Convolutional Neural Network for Edge-Enabled Intelligent Bearing Fault Diagnosis and Health Management,” *Sensors*, 26(18), 5861 (16 September 2026).  
**DOI:** https://doi.org/10.3390/s26185861

- **Method:** A fixed FFT/log-compression/resampling/normalization front end converts each 1,024-sample segment to a 128-dimensional spectrum. A frequency-aware block uses parallel 1D convolutions with kernel sizes 3, 7, and 15, followed by a depthwise-separable backbone.
- **Datasets/protocol:** CWRU with time-blocked splits, sample-isolation gaps and five random seeds; Paderborn with bearing-level nested validation.
- **Reported efficiency:** 65,122 parameters, 3.88 million MACs (7.76 million FLOPs), 288.8 KB ONNX model. Reported development-CPU latency was 0.17 ms (0.27 ms end-to-end); the authors explicitly state that physical edge-device profiling remains necessary.
- **Reported performance:** 99.78% mean accuracy over the CWRU four-load experiments. Under operating-load shift, FAL-CNN achieved 88.42% versus 75.15% for a parameter-matched plain CNN. On the Paderborn real-damage benchmark with unseen-bearing validation, reported accuracy was 59.86% ± 18.86%; the difference from the parameter-matched baseline was not statistically significant.
- **Why it matters here:** It is a close comparator for frequency-informed lightweight inference and, importantly, reports both saturated in-domain performance and more demanding shift/bearing-level results.
- **Comparison caveat:** Their signal length, labels, split protocol, architecture, and hardware context differ from our engineered-feature Logistic Regression baseline. Do not compare 99.78% directly with our results.

#### 3. Yifan Guo et al. (2026) — lightweight 1D-CNN-Transformer

**Reference:** Yifan Guo, Yijie Zhi, Renyi Qi, and Ming Cai, “A Lightweight 1D-CNN-Transformer for Bearing Fault Diagnosis Under Limited Data and AWGN Interference,” *Sensors*, 26(9), 2574 (22 April 2026).  
**DOI:** https://doi.org/10.3390/s26092574  
**Full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC13165830/

- **Method:** A compact 1D CNN extracts local signal patterns; a two-layer, four-head Transformer encoder models longer-range relationships; an optional squeeze-and-excitation module recalibrates channels.
- **Datasets/protocol:** CWRU and JNU; evaluates limited labelled samples, additive white Gaussian noise, and dynamic non-stationary interference. CWRU results are reported over ten random seeds.
- **Deployment evidence:** Reports Raspberry Pi 4B inference latency of 1.9828 ms (FP32, batch size 1); the paper also reports desktop reference-platform timings and INT8 results.
- **Why it matters here:** Provides a reference for compact learned sequence models and a concrete physical edge-device measurement.
- **Comparison caveat:** It is an end-to-end raw-signal CNN-Transformer; our baseline uses precomputed engineered features and Logistic Regression. Hardware, signal length, and protocols differ.

#### 4. “Deployment-Aware Comparison of Classical Machine Learning and 1D CNN for Bearing Fault Diagnosis on Embedded Hardware” (2026)

**Reference:** IEEE, *2026 27th International Carpathian Control Conference (ICCC)*, conference dates 1–3 June 2026; added to IEEE Xplore 7 July 2026.  
**DOI:** https://doi.org/10.1109/ICCC71363.2026.11593288

- **Method:** Compares feature-based linear SVM and XGBoost with an end-to-end 1D CNN.
- **Dataset:** CWRU.
- **Deployment protocol:** NVIDIA Jetson Orin Nano; compares CPU execution and TensorRT GPU inference with FP32, FP16, and INT8 precision.
- **Main conclusion:** Lightweight feature-based classifiers can be attractive for low-overhead deployment, while TensorRT-optimized CNN inference offers a favorable latency–accuracy trade-off under GPU acceleration.
- **Why it matters here:** Supports the decision to include classical ML as a serious edge-oriented baseline and to evaluate memory/latency rather than accuracy alone.
- **Comparison caveat:** This is an embedded-hardware comparison with a different target board and model implementations. We have not run equivalent measurements on our laptop replay or on a Raspberry Pi.

### 2.3 Signal processing, physics-informed representations, and operating-condition transfer

#### 5. Benjie Zhang, Wei Wang, and Yan He (2025) — signal processing plus deep learning and transfer learning

**Reference:** “A hybrid approach combining deep learning and signal processing for bearing fault diagnosis under imbalanced samples and multiple operating conditions,” *Scientific Reports*, 15, 13606 (19 April 2025).  
**DOI:** https://doi.org/10.1038/s41598-025-98138-1  
**Full text:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12009360/

- **Method:** GAN-based augmentation, wavelet time-frequency representation, an asymmetric convolutional network (MAC-MHA), multi-head attention, and transfer learning.
- **Datasets:** Paderborn and CWRU.
- **Protocol/details:** Paderborn measurements are described at 64 kHz, with repeated 4-second measurements under varying conditions. For the transfer experiment, the source model is fine-tuned using 10% of target-domain data.
- **Reported result:** The paper reports 97.52% accuracy after fine-tuning in the varying-condition transfer experiment.
- **Why it matters here:** Demonstrates a hybrid of signal processing, learning and transfer across operating conditions.
- **Comparison caveat:** The GAN augmentation, wavelet representation, deep architecture and target-data fine-tuning differ substantially from our fixed-feature classical baseline. The 97.52% figure is not directly comparable.

#### 6. PGDA-Net (2026) — physics-guided domain adaptation with order-domain envelope descriptors

**Reference:** “PGDA-Net: Physics-Guided Domain Adaptation for Cross-Condition Bearing Fault Diagnosis via Order-Domain Envelope Descriptors,” *Applied Sciences*, 16(21), 10802 (2026).  
**DOI:** https://doi.org/10.3390/app162110802

- **Method:** A differentiable physics path derives order-domain envelope descriptors using per-sample rotational speed and theoretical bearing fault orders. It combines raw-waveform representations with physics-guided features and uses physics-weighted distribution alignment, consistency constraints, pseudo-label filtering and order-centre regularization.
- **Datasets/protocol:** Paderborn multimodal vibration/current data with leave-one-condition-out evaluation; CWRU cross-load transfer. Paderborn uses recording-level partitioning before segmentation, with 4,096-sample windows and 50% overlap.
- **Why it matters here:** It is closely related to our BPFO/BPFI/BSF feature engineering and illustrates a more elaborate way to use fault kinematics under speed changes.
- **Comparison caveat:** PGDA-Net is a transductive unsupervised-domain-adaptation network using unlabeled target-domain information. Our current pipeline does not perform domain adaptation.

#### 7. Syed Haseeb Haider Zaidi et al. (2026) — physics-guided representation learning and selective adaptation

**Reference:** Syed Haseeb Haider Zaidi, Alex Shenfield, Hongwei Zhang, and Augustine Ikpehai, “Physics-Guided Representation Learning and Selective Domain Adaptation for Cross-Condition Bearing Fault Diagnosis,” *Machine Learning and Knowledge Extraction*, 8(11), 348 (23 October 2026).  
**DOI:** https://doi.org/10.3390/make8110348

- **Method:** Physics-Guided Domain Representation Network (PGDRN) combines temporal, spectral and physics-guided representations. A discrepancy-aware gate decides whether to adapt to an unlabeled target domain or retain the source model.
- **Dataset/protocol:** Paderborn; three operating-condition transfer tasks, with five independent runs per task.
- **Reported findings:** Gated deployment achieved a mean Macro-F1 gain of +0.0105 relative to a frozen model, reduced observed negative transfer from 33.3% under unconditional adaptation to 13.3%, and reduced average adaptation time by 26.3%. Reported overall Macro-F1 was 0.7763 for gated PGDRN, versus 0.7667 for Deep CORAL and 0.7706 for DANN.
- **Why it matters here:** Shows that adaptation itself should be evaluated as a deployment decision; more adaptation is not always better. It also reinforces the use of physical and spectral information.
- **Comparison caveat:** This is a cross-condition adaptation framework, not a compact fixed-feature Logistic Regression system.

#### 8. Kaisheng Deng and Ping Qu (2026) — frequency-domain augmentation and contrastive learning

**Reference:** “Bearing Single-Source Domain Generalization Fault Diagnosis Method Based on Adaptive Frequency-Domain Augmentation and Unsupervised Contrastive Learning,” *Sensors*, 26(17), 5349 (24 August 2026).  
**DOI:** https://doi.org/10.3390/s26175349

- **Method:** FDACL learns domain-invariant representations using adaptive frequency-domain augmentation and unsupervised contrastive learning. Learnable weights distinguish fault-sensitive spectral components from less informative bands.
- **Datasets:** CWRU, Paderborn University, and an industrial CRRC Qingdao Sifang railway wheelset bearing dataset.
- **Reported performance:** Average cross-speed accuracies of 92.68% on CWRU and 77.85% on Paderborn; the paper reports improvements of 4.23–8.71 percentage points over its comparison methods across transfer tasks.
- **Why it matters here:** Directly supports the importance of frequency-sensitive representations and evaluation under unknown operating conditions.
- **Comparison caveat:** It addresses single-source domain generalisation with a learned contrastive encoder; our feature-ablation results are on a much smaller selected CWRU subset.

#### 9. Dongliang Zhang and Hanming Tao (2024) — optimized variational mode extraction and 1D multi-scale residual CNN

**Reference:** “Bearing Fault Diagnosis Based on Parameter-Optimized Variational Mode Extraction and an Improved One-Dimensional Convolutional Neural Network,” *Applied Sciences*, 14(8), 3289 (13 April 2024).  
**DOI:** https://doi.org/10.3390/app14083289

- **Method:** Uses the Archimedes Optimization Algorithm to optimize Variational Mode Extraction (VME) parameters, followed by a one-dimensional multi-scale residual CNN (1D-MRCNN).
- **Dataset:** The paper reports CWRU drive-end data sampled at 12 kHz, with healthy and inner-race, outer-race, and ball fault states at multiple defect diameters.
- **Why it matters here:** Provides a signal-decomposition plus learned-classifier comparator, relevant to vibration preprocessing and feature representation.
- **Comparison caveat:** The selected CWRU subset and task setup differ from our 48 kHz four-class engineered-feature evaluation; do not transfer its reported results without checking the exact split and class definition.

### 2.4 Reviews and broader research directions

#### 10. Rohan Puntambekar et al. (2026) — survey of ML and DL methods

**Reference:** Rohan Puntambekar, Pratyaksh Vyas, Ankit Thakkar, and Dhaval Patel, “A survey of machine learning and deep learning methods for vibration-based Bearing fault diagnosis: The need, challenges, and potential future research directions,” *Neurocomputing*, 659, 131628 (1 January 2026).  
**DOI:** https://doi.org/10.1016/j.neucom.2025.131628

- **Coverage:** Traditional ML (KNN, SVM, ANN, ELM and trees), signal processing, CNNs, RNNs, GRUs, autoencoders, GANs, GNNs, deep reinforcement learning, attention, Transformers, transfer learning and explainability.
- **Key challenges identified:** Data imbalance, transferability across domains, real-time constraints and incipient-fault detection.
- **Use in this project:** Background taxonomy and explanation of why classical engineered-feature models remain relevant alongside deep architectures.

#### 11. Zhiqiang Bao et al. (2026) — theory-to-industry review

**Reference:** Zhiqiang Bao, Changfu Liu, Hui Yang, Jiayao Zhang, and Yuqi Li, “From theory to industry: A survey of deep learning-enabled bearing fault diagnosis in complex environments,” *Engineering Applications of Artificial Intelligence*, 163, 113068 (1 January 2026).  
**DOI:** https://doi.org/10.1016/j.engappai.2025.113068

- **Coverage:** Organizes deep-learning bearing diagnosis around data, models, applications, challenges and prospects.
- **Relevant themes:** Digital-twin-enhanced small-sample learning, physically constrained interpretable models, and edge-cloud collaborative architectures.
- **Use in this project:** Supports the distinction between a high-scoring laboratory model and a complete industrial diagnostic system.

#### 12. Li Cui et al. (2026) — signal-processing priors and trustworthy deployment review

**Reference:** Li Cui, Xingyu Yan, Xin Wang, Ning Li, Xishuo Yao, and Peter K. Liaw, “Deep Learning for Vibration-Based Rolling Bearing Fault Diagnosis: From Signal-Processing Priors to Trustworthy Industrial Deployment,” *Measurement: Digitalization*, Article 100063 (available online 16 October 2026).  
**DOI:** https://doi.org/10.1016/j.meadig.2026.100063

- **Coverage:** Wavelets, variational mode decomposition, cyclostationary processing, CNNs, recurrent networks, graph models, generative/adversarial methods and Transformers.
- **Key message:** No architecture is universally best; physically meaningful signal priors, uncertainty, data scarcity, domain shift and trustworthy deployment should guide model choice.
- **Use in this project:** Strong support for explaining why signal-processing features and a simple baseline are reasonable engineering choices.

#### 13. Rihab Marhrane et al. (2026) — systematic review of vibration-based bearing diagnosis

**Reference:** Rihab Marhrane, Faouaz Jeffali, Jamal Bouchnaif, et al., “Vibration-based fault diagnosis of rolling bearings in rotating machinery: a systematic review of signal processing, machine learning, deep learning, and transfer learning approaches,” *Journal of Intelligent Manufacturing* (23 October 2026).  
**DOI:** https://doi.org/10.1007/s10845-026-03011-1

- **Coverage:** PRISMA 2020 systematic review; 197 studies retained for qualitative synthesis, covering signal processing, classical ML, deep learning and transfer learning.
- **Key challenges:** Domain shift, limited fault data, computational requirements, interpretability and insufficient field validation.
- **Use in this project:** Broad source for the research context and the gap between laboratory benchmark results and industrial deployment.

## 3. Cross-paper comparison

| Research direction | Representative papers | Common strength | Important limitation or open issue |
|---|---|---|---|
| Leakage-aware evaluation | Vieira et al. (2026); Tian et al. (2026) | Explicitly studies split construction and unseen-bearing evaluation | Requires enough independent bearing specimens; small datasets produce high uncertainty |
| Lightweight edge CNNs | Tian et al. (2026); Guo et al. (2026) | Reduces model size and reports latency/complexity | Results depend on hardware, preprocessing, signal length and test protocol |
| Hybrid signal processing + DL | Zhang et al. (2025); Zhang & Tao (2024) | Injects structured signal information before or within learned models | Pipeline complexity and protocol differences complicate comparison |
| Physics-guided transfer | PGDA-Net (2026); Zaidi et al. (2026) | Uses bearing physics to improve cross-condition representations/adaptation | Often needs target-domain information, accurate speed/geometry, and more complex training |
| Frequency-domain generalisation | Deng & Qu (2026) | Attempts to retain fault-related spectral structure under condition shifts | Transfer accuracy varies by dataset; still a learned deep model |
| Classical ML and embedded trade-offs | IEEE ICCC (2026); Vieira et al. (2026) | Recognizes the efficiency and value of feature-based methods | Accuracy alone is not enough; device profiling and strict splits remain essential |
| Reviews | Puntambekar et al.; Bao et al.; Cui et al.; Marhrane et al. | Maps techniques and identifies system-level challenges | Survey results are not independent benchmark experiments |

## 4. How the literature relates to this project

### 4.1 What is aligned

- **Signal-processing priors:** The project explicitly engineers time-domain, frequency-domain and bearing-characteristic envelope features rather than relying only on learned raw-signal representations.
- **Compact baseline:** Logistic Regression with feature scaling is a low-complexity reference point. The literature supports retaining classical feature-based models as meaningful baselines rather than assuming deep learning must always win.
- **Operating-condition evaluation:** Paderborn and CWRU provide complementary laboratory datasets for studying vibration diagnosis under different conditions.
- **Inference workflow:** The saved preprocessing/classification pipeline and offline replay make the path from recorded signal to prediction inspectable.

### 4.2 What is not yet demonstrated

- Physical edge-device latency, memory and energy measurements.
- Live sensor acquisition or guaranteed real-time operation.
- Cross-machine or cross-dataset transfer by the deployed Paderborn classifier.
- Physical-bearing-independent generalisation for the selected CWRU subset.
- A newly proposed or state-of-the-art neural architecture.

### 4.3 Defensible project positioning

The contribution should be framed as an **integrated, lightweight, signal-processing-based diagnostic prototype with comparative experiments and an offline inference/replay workflow**. The work evaluates how engineered time, frequency and envelope representations behave with compact classifiers, documents cross-condition and recording-grouped results, and makes the constraints of the selected datasets explicit.

It should not be framed as:
- a new state-of-the-art model;
- proof of generalisation to arbitrary unseen bearings;
- evidence of physical Raspberry Pi deployment;
- a direct performance win over papers using different tasks or splits.

## 5. Research gaps and possible future work

1. **Bearing-wise validation:** Increase the number of independent physical bearings in each class and make component identity the grouping variable across all splits.
2. **Cross-dataset generalisation:** Train on Paderborn and test on an independently prepared CWRU task, or vice versa, after aligning label definitions and handling sensor/sampling differences.
3. **Deployment profiling:** Measure the complete pipeline (signal buffering, feature extraction, scaling, inference and output) on a specified edge device. Record latency distributions, peak RAM, model storage and energy.
4. **Feature cost analysis:** Compare not only feature count but also extraction time and memory for time-, frequency- and envelope-domain families.
5. **Uncertainty and abstention:** Calibrate probabilities and allow an uncertain/unknown output when the input differs from the training distribution.
6. **Reproducibility:** Record exact bearing IDs, source files, sample ranges, random seeds, software versions and split manifests.

## 6. References and source links

1. Vieira et al. (2026), https://doi.org/10.1016/j.ymssp.2026.114640
2. Tian et al. (2026), https://doi.org/10.3390/s26185861
3. Guo et al. (2026), https://doi.org/10.3390/s26092574
4. Zhang et al. (2025), https://doi.org/10.1038/s41598-025-98138-1
5. PGDA-Net (2026), https://doi.org/10.3390/app162110802
6. Zaidi et al. (2026), https://doi.org/10.3390/make8110348
7. Deng & Qu (2026), https://doi.org/10.3390/s26175349
8. Zhang & Tao (2024), https://doi.org/10.3390/app14083289
9. IEEE ICCC (2026), https://doi.org/10.1109/ICCC71363.2026.11593288
10. Puntambekar et al. (2026), https://doi.org/10.1016/j.neucom.2025.131628
11. Bao et al. (2026), https://doi.org/10.1016/j.engappai.2025.113068
12. Cui et al. (2026), https://doi.org/10.1016/j.meadig.2026.100063
13. Marhrane et al. (2026), https://doi.org/10.1007/s10845-026-03011-1
