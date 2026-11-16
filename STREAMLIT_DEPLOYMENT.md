# Streamlit presentation demo

This repository includes a public-presentation Streamlit application. The application demonstrates the separately packaged Paderborn 34-feature binary Logistic Regression baseline.

## Run locally

Use Python 3.11, then run:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The app expects a Paderborn MATLAB recording containing the named `vibration_1` and `speed` channels. It uses the project's nominal 64 kHz assumption and requires at least 64,000 vibration samples.

## Deploy to Streamlit Community Cloud

After this branch is merged into `main`:

1. Sign in at https://share.streamlit.io/ using the GitHub account that can access this repository.
2. Create an app and select this repository.
3. Select the `main` branch and `streamlit_app.py` as the app entrypoint.
4. In Advanced settings, select Python 3.11.
5. Deploy and save the public `*.streamlit.app` URL.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## What is demonstrated

- Upload a Paderborn `.mat` recording.
- Extract 22 general statistical/spectral features and 12 BPFO/BPFI envelope-energy features.
- Run the saved StandardScaler + Logistic Regression model.
- Display the Healthy/Damaged prediction, class probabilities, waveform, spectrum and feature vector.
- Review selected Paderborn V2 and CWRU experiments with their limitations.

## Important scope distinction

The interactive predictor is the standalone Paderborn 34-feature binary baseline. It is **not** the V2 sequence-model Stage 1/Stage 2 system. Hosting the Streamlit app in the cloud is for presentation and demonstration; it is **not** a physical edge-device deployment, Raspberry Pi benchmark, live sensor stream or real-time validation.

The model artefact is stored as a small Base64 text file in this repository and decoded to a temporary file when the app loads. Uploaded MAT files are written to temporary files for parsing and deleted afterwards. Raw Paderborn/CWRU datasets are not included.
