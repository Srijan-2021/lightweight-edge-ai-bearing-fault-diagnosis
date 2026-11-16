# Run the project locally on Windows

This guide covers the public Streamlit presentation demo and the research notebooks. The research notebooks and Module 2–4 notes have been merged into `main`. Download the main repository version.

## 0. Get the project files

Beginner route: open the [research branch on GitHub](https://github.com/Srijan-2021/lightweight-edge-ai-bearing-fault-diagnosis), select **Code → Download ZIP**, and extract the ZIP to a folder such as `Documents\lightweight-edge-ai-bearing-fault-diagnosis`.

Alternative, if Git for Windows is installed, open PowerShell in the parent folder where you want the project and run:

```powershell
git clone https://github.com/Srijan-2021/lightweight-edge-ai-bearing-fault-diagnosis.git
cd lightweight-edge-ai-bearing-fault-diagnosis
```

All commands below should be run from the extracted/cloned project root (the folder containing `streamlit_app.py`, `requirements.txt` and `notebooks`).

## 1. Check Python

The repository's app and research requirements are pinned for Python 3.11. Open PowerShell in the project root and run:

```powershell
py -3.11 --version
```

If this prints a Python 3.11 version, continue. If Windows says that `py` is not recognised or cannot find Python 3.11, install Python 3.11 first and reopen PowerShell.

## 2. Run the Streamlit app locally

The Streamlit app has its own small environment. It does not need the raw dataset just to open the home, methodology and results pages. A Paderborn MAT recording is needed when testing the interactive diagnosis page.

Create and install the environment:

```powershell
py -3.11 -m venv .venv-streamlit
.\.venv-streamlit\Scripts\python.exe -m pip install --upgrade pip
.\.venv-streamlit\Scripts\python.exe -m pip install -r requirements.txt
```

Start the app:

```powershell
.\.venv-streamlit\Scripts\python.exe -m streamlit run streamlit_app.py
```

Streamlit will print a local address, normally `http://localhost:8501`. Open it in your browser. Keep the PowerShell window open while using the app. Press `Ctrl+C` in that window to stop it.

To test a diagnosis, choose **Run diagnosis** and upload a valid Paderborn `.mat` file containing the named `vibration_1` and `speed` channels. The recording must contain at least 64,000 vibration samples. Raw MAT files are not included in Git.


## 2A. Confirm the local app is working

A successful local smoke check should show:

- The Overview page loads.
- The Run diagnosis page accepts a Paderborn `.mat` file.
- A valid recording with `vibration_1` and `speed` channels produces a Healthy/Damaged output, class probabilities, processing-time display, waveform and spectrum.
- The Experimental results and Methodology pages load.

A Windows local run was manually checked on 16 November 2026. The example recording `N09_M07_F10_K001_1.mat` produced a Healthy prediction with 99.80% displayed Healthy probability and 0.20% displayed Damaged probability. This confirms that the demo pipeline executed for that recording; it is not a new independent accuracy evaluation, proof of unseen-bearing generalisation, or an edge-device performance benchmark.

## 3. Run the learning notebooks (Modules 1–4)

For the first four notebooks, use the smaller core environment. It installs NumPy, Pandas, SciPy, Matplotlib and JupyterLab, and avoids installing TensorFlow before you need the teacher-model notebook.

Create the environment from the project root:

```powershell
py -3.11 -m venv .venv-notebooks
.\.venv-notebooks\Scripts\python.exe -m pip install --upgrade pip
.\.venv-notebooks\Scripts\python.exe -m pip install -r requirements-notebooks-core.txt
.\.venv-notebooks\Scripts\python.exe -m ipykernel install --user --name bearing-notebooks --display-name "Python 3.11 (Bearing Notebooks)"
```

### Add the original Paderborn files

Raw measurement files are deliberately not in this GitHub repository. Create this local folder layout:

```text
project-root/
├── data/
│   ├── raw/
│   │   ├── K001/
│   │   │   └── *.mat
│   │   ├── K002/
│   │   │   └── *.mat
│   │   ├── KA01/
│   │   │   └── *.mat
│   │   └── ... other bearing folders ...
│   └── processed/
├── notebooks/
├── streamlit_app.py
└── requirements-notebooks-core.txt
```

Put the bearing folders directly inside `data/raw/` (for example, `data/raw/K001/`), not inside an extra nested folder. Use the original provider's data and follow its usage conditions.

### Open JupyterLab

From the project root, run:

```powershell
cd notebooks
..\.venv-notebooks\Scripts\python.exe -m jupyter lab
```

JupyterLab opens in your browser. Open and run the notebooks in this order:

1. `01_dataset_verification.ipynb`
2. `02_metadata_preparation.ipynb`
3. `03_feature_extraction.ipynb`
4. `04_feature_analysis.ipynb`

Choose the kernel **Python 3.11 (Bearing Notebooks)** if Jupyter asks. Run a cell with `Shift+Enter`. To run the whole notebook, use JupyterLab's **Run → Run All Cells**. Start with notebook 01 and complete it before moving to notebook 02.

Notebook 02 creates `data/processed/bearing_metadata.csv`, which notebook 03 expects. Notebook 03 also expects the original Paderborn bearing directories under `data/raw/`. If those files are absent, the first data-loading cell will not run.

Start Jupyter from the `notebooks` folder because the notebooks use relative paths such as `../data/raw` and `../data/processed`.


### Current setup checkpoint (16 November 2026)

The user has successfully installed `requirements-notebooks-core.txt` into `.venv-notebooks`. The next step is to register the kernel and launch JupyterLab:

```powershell
.\.venv-notebooks\Scripts\python.exe -m ipykernel install --user --name bearing-notebooks --display-name "Python 3.11 (Bearing Notebooks)"
cd notebooks
..\.venv-notebooks\Scripts\python.exe -m jupyter lab
```

The current Streamlit virtual environment was created at `C:\.venv-streamlit` (outside the project folder). To restart the local app from the project root on a later day, run:

```powershell
C:\.venv-streamlit\Scripts\python.exe -m streamlit run streamlit_app.py
```

## 4. Later research notebooks and model training

`requirements-research.txt` is the fuller research environment and includes TensorFlow, scikit-learn, JupyterLab and the scientific Python packages. Create a separate environment when you are ready to run the later classical/deep-learning notebooks:

```powershell
py -3.11 -m venv .venv-research
.\.venv-research\Scripts\python.exe -m pip install --upgrade pip
.\.venv-research\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv-research\Scripts\python.exe -m ipykernel install --user --name bearing-research --display-name "Python 3.11 (Full Bearing Research)"
```

Then open JupyterLab from the `notebooks` folder using:

```powershell
..\.venv-research\Scripts\python.exe -m jupyter lab
```

Choose the **Python 3.11 (Full Bearing Research)** kernel for the advanced notebooks. This full environment is larger because it includes TensorFlow. You do not need it just to learn Modules 1–4.

## 5. If PowerShell blocks environment activation

You do not have to run `Activate.ps1` at all. The commands in this guide call the virtual environment's Python executable directly, so they avoid the PowerShell execution-policy activation problem.

## 6. Common errors

| Message | What to check |
|---|---|
| `py is not recognized` | Install Python 3.11 and reopen PowerShell. |
| `No module named streamlit` | Install `requirements.txt` using `.venv-streamlit`'s Python executable. |
| `No module named scipy` | Install `requirements-notebooks-core.txt` into `.venv-notebooks`. |
| `StopIteration` or no sample file found | Check that `data/raw/K001/` exists and contains `.mat` files. |
| `bearing_metadata.csv` not found | Run notebook 02 successfully before running notebook 03's batch extraction cells. |
| `Expected 2560 .mat files` | Check that all expected bearing folders and MAT files are directly under `data/raw/`. |
| MATLAB channel not found | Check that the file is a Paderborn MAT recording with the expected named `vibration_1` channel. |

## 7. Keep the two environments separate

- `.venv-streamlit`: runs the presentation app.
- `.venv-notebooks`: runs the early learning and feature-extraction notebooks.
- `.venv-research`: runs the heavier later research/model-training notebooks.

These environments are local folders and should not be committed to Git. The raw dataset and generated processed CSV files also remain local.