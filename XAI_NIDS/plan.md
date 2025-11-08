# Infersec Project Plan and Benchmarking

## 1. Project Goal

To develop and benchmark an end-to-end, explainable AI framework for network intrusion detection. This project involves consolidating existing scripts, benchmarking various machine learning models on multiple datasets, and integrating XAI techniques to interpret their decisions. The final output will be a comprehensive and interactive Google Colab notebook that allows for easy experimentation and visualization of results.

## 2. Recommendations

- **Code Consolidation:** The current project structure has many redundant scripts. I recommend refactoring the code to create a single, configurable pipeline. This will make the project easier to maintain, extend, and run. Instead of having separate scripts for each model and dataset (e.g., `RF_final.py`, `DNN_all_final.py`), we should create a modular system where the dataset, model, and features can be selected as parameters. The code in the `Framework` folder is a good starting point for this.
- **Environment:** For the security analysis part of the project, using two separate VMs (Kali and Ubuntu) in VMware with a host-only network is the most reliable setup. It avoids the networking complexities of WSL2 and provides a more realistic environment.
- **Interactive Notebook:** The plan to create an interactive Google Colab notebook is excellent. It will make the project accessible and easy to use for other researchers. We should use widgets like `ipywidgets` to create dropdowns for selecting datasets, models, and features.

## 3. Environment Setup

### 3.1. Local Environment (for data preparation and analysis)

1.  **Virtualization:**
    *   Install VMware Workstation or Player.
    *   Create two virtual machines: one for Kali Linux and one for Ubuntu.
    *   Configure a "Host-Only" or "NAT" network in VMware to allow communication between the two VMs.

2.  **Python Setup:**
    *   On your development machine (or the Ubuntu VM), install Python 3.8 or higher.
    *   Create a virtual environment: `python -m venv venv`
    *   Activate the environment: `source venv/bin/activate` (on Linux) or `venv\Scripts\activate` (on Windows).
    *   Install the required packages: `pip install -r requirements.txt`.

### 3.2. Google Colab Setup (for training)

1.  **Google Drive:**
    *   Create a folder for the Infersec project in your Google Drive.
    *   Upload the dataset zip files to this folder.
2.  **Colab Notebook:**
    *   In your Colab notebook, the first step will be to mount your Google Drive:
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```
    *   The notebook should then have cells to:
        *   Unzip the datasets.
        *   Install dependencies from `requirements.txt`.
        *   Run the training and evaluation pipeline.

## 4. Project Phases and Tasks

### Phase 1: Code Consolidation and Refactoring

- [ ] **Analyze Existing Scripts:** Review all scripts in the `CICIDS-2017`, `NSL-KDD`, and `RoEduNet-SIMARGL2021` folders to identify common patterns and functions.
- [ ] **Create Unified Data Module:** Develop a `data_loader.py` module that contains functions for:
    - [ ] Loading each of the three datasets.
    - [ ] Applying the necessary preprocessing steps (as mentioned in the README, e.g., fixing labels in CICIDS-2017).
    - [ ] Performing feature selection (all vs. top 15).
    - [ ] Handling data balancing and normalization.
- [ ] **Create Unified Model Module:** Develop a `model_trainer.py` module that contains:
    - [ ] A function that takes a model name (e.g., "RandomForest", "DNN") and hyperparameters as input and returns an untrained model instance.
    - [ ] A generic `train_and_evaluate` function that takes the data and model, performs training, and returns the evaluation metrics and confusion matrix.
- [ ] **Refactor Main Script:** Create a main script `main.py` (or a main Colab notebook) that uses the new modules to run experiments. This script will replace the need for the numerous standalone scripts.

### Phase 2: Data Preparation

- [ ] **Download Datasets:** Download the zip files for CICIDS-2017, NSL-KDD, and RoEduNet-SIMARGL2021 from the links in the `README.md`.
- [ ] **Place Datasets:** Place the downloaded zip files in a `datasets` directory within the project.
- [ ] **Run Preprocessing:** Use the `data_loader.py` module to process each dataset and save the cleaned `X_train`, `X_test`, `y_train`, `y_test` files. This only needs to be done once.

### Phase 3: Model Training and Benchmarking (on Colab)

- [ ] **Setup Colab Notebook:** Create a new Colab notebook and implement the setup steps from section 3.2.
- [ ] **Run Experiments:** Using the refactored code, write a loop in the Colab notebook to iterate through all models and datasets.
- [ ] **Log Results:** For each experiment, save the evaluation metrics.
- [ ] **Fill Benchmarking Table:** Populate the benchmarking table below with the results from your experiments.

### Phase 4: Explainable AI (XAI) Integration

- [ ] **Integrate XAI into the Pipeline:** Add a step in the `train_and_evaluate` function to generate explanations after a model is trained.
- [ ] **Generate Global Explanations:** For each model, generate and save the SHAP summary and beeswarm plots.
- [ ] **Generate Local Explanations:** Select a few interesting samples from each dataset (e.g., a normal sample, a specific attack type) and generate local explanations using both SHAP and LIME. Save these explanation plots.

### Phase 5: Interactive Colab Notebook

- [ ] **Design UI:** Plan the layout of the interactive notebook. It should be clean and user-friendly.
- [ ] **Implement Widgets:** Use `ipywidgets` to create dropdowns, buttons, and sliders for:
    - [ ] Selecting the dataset.
    - [ ] Selecting the model.
    - [ ] Choosing between all features or top 15 features.
    - [ ] Triggering the training and evaluation.
    - [ ] Selecting a sample for local explanations.
- [ ] **Display Results:** Create functions to neatly display the confusion matrix, metrics table, and XAI plots in the notebook.

## 5. Benchmarking Strategy

### Models to be Benchmarked

- AdaBoost
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- Random Forest
- Deep Neural Network (DNN)
- LightGBM
- Support Vector Machine (SVM)
- XGBoost

### Datasets

- CICIDS-2017
- NSL-KDD
- RoEduNet-SIMARGL2021

### Performance Metrics

We will use the following metrics to evaluate the performance of each model on each dataset. The results will be recorded in the table below.

| Dataset      | Model          | Features | Accuracy | Precision | Recall | F1-score | MCC    | BACC   | AUCROC |
|--------------|----------------|----------|----------|-----------|--------|----------|--------|--------|--------|
| **CICIDS-2017** | AdaBoost       | All      |          |           |        |          |        |        |        |
|              | AdaBoost       | Top 15   |          |           |        |          |        |        |        |
|              | KNN            | All      |          |           |        |          |        |        |        |
|              | KNN            | Top 15   |          |           |        |          |        |        |        |
|              | ...            | ...      |          |           |        |          |        |        |        |
| **NSL-KDD**    | AdaBoost       | All      |          |           |        |          |        |        |        |
|              | AdaBoost       | Top 15   |          |           |        |          |        |        |        |
|              | ...            | ...      |          |           |        |          |        |        |        |
| **RoEduNet**   | AdaBoost       | All      |          |           |        |          |        |        |        |
|              | AdaBoost       | Top 15   |          |           |        |          |        |        |        |
|              | ...            | ...      |          |           |        |          |        |        |        |


## 6. Experiment Log

Use the following template to log the details of each experiment.

---

**Experiment ID:** 001

**Date:** YYYY-MM-DD

**Dataset:** CICIDS-2017

**Model:** Random Forest

**Parameters:**
- Features: Top 15
- `n_estimators`: 200
- `max_depth`: 15

**Results:**
- Accuracy: 0.99
- F1-score: 0.99
- Link to Colab notebook: [link]

**Notes:**
- Initial test of the consolidated pipeline.

---