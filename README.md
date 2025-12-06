# Meta1A: Bias Detection Project

This repository contains two main components:
- A training pipeline for bias detection model
- A Streamlit dashboard for real-time bias detection

## Project Structure
 
```
Meta1A/
├── dashboard/     # Streamlit application
├── train/         # Model training pipeline
```

## Setup and Installation

We recommend using `uv` as the package installer for better performance and dependency management.

### Install uv

```bash
pip install uv
```

## Dashboard (`/dashboard`)

The dashboard provides a web interface for real-time bias detection.

### Setup
```bash
cd dashboard
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Run
```bash
streamlit run run.py
```

The dashboard will be available at `http://localhost:8501`

## Training Pipeline (`/train`)

The training pipeline consists of two main steps:

### 1. Data Processing (`/train/data_processing`)
- Uses `create_dataset.ipynb` to generate the training dataset
- Clones and processes data from RedditBias and another repository
- Combines and cleans the data
- Outputs `all_data.csv` in the `model_training` directory

#### Setup
```bash
cd train
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

To run the notebook:
1. Open `data_processing/create_dataset.ipynb`
2. Install the kernel:
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name meta1a-train-venv --display-name "Python (train/.venv)"
```
3. Select the kernel "Python (train/.venv)" in VS Code
4. Run all cells to generate the dataset

### 2. Model Training (`/train/model_training`)
- Uses `train.ipynb` to train the bias detection model
- Trains on the combined dataset from `all_data.csv`
- Saves the trained model locally
- Optionally pushes the model to HuggingFace

To train the model:
1. Open `model_training/train.ipynb`
2. Use the same kernel as above ("Python (train/.venv)")
3. Run all cells to:
   - Load and preprocess the data
   - Train the model
   - Evaluate performance
   - Save the model locally
4. Optional: Set up HuggingFace credentials to push the model
   - Create a `.env` file in the `train` directory
   - Add your HuggingFace token: `HF_TOKEN=your_token_here`
   - Run the last cell to upload the model

## Note

If `all_data.csv` is not present in model_training directory, make sure to run the data processing notebook before the training notebook, as the training depends on the generated `all_data.csv` file.
