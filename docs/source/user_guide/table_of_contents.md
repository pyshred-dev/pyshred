# Table of Contents

### 1. Introduction:
* What is PySHRED?
    - Overview of SHallow REcurrent Decoder architrcture
    - Key features: lightweight, modular, sequence + decoder separation
* When to use it
    - Full-state reconstruction, forecasting from sparse sensors

### 2. Installation 
* Python version, required packages
* `pip install pyshred` or from GitHub
* Brief note on GPU vs CPU usage

### 3. Overview of Core Objects, pros/cons, benefits, explanation, no code/depth (high-level)
* DataManager
* SHRED
    - Sequence Model
        - GRU
        - LSTM
        - Transformers
    - Decoder Model
        - MLP
        - UNET
    - Latent Forecaster Model
        - SINDy
        - LSTM
* SHREDEngine

### 4. Quickstart Example
* Jupyter Notebook just for Quickstart (convert to .md)

### 5. Data Prep
- Format Data
- Initialize/configure data manager
- Add data and sensors to data manager
- Analyze data manager attributes
- train/val/test split

### 6. SHRED Training
- Initialize latent forecaster
- Initialize SHRED
- Fit SHRED
- Evaluate SHRED

### 7. Downstream Tasks
- Initialize SHREDEngine
- Sensor measurements to latent space
- Forecast latent space
- Decode latent space to full-state space
- Evaluate against ground truth