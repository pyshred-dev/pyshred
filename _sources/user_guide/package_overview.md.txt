# Package Overview

## Core Modules:
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

## DataManager
The `DataManager` gets the data ready for SHRED.

## SHRED
`SHRED` is the brains of PySHRED. It learns the mapping between sensor trajectories and the full-state space. `SHRED` takes in three key models:
* Sequence Model
* Decoder Model
* Latent Forecaster Model

### Sequence Model
The sequence model takes in sensor trajectories, and outputs a rich time-embedded latent space.

Tables of pros and cons of each sequence model available.

### Decoder
The decoder model takes in the rich time-embedded latent space outputted by the sequence model, and outputs the full-state reconstruction.

Tables of pros and cons of each decoder model available.

### Latent Forecaster Model
The latent forecaster model takes in an initial latent space seed, and forecasts the latent space into the future.

Table of pros and cons of each latent forecaster model.

## SHREDEngine
This is where the magic happens. After all that hard work preparing the data and training SHRED, `SHREDEngine` takes in both the `DataManager` and the `SHRED` model to perform all downstram tasks. The `SHREDEngine` supports three core tasks:
1. Mapping raw sensor measurements to the rich time-embedded latent space
2. Forecasting the latent space into the future starting with an initial latent space seed
3. Decoding the latent space back to the full-state space