# SHRED Training
- Initialize latent forecaster
- Initialize SHRED
- Fit SHRED
- Evaluate SHRED

## Initialize SHRED
The SHRED model takes in three seperate models:
- sequence model: responsible for encoding the temporal dynamics of sensor measurement trajectories, producing the latent space. The sequence models take in the lagged sensor measurements as input and produces a rich latent space encoding the temporal dynamics of sensor measurements.
- decoder model: responsible for mapping the latent space outputted by the sequence model to the high-dimensional full-state space.
- latent forecaster model: responsible for forecasting future latent space states. The latent forecaster models are used to forecast the latent space into the future. There models are needed to forecast the full-state space into the future when there are no new sensor measurements available. If set to `None`, you will not be able to forecast the latent space as a downstream task. 

Currently there are two latent forecaster models:
- `LSTM_Forecaster`: trains an LSTM neural network on one-step-ahead forecasting in the latent space. This model performs performs rolling forecasts (uses model predictions are new inputs to predict further into the future).
- `SINDy_Forecaster`: fits a **S**parse **I**dentification of **N**on-linear **DY**namics model on the latent space. SINDy discovers the set of dynamical equations governing the latent space. The number of governing equations discovered will be equal to the size of the latent space. This forecaster is ideal for stable long-term forecasts.

**NOTE:** Currently, latent forecasters only support using MLP (Shallow Decoder Network) as the decoder model. 

### Quick Initialization (Presets)
Using preset models, you can initialize each of the three models with any of the strings below.
Sequence model choices: `"LSTM"`, `"GRU"`, `"Transformer"`
Decoder model choices: `"MLP"`, `"UNET"`


Latent forecaster model choices: `"SINDy_Forecaster"`, `"LSTM_Forecaster"`

### Custom Initialization

#### Sequence Models

| Sequence Models | Description | Code Sample |
|-----------------|-------------|-------------|
|GRU              |Gated recurrent unit is the lightest and fastest sequence model to train.| <pre><code>from pyshred import GRU<br>seq_model = GRU(hidden_size=3,<br>                num_layers=1)</code></pre>|
|LSTM             |Long short-term memory is a step up in complexity and training time compared to GRU.|<pre><code>from pyshred import LSTM<br>seq_model = LSTM(hidden_size=64,<br>                 num_layers=2)</code></pre>|
|Transformer | Transformer is by far the heaviest and slowered sequence model to train. |<pre><code>from pyshred import TRANSFORMER<br>seq_model = TRANSFORMER(d_model=128,<br>                        nhead=16,<br>                        dropout=0.2)</code></pre>|

#### Decoder Models
| Decoder Models | Description | Code Sample |
|-----------------|-------------|-------------|
|MLP              |**S**hallow decoder network is a lightweight multilayer perceptron.| <pre><code>from pyshred import MLP<br>seq_model = GRU(hidden_size=3,<br>                num_layers=1)</code></pre>|
|LSTM             |Long short-term memory is a step up in complexity and training time compared to GRU.|<pre><code>from pyshred import LSTM<br>seq_model = LSTM(hidden_size=64,<br>                 num_layers=2)</code></pre>|
|Transformer | Transformer is by far the heaviest and slowered sequence model to train. |<pre><code>from pyshred import TRANSFORMER<br>seq_model = TRANSFORMER(d_model=128,<br>                        nhead=16,<br>                        dropout=0.2)</code></pre>|
#### Latent Forecaster Models

### Common Pairings
GRU MLP SINDy
LSTM MLP LSTM
Transformer UNet None

