# Downstream Tasks
First initialize SHREDEngine using the data manager and shred model. Then perform any of the following three downstream tasks:
* encode sensor measurements to latent space
* forecast future latent space state
* decode latent space back to full-state space
* evaluate reconstructions from sensor measurements against ground truth

## Initialize SHREDEngine


## Encode sensor measurements to latent space
```{figure} /_static/sensor_to_latent_figure.png
:alt: Sensor to Latent Pipeline
:name: fig:sensor-to-latent-pipeline
:width: 100%

**Figure:** The `sensor_to_latent` method takes in raw sensor measurements, scales it, generates lagged sequences of the scaled sensor measurements, then passes it through SHRED's `sequence_model` to obtain the latent space associated with the raw sensor measurements.

Note: the lagged sequences near the start are padded by zeros because there is not enough sensor measurements to look back on.
```

## Forecast future latent space states
```{figure} /_static/forecast_latent_figure.png
:alt: Latent Space Forecasting Pipeline
:name: fig:latent-forecasting-pipeline
:width: 100%

**Figure:** The `forecast_latent` method takes in a forecast horizon (number of steps to forecast into the future) and an initial latent space seed. The timesteps requried for the seed depends on SHRED's `latent_forecaster` model selected. `SINDy_forecaster` requires only a single latent space timestep. `LSTM_forecaster` requires the latent space seed to be of length lags (lags set in `LSTM_forecaster`). The seed is passed into the SHRED's `latent_forecaster`, which then forecasts the latent space `h` timesteps into the future.
```

## Decode latent space back to full-state space
```{figure} /_static/decoder_figure.png
:alt: Latent Space to Full-State Space Pipeline
:name: fig:latent-to-full-state-pipeline
:width: 100%

**Figure:** The `decoder` method takes in a latent space, passes it through SHRED's `decoder_model`, and returns the full-state reconstruction of the latent space..
```

## Evaluate reconstructions from sensor measurements against ground truth