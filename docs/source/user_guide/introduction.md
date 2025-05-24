# Introduction to PySHRED

## What is PySHRED?

PySHRED is a Python package for the SHallow REcurrent Decoder (SHRED) neural network architecture. PySHRED achieves state-of-the-art performance on sensing tasks. Particularly, PySHRED achieves state-of-the-art accuracy on full-state reconstruction and forecasts from sparse sensors.

SHRED leverages sequence neural networks to learn a latent representation of the temporal dynamics of sensor measurement trajectories, and decoder neural networks to learn a mapping between this latent representation and the high-dimensional state space. To perform forecasting, SHRED leverage PySINDy and other sequence neural networks.

![SHRED Architecture](/_static/main_figure.png)

## When to use PySHRED?
PySHRED is for reconstructing the full-state space from a few sensors. The sensors feed timeseries sensor measurements into PySHRED, which then reconstructs the full-state space. PySHRED is also capable of forecasting the full-state space into the future, without additional sensor measurements.

Please note into order to traub PySHRED, you must have the full-state space dynamics for PySHRED to train on.

