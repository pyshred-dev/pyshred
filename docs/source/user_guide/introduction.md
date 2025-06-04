# Introduction to PySHRED

## What is PySHRED?

PySHRED is a Python package implementing the SHallow REcurrent Decoder (SHRED) architecture for sensing applications. PySHRED achieves state-of-the-art accuracy on full-state reconstructions and forecasting of spatiotemporal dynamics from very few sensors.

The three core components of PySHRED are the sequence, decoder, and latent forecaster models.

- **Sequence model:** a neural architecture that learns the temporal dependencies in sensor measurements and projects them into a low-dimensional latent representation.
- **Decoder model:** a neural architecture that learns a mapping between the low-dimensional latent representation and the high-dimensional state space.
- **Latent forecaster model:** a model that takes in the current latent states and predicts the future latent states.

The sequence and decoder models work together to reconstruct the high-dimensional state space from sensor measurements. The latent forecaster and decoder models work together to forecast high-dimensional state space dynamics without needing additional sensor measurements.

PySHRED achieves amazing performance on most applications straight out-of-the-box. If you want to build your own SHRED model optimized for your application, PySHRED allows you to seamlessly mix-and-match different built-in sequence, decoder, and latent forecaster models. Furthermore, you can customize the architecture of each specific sequence/decoder/latent forecaster model to best fit your specific application.

<!-- ![SHRED Architecture](/_static/main_figure.png) -->

```{figure} /_static/main_figure.png
:alt: SHRED architecture diagram
:name: fig:shred-architecture
:width: 100%

**Figure:** The SHRED architecture includes a sequence model that encodes temporal sensor dynamics into a low-dimensional latent space, and a decoder that reconstructs the full-state space from that latent representation. Additionally, SHRED includes a latent forecaster model to predict the future evolution of the latent space, which is decoded to reconstruct the future full-state spatiotemporal dynamics.
```

## When to use PySHRED?

PySHRED is a powerful tool for any application involving the reconstruction/forecasting of spatiotemporal dynamics from timeseries sensors measurements.

- **Reconstruction** is a sensing task involving the inference of full-state system dynamics from sensor measurements.

- **Forecasting** is a sensing task involving the inference of future full-state system dynamics without using additional sensor measurements.

**Note:** PySHRED is a supervised learning model, and requires access to full-state data during training.

## Related Publications

PySHRED is built on cutting-edge research at the nexus of machine learning and dynamical systems. To learn more about the underlying research, methodology, and real-world applications, refer to the following publications:

- Williams, J. P., Zahn, O., & Kutz, J. N. (2024).  
  **[Sensing with shallow recurrent decoder networks](https://doi.org/10.1098/rspa.2024.0054)**.  
  _Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences_, **480**(2298), 20240054.

- Gao, M. L., Williams, J. P., & Kutz, J. N. (2025).  
  **[Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks](https://arxiv.org/abs/2501.13329)**.  
  _arXiv preprint_, arXiv:2501.13329.

- Tomasetto, M., Williams, J. P., Braghin, F., Manzoni, A., & Kutz, J. N. (2025).  
  **[Reduced Order Modeling with Shallow Recurrent Decoder Networks](https://arxiv.org/abs/2502.10930)**.  
  _arXiv preprint_, arXiv:2502.10930.

- Kutz, J. N., Reza, M., Faraji, F., & Knoll, A. (2024).  
  **[Shallow Recurrent Decoder for Reduced Order Modeling of Plasma Dynamics](https://arxiv.org/abs/2405.11955)**.  
  _arXiv preprint_, arXiv:2405.11955.

- Ebers, M. R., Williams, J. P., Steele, K. M., & Kutz, J. N. (2024).  
  **[Leveraging arbitrary mobile sensor trajectories with Shallow Recurrent Decoder Networks for full-state reconstruction](https://doi.org/10.1109/ACCESS.2024.3423679)**.  
  _IEEE Access_, **12**, 97428–97439.
