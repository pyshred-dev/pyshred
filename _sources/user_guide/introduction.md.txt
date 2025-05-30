# Introduction to PySHRED

## What is PySHRED?

PySHRED is a Python package implementing the SHallow REcurrent Decoder (SHRED) architecture for sensing applications. PySHRED achieves state-of-the-art accuracy on full-state reconstructions and forecasting of spatiotemporal dynamics from very few sensors.

The three core models in PySHRED are the sequence, decoder, and latent forecaster models.
- Sequence model: learns a low-dimensional latent representation of temporal sensor measurements.
- Decoder model: learns a mapping from the latent representation to the high-dimensional state space.
- Latent forecaster model: learns the latent dynamics and forecasts future latent states

The sequence and decoder models work together to reconstruct the high-dimensional state space from temporal sensor measurements. The latent forecaster and decoder models work together to forecast future high-dimensional states without requiring additional sensor measurements.

PySHRED achieves amazing performance on most applications right out-of-the-box. If you want to build your own SHRED model optimized for your application, PySHRED allows you to seamlessly mix-and-match different built-in sequence, decoder, and latent forecaster models. Furthermore, you can customize any of the built-in models to better suite your application.

<!-- ![SHRED Architecture](/_static/main_figure.png) -->
```{figure} /_static/main_figure.png
:alt: SHRED architecture diagram
:name: fig:shred-architecture
:width: 100%

**Figure:** The SHRED architecture consists of a sequence model that encodes temporal sensor dynamics into a low-dimensional latent space, and a decoder that reconstructs the full-state space from that latent representation. Additionally, SHRED leverages a latent forecaster model to predict the future evolution of the latent space, which is decoded to reconstruct the future full-state spatiotemporal dynamics.
```

## When to use PySHRED?

PySHRED is a powerful tool for any application involving the reconstruction/forecasting of spatiotemporal dynamics from sensors measurements.

**Reconstruction:** Inferring the full-state dynamics of a system from sensor measurements.

**Forecasting:** Predicting future full-state dynamics of a system without using additional sensor measurements.

**Note:** SHRED is a supervised learning model, and training SHRED requires access to full-state data.

## Related Publications

PySHRED is built on cutting-edge research at the nexus of machine learning and dynamical systems. To learn more about the underlying research, methodology, and real-world applications, refer to the following publications:

- Williams, J. P., Zahn, O., & Kutz, J. N. (2024).  
  **Sensing with shallow recurrent decoder networks**.  
  *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, **480**(2298), 20240054.
  https://doi.org/10.1098/rspa.2024.0054


- Gao, M. L., Williams, J. P., & Kutz, J. N. (2025).  
  **Sparse identification of nonlinear dynamics and Koopman operators with Shallow Recurrent Decoder Networks**.  
  *arXiv preprint*, arXiv:2501.13329.  
  https://arxiv.org/abs/2501.13329


- Tomasetto, M., Williams, J. P., Braghin, F., Manzoni, A., & Kutz, J. N. (2025).  
  **Reduced Order Modeling with Shallow Recurrent Decoder Networks**.  
  *arXiv preprint*, arXiv:2502.10930.  
  https://arxiv.org/abs/2502.10930


- Ebers, M. R., Williams, J. P., Steele, K. M., & Kutz, J. N. (2024).  
  **Leveraging arbitrary mobile sensor trajectories with Shallow Recurrent Decoder Networks for full-state reconstruction**.  
  *IEEE Access*, **12**, 97428–97439.  
  https://doi.org/10.1109/ACCESS.2024.3423679