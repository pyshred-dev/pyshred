# PySHRED

PySHRED is a Python library that implements the **SH**allow **RE**current **D**ecoder (SHRED) architecture, providing a high-level interface for accurate sensor-based reconstruction and forecasting of spatiotemporal systems.

![SHRED architecture](https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/main_figure.png)

The three core components of PySHRED are the sequence, decoder, and latent forecaster models.

- **Sequence model:** a neural architecture that learns the temporal dependencies in sensor measurements and projects them into a low-dimensional latent representation.
- **Decoder model:** a neural architecture that learns a mapping between the low-dimensional latent representation and the high-dimensional state space.
- **Latent forecaster model:** a model that takes in the current latent states and predicts the future latent states.

The sequence and decoder models work together to reconstruct the high-dimensional state space from sensor measurements. The latent forecaster and decoder models work together to forecast high-dimensional state-space dynamics without needing additional sensor measurements.

The models in PySHRED works great out-of-the-box, but you can also mix-and-match or fully customize sequence, decoder, and latent forecaster models to suit your application.

## Documentation:

Online documentation is available at [pyshred-dev.github.io/pyshred/](https://pyshred-dev.github.io/pyshred/).

The docs include a [tutorial](https://pyshred-dev.github.io/pyshred/user_guide/tutorial_bunny_hill.html), [user guide](https://pyshred-dev.github.io/pyshred/user_guide/index.html), [example gallery](https://pyshred-dev.github.io/pyshred/examples/index.html), [API reference](https://pyshred-dev.github.io/pyshred/pyshred/modules.html), and other useful information.

## Installation

- **Installing from PyPI**

  The latest stable release (and required dependencies) can be installed from PyPI:

  ```
  pip install pyshred
  ```

- **Installing from source**

  PySHRED can be installed via source code on GitHub.

  ```
  git clone https://github.com/pyshred-dev/pyshred.git
  cd pyshred
  pip install .
  ```

## Citing

_Citation instructions coming soon._

## Resources

- Docs: [https://pyshred-dev.github.io/pyshred/](https://pyshred-dev.github.io/pyshred/)
- Issue Tracking: [https://github.com/pyshred-dev/pyshred/issues](https://github.com/pyshred-dev/pyshred/issues)
- Source code: [https://github.com/pyshred-dev/pyshred](https://github.com/pyshred-dev/pyshred)

## Main Contributors
<a href="https://github.com/pyshred-dev/pyshred/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pyshred-dev/pyshred" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## References

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
