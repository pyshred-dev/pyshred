![PyPI](https://img.shields.io/pypi/v/pyshred)
![Python](https://img.shields.io/pypi/pyversions/pyshred)

![License](https://img.shields.io/github/license/pyshred-dev/pyshred)
![CI](https://github.com/pyshred-dev/pyshred/actions/workflows/sphinx.yml/badge.svg)

# PySHRED

**PySHRED** is a deep-learning library for reconstructing and forecasting high-dimensional spatiotemporal systems from sparse sensor data.

Built on the **SH**allow **RE**current **D**ecoder (SHRED) architecture, PySHRED provides a seamless pipeline from raw sensor measurements to high-fidelity reconstructions and long-horizon forecasts.

![SHRED architecture](https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/main_figure.png)

## SHRED in a Nutshell

| Component             | Role                                                                      | Models                 |
| --------------------- | ------------------------------------------------------------------------- | ---------------------- |
| **Sequence model**    | Encodes temporal sensor measurements into a low-dimensional latent state. | LSTM, GRU, Transformer |
| **Decoder model**     | Reconstructs the full high-dimensional state from the latent state.       | MLP, U-Net             |
| **Latent forecaster** | Propagates latent dynamics forward in time for long-horizon prediction.   | LSTM, SINDy            |

The **sequence + decoder** pair reconstructs the full high-dimensional state space from sparse sensors, while the **forecaster + decoder** pair enables multi-step forecasting with no additional sensor measurements.

PySHRED is a powerful tool for:

- System identification
- Reduced-order modeling
- Long-horizon forecasting
- Latent dynamics discovery
- Parametric systems analysis
- Control and decision-making

PySHRED offers a high-level interface and a simple three-step pipeline, making it easy for anyone to get started.

![PySHRED Pipeline](https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/pipeline.png)

## Documentation

Online documentation: [pyshred-dev.github.io/pyshred/stable](https://pyshred-dev.github.io/pyshred/stable)

The docs include:

- 📘 [**Getting Started**](https://pyshred-dev.github.io/pyshred/stable/user_guide/introduction.html)
- 📖 [**User Guide**](https://pyshred-dev.github.io/pyshred/stable/user_guide/index.html)
- 🧪 [**Example Gallery**](https://pyshred-dev.github.io/pyshred/stable/examples/index.html)
- 🛠️ [**API Reference**](https://pyshred-dev.github.io/pyshred/stable/pyshred/modules.html)

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

- Docs: [https://pyshred-dev.github.io/pyshred/stable](https://pyshred-dev.github.io/pyshred/stable)
- Issue Tracking: [https://github.com/pyshred-dev/pyshred/issues](https://github.com/pyshred-dev/pyshred/issues)
- Source code: [https://github.com/pyshred-dev/pyshred](https://github.com/pyshred-dev/pyshred)

## Contributors and Developers

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/contributors/Nathan_Kutz.png" width="100" style="border-radius:50%"><br>
      <sub><b>Nathan Kutz</b></sub><br>
      <!-- <sub><i></i></sub> -->
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/contributors/Jan_Williams.jpg" width="100" style="border-radius:50%"><br>
      <sub><b>Jan Williams</b></sub><br>
      <!-- <sub><i></i></sub> -->
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/contributors/David_Ye.jpg" width="100" style="border-radius:50%"><br>
      <sub><b>David Ye</b></sub><br>
      <!-- <sub><i></i></sub> -->
    </td>
        <td align="center">
      <img src="https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/contributors/Mars_Gao.jpg" width="100" style="border-radius:50%"><br>
      <sub><b>Mars Gao</b></sub><br>
      <!-- <sub><i></i></sub> -->
    </td>
        <td align="center">
      <img src="https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/contributors/Matteo_Tomasetto.png" width="100" style="border-radius:50%"><br>
      <sub><b>Matteo Tomasetto</b></sub><br>
      <!-- <sub><i></i></sub> -->
    </td>
        <td align="center">
      <img src="https://raw.githubusercontent.com/pyshred-dev/pyshred/main/docs/source/_static/contributors/Stefano_Riva.png" width="100" style="border-radius:50%"><br>
      <sub><b>Stefano Riva</b></sub><br>
      <!-- <sub><i></i></sub> -->
    </td>
  </tr>
</table>

<hr>

<a href="https://github.com/pyshred-dev/pyshred/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pyshred-dev/pyshred" />
</a>
<sub>Made with <a href="https://contrib.rocks">contrib.rocks</a>.</sub>

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
