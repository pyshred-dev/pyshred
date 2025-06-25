__version__ = "v1.0.21"

from .engine.engine import SHREDEngine
from .models.latent_forecaster_models.sindy import SINDy_Forecaster
from .models.latent_forecaster_models.lstm import LSTM_Forecaster
from .models.decoder_models.mlp_model import MLP
from .models.decoder_models.unet_model import UNET
from .models.sequence_models.lstm_model import LSTM
from .models.sequence_models.transformer_model import TRANSFORMER
from .models.sequence_models.gru_model import GRU
from .models.shred import SHRED
from .processor.data_manager import DataManager
from .processor.parametric_data_manager import ParametricDataManager
from .engine.parametric_engine import ParametricSHREDEngine
from .objects.device import get_device, set_default_device, DeviceConfig, print_device_info

__all__ = [
    "SHREDEngine",
    "SINDy_Forecaster",
    "LSTM_Forecaster",
    "MLP",
    "UNET",
    "LSTM",
    "TRANSFORMER",
    "GRU",
    "SHRED",
    "DataManager",
    "ParametricDataManager",
    "ParametricSHREDEngine",
    "get_device",
    "set_default_device",
    "DeviceConfig",
    ]

# Convenience functions for device management
def device_info():
    """Print information about available devices and current configuration."""
    print_device_info()

def set_device(device=None, **kwargs):
    """
    Convenience function to set the default device.
    
    Examples:
    ---------
    >>> import pyshred
    >>> pyshred.set_device("cuda")  # Use CUDA
    >>> pyshred.set_device("mps")   # Use MPS (Apple Silicon)
    >>> pyshred.set_device("cpu")   # Force CPU
    >>> pyshred.set_device("cuda", device_id=1)  # Use specific GPU
    """
    return set_default_device(device, **kwargs)