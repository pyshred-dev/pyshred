"""
Device management utilities for PyShred.

This module provides centralized device detection and management functionality
that supports CUDA, MPS (Apple Silicon), and CPU backends with proper fallback logic.
"""

import torch
import warnings
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum


class DeviceType(Enum):
    """Supported device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class DeviceConfig:
    """Configuration for device selection and management."""
    device_type: DeviceType = DeviceType.AUTO
    device_id: Optional[int] = None  # For multi-GPU setups
    force_cpu: bool = False  # Override auto-detection to force CPU
    warn_on_fallback: bool = True  # Warn when falling back to different device


# Global device configuration
_global_device_config = DeviceConfig()


def _detect_best_device() -> torch.device:
    """
    Automatically detect the best available device with proper fallback logic.
    
    Priority order:
    1. CUDA (if available and not forced to CPU)
    2. MPS (if available on Apple Silicon and not forced to CPU)  
    3. CPU (fallback)
    
    Returns
    -------
    torch.device
        The best available device.
    """
    if _global_device_config.force_cpu:
        return torch.device("cpu")
    
    # Check for CUDA first (highest priority for ML workloads)
    if torch.cuda.is_available():
        device_id = _global_device_config.device_id
        if device_id is not None:
            if device_id >= torch.cuda.device_count():
                if _global_device_config.warn_on_fallback:
                    warnings.warn(
                        f"CUDA device {device_id} not available. "
                        f"Only {torch.cuda.device_count()} CUDA devices found. "
                        f"Falling back to cuda:0.",
                        UserWarning
                    )
                device_id = 0
            return torch.device(f"cuda:{device_id}")
        else:
            return torch.device("cuda")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # Fallback to CPU
    return torch.device("cpu")


def _get_device_from_config() -> torch.device:
    """
    Get device based on current configuration.
    
    Returns
    -------
    torch.device
        Device according to current configuration.
    
    Raises
    ------
    RuntimeError
        If requested device is not available.
    """
    config = _global_device_config
    
    if config.device_type == DeviceType.AUTO:
        return _detect_best_device()
    
    elif config.device_type == DeviceType.CPU:
        return torch.device("cpu")
    
    elif config.device_type == DeviceType.CUDA:
        if not torch.cuda.is_available():
            if config.warn_on_fallback:
                warnings.warn(
                    "CUDA requested but not available. Falling back to CPU.",
                    UserWarning
                )
            return torch.device("cpu")
        
        device_id = config.device_id
        if device_id is not None:
            if device_id >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device {device_id} not available. "
                    f"Only {torch.cuda.device_count()} CUDA devices found."
                )
            return torch.device(f"cuda:{device_id}")
        else:
            return torch.device("cuda")
    
    elif config.device_type == DeviceType.MPS:
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            if config.warn_on_fallback:
                warnings.warn(
                    "MPS requested but not available. Falling back to CPU.",
                    UserWarning
                )
            return torch.device("cpu")
        return torch.device("mps")
    
    else:
        raise ValueError(f"Unknown device type: {config.device_type}")


def get_device() -> torch.device:
    """
    Get the current default device for PyShred operations.
    
    Returns
    -------
    torch.device
        The current default device.
    
    Examples
    --------
    >>> device = get_device()
    >>> tensor = torch.zeros(10, device=device)
    """
    return _get_device_from_config()


def set_default_device(
    device: Union[str, torch.device, DeviceType, None] = None,
    device_id: Optional[int] = None,
    force_cpu: bool = False,
    warn_on_fallback: bool = True
) -> torch.device:
    """
    Set the default device for PyShred operations.
    
    Parameters
    ----------
    device : str, torch.device, DeviceType, or None, optional
        Device to use. Can be:
        - "auto": Auto-detect best device (default)
        - "cpu": Force CPU usage
        - "cuda": Use CUDA if available
        - "mps": Use MPS if available (Apple Silicon)
        - torch.device object
        - DeviceType enum value
    device_id : int, optional
        Specific device ID for multi-GPU setups (only used with CUDA).
    force_cpu : bool, optional
        If True, override all other settings and force CPU usage.
    warn_on_fallback : bool, optional
        Whether to warn when falling back to a different device.
        
    Returns
    -------
    torch.device
        The device that will be used.
        
    Examples
    --------
    >>> # Auto-detect best device
    >>> device = set_default_device()
    
    >>> # Force CUDA usage
    >>> device = set_default_device("cuda")
    
    >>> # Use specific GPU
    >>> device = set_default_device("cuda", device_id=1)
    
    >>> # Force CPU (useful for debugging)
    >>> device = set_default_device(force_cpu=True)
    
    >>> # Use MPS on Apple Silicon
    >>> device = set_default_device("mps")
    """
    global _global_device_config
    
    # Handle different input types
    if device is None or device == "auto":
        device_type = DeviceType.AUTO
    elif isinstance(device, str):
        device_type = DeviceType(device.lower())
    elif isinstance(device, torch.device):
        device_str = device.type
        if device_str == "cuda" and device.index is not None:
            device_id = device.index
        device_type = DeviceType(device_str)
    elif isinstance(device, DeviceType):
        device_type = device
    else:
        raise TypeError(f"Invalid device type: {type(device)}")
    
    # Update global configuration
    _global_device_config = DeviceConfig(
        device_type=device_type,
        device_id=device_id,
        force_cpu=force_cpu,
        warn_on_fallback=warn_on_fallback
    )
    
    # Return the actual device that will be used
    return get_device()


def get_device_info() -> dict:
    """
    Get information about current device configuration and availability.
    
    Returns
    -------
    dict
        Dictionary containing device information.
    """
    current_device = get_device()
    
    info = {
        "current_device": str(current_device),
        "device_config": _global_device_config,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(i) 
            for i in range(torch.cuda.device_count())
        ]
    
    return info


def print_device_info():
    """Print detailed information about device configuration and availability."""
    info = get_device_info()
    
    print("=== PyShred Device Information ===")
    print(f"Current device: {info['current_device']}")
    print(f"Device config: {info['device_config']}")
    print()
    print("Device Availability:")
    print(f"  CUDA available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"  CUDA devices: {info['cuda_device_count']}")
        for i, name in enumerate(info['cuda_device_names']):
            print(f"    cuda:{i} - {name}")
    print(f"  MPS available: {info['mps_available']}")
    print(f"  CPU: Always available") 