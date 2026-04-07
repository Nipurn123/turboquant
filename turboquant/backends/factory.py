"""
Backend factory for GPU-specific implementations.

Automatically selects the best available backend based on hardware.
"""

from typing import Dict, Optional, Type
import torch
from .base import BackendBase
from .pytorch import PyTorchBackend


_registered_backends: Dict[str, Type[BackendBase]] = {
    "pytorch": PyTorchBackend,
}


def register_backend(name: str, backend_class: Type[BackendBase]) -> None:
    """
    Register a new backend.
    
    Args:
        name: Backend name
        backend_class: Backend class (must inherit from BackendBase)
    """
    _registered_backends[name] = backend_class


def get_backend(
    engine: "TurboQuantEngine",
    backend_name: Optional[str] = None,
    device: Optional[str] = None,
) -> BackendBase:
    """
    Get appropriate backend for current hardware.
    
    Args:
        engine: TurboQuant engine instance
        backend_name: Specific backend to use (None for auto-detect)
        device: Device to use (None for auto-detect)
        
    Returns:
        Backend instance
    """
    if device is None:
        device = engine.device
    
    if backend_name is not None:
        if backend_name not in _registered_backends:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available: {list(_registered_backends.keys())}"
            )
        backend_class = _registered_backends[backend_name]
        return backend_class(engine)
    
    if device.startswith("cuda"):
        try:
            if _is_blackwell():
                backend_class = _registered_backends.get("blackwell", PyTorchBackend)
            elif _is_hopper():
                backend_class = _registered_backends.get("hopper", PyTorchBackend)
            else:
                backend_class = PyTorchBackend
        except Exception:
            backend_class = PyTorchBackend
    else:
        backend_class = PyTorchBackend
    
    return backend_class(engine)


def _is_blackwell() -> bool:
    """Check if current GPU is Blackwell architecture (B200, etc.)."""
    if not torch.cuda.is_available():
        return False
    
    major, minor = torch.cuda.get_device_capability()
    return major == 10


def _is_hopper() -> bool:
    """Check if current GPU is Hopper architecture (H200, H100, etc.)."""
    if not torch.cuda.is_available():
        return False
    
    major, minor = torch.cuda.get_device_capability()
    return major == 9


def list_backends() -> Dict[str, bool]:
    """
    List all registered backends and their availability.
    
    Returns:
        Dictionary mapping backend names to availability
    """
    availability = {}
    for name, backend_class in _registered_backends.items():
        try:
            if hasattr(backend_class, "is_available"):
                availability[name] = backend_class.is_available
            else:
                availability[name] = True
        except Exception:
            availability[name] = False
    return availability
