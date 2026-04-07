"""
Bit-packing utilities for TurboQuant compression.

Achieves paper-compliant compression ratios by packing indices and signs
into individual bits rather than using uint8/int8.
"""

import torch
import numpy as np
from typing import Tuple


def pack_bits(data: torch.Tensor, bits_per_value: int) -> torch.Tensor:
    """
    Pack values into bits.
    
    Args:
        data: Tensor of values to pack (any shape, will be flattened)
        bits_per_value: Number of bits per value (1-8)
        
    Returns:
        Packed uint8 tensor
    """
    if bits_per_value == 8:
        return data.to(torch.uint8)
    
    data_flat = data.flatten()
    num_values = data_flat.shape[0]
    
    total_bits = num_values * bits_per_value
    num_bytes = (total_bits + 7) // 8
    
    packed = torch.zeros(num_bytes, dtype=torch.uint8, device=data.device)
    
    data_np = data_flat.cpu().numpy().astype(np.uint8)
    packed_np = packed.cpu().numpy()
    
    bit_pos = 0
    
    for val in data_np:
        mask = (1 << bits_per_value) - 1
        val_bits = val & mask
        
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        
        # Pack into current byte
        packed_np[byte_idx] |= (val_bits << bit_offset)
        
        # Handle overflow to next byte
        bits_remaining = 8 - bit_offset
        if bits_remaining < bits_per_value:
            overflow_bits = bits_per_value - bits_remaining
            packed_np[byte_idx + 1] |= (val_bits >> bits_remaining)
        
        bit_pos += bits_per_value
    
    return torch.from_numpy(packed_np).to(device=data.device)


def unpack_bits(packed: torch.Tensor, num_values: int, bits_per_value: int) -> torch.Tensor:
    """
    Unpack bits into values.
    
    Args:
        packed: Packed uint8 tensor
        num_values: Number of values to unpack
        bits_per_value: Number of bits per value (1-8)
        
    Returns:
        Unpacked tensor of shape (num_values,)
    """
    if bits_per_value == 8:
        return packed[:num_values].long()
    
    mask = (1 << bits_per_value) - 1
    
    unpacked = torch.zeros(num_values, dtype=torch.long, device=packed.device)
    
    packed_np = packed.cpu().numpy()
    
    bit_pos = 0
    
    for i in range(num_values):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        
        # Extract bits from current byte
        val = (packed_np[byte_idx] >> bit_offset) & mask
        
        # Handle overflow from next byte
        bits_remaining = 8 - bit_offset
        if bits_remaining < bits_per_value:
            overflow_bits = bits_per_value - bits_remaining
            overflow_mask = (1 << overflow_bits) - 1
            val |= ((packed_np[byte_idx + 1] & overflow_mask) << bits_remaining)
        
        unpacked[i] = val
        
        bit_pos += bits_per_value
    
    return unpacked


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """
    Pack binary signs (+1/-1) into 1 bit per value.
    
    Args:
        signs: Tensor of +1 or -1 values (any shape)
        
    Returns:
        Packed uint8 tensor (1 bit per value)
    """
    binary = (signs > 0).flatten()
    return pack_bits(binary.long(), bits_per_value=1)


def unpack_signs(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """
    Unpack bits into signs (+1/-1).
    
    Args:
        packed: Packed uint8 tensor
        num_values: Number of signs to unpack
        
    Returns:
        Tensor of +1 or -1 values
    """
    binary = unpack_bits(packed, num_values, bits_per_value=1)
    return torch.where(binary == 1, torch.tensor(1.0), torch.tensor(-1.0)).to(packed.device)


def calculate_packed_size(num_values: int, bits_per_value: int) -> int:
    """Calculate number of bytes needed to pack num_values with bits_per_value each."""
    values_per_byte = 8 // bits_per_value
    return (num_values + values_per_byte - 1) // values_per_byte


def get_bytes_per_coordinate(mse_bits: int, qjl_bits: int = 1) -> float:
    """
    Calculate actual bytes per coordinate with bit-packing.
    
    Args:
        mse_bits: Bits for MSE quantization (2-4)
        qjl_bits: Bits for QJL correction (1)
        
    Returns:
        Bytes per coordinate after packing
    """
    total_bits = mse_bits + qjl_bits
    return total_bits / 8.0


def get_compression_ratio(mse_bits: int, qjl_bits: int = 1) -> float:
    """
    Get compression ratio vs FP16.
    
    Args:
        mse_bits: Bits for MSE quantization
        qjl_bits: Bits for QJL correction
        
    Returns:
        Compression ratio (e.g., 5.33 for 3-bit)
    """
    total_bits = mse_bits + qjl_bits
    return 16.0 / total_bits
