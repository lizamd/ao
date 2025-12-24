"""Python wrapper for MXFP8 HIP kernel on AMD MI350/MI355"""

import torch
import os
import ctypes
from typing import Tuple

_hip_lib = None

def _load_hip_library():
    """Load the compiled MXFP8 HIP library."""
    global _hip_lib
    if _hip_lib is not None:
        return _hip_lib

    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(current_dir, "_mxfp8_hip_kernel.so")

    if not os.path.exists(lib_path):
        raise RuntimeError(f"MXFP8 HIP kernel library not found at {lib_path}")

    _hip_lib = ctypes.CDLL(lib_path)

    _hip_lib.mxfp8_quantize_hip_launcher.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # output
        ctypes.c_void_p,  # scales
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # cols
        ctypes.c_int64,   # scale_dim_y
        ctypes.c_int,     # input_dtype
        ctypes.c_void_p,  # stream
    ]
    _hip_lib.mxfp8_quantize_hip_launcher.restype = None

    return _hip_lib


def _mxfp8_quantize_hip_impl(
    x: torch.Tensor,
    rowwise: bool = False,
    colwise: bool = True,
    scale_dim_x: int = 1,
    scale_dim_y: int = 32,
    fp8_format: str = "e4m3",
    scaling_mode: str = "rceil",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal implementation - not safe for torch.compile."""
    if not x.is_cuda:
        raise ValueError("Input must be on CUDA device")
    if x.dim() != 2:
        raise ValueError(f"Input must be 2D, got {x.dim()}D")
    if rowwise:
        raise ValueError("Rowwise scaling not supported yet")
    if not colwise:
        raise ValueError("At least colwise must be True")

    rows, cols = x.shape
    device = x.device

    # Map dtype
    if x.dtype == torch.float32:
        input_dtype = 0
    elif x.dtype == torch.bfloat16:
        input_dtype = 1
    elif x.dtype == torch.float16:
        input_dtype = 2
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    # Create output tensors - column-major layout
    output_rowwise = torch.empty(0, dtype=torch.float8_e4m3fnuz, device=device)
    scales_rowwise = torch.empty(0, dtype=torch.float8_e8m0fnu, device=device)

    num_row_blocks = (rows + scale_dim_y - 1) // scale_dim_y
    
    # Column-major: stride[0]=1, stride[1]=rows
    output_colwise = torch.empty_strided((rows, cols), (1, rows), dtype=torch.float8_e4m3fnuz, device=device)
    scales_colwise = torch.zeros((cols, num_row_blocks), dtype=torch.float8_e8m0fnu, device=device).as_strided((cols, num_row_blocks), (1, cols))

    # Load library
    lib = _load_hip_library()

    # Get stream
    stream = torch.cuda.current_stream(device).cuda_stream

    # Call HIP kernel
    lib.mxfp8_quantize_hip_launcher(
        x.data_ptr(),
        output_colwise.data_ptr(),
        scales_colwise.data_ptr(),
        ctypes.c_int64(rows),
        ctypes.c_int64(cols),
        ctypes.c_int64(scale_dim_y),
        ctypes.c_int(input_dtype),
        ctypes.c_void_p(stream),
    )

    torch.cuda.synchronize(device)
    return output_rowwise, output_colwise, scales_rowwise, scales_colwise


# Register as a custom op to make it torch.compile compatible
@torch.library.custom_op("torchao::mxfp8_quantize_hip", mutates_args=())
def mxfp8_quantize_hip_op(
    x: torch.Tensor,
    scale_dim_y: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom op wrapper for MXFP8 HIP kernel - torch.compile safe."""
    _, output_colwise, _, scales_colwise = _mxfp8_quantize_hip_impl(
        x, 
        rowwise=False, 
        colwise=True,
        scale_dim_y=scale_dim_y
    )
    return output_colwise, scales_colwise


@mxfp8_quantize_hip_op.register_fake
def _(x: torch.Tensor, scale_dim_y: int = 32):
    """Fake implementation for torch.compile meta analysis."""
    rows, cols = x.shape
    num_row_blocks = (rows + scale_dim_y - 1) // scale_dim_y
    
    # Column-major layout
    output_colwise = torch.empty_strided(
        (rows, cols), (1, rows), 
        dtype=torch.float8_e4m3fnuz, 
        device=x.device
    )
    scales_colwise = torch.empty_strided(
        (cols, num_row_blocks), (1, cols),
        dtype=torch.float8_e8m0fnu,
        device=x.device
    )
    return output_colwise, scales_colwise


# Compatibility wrapper that matches the original API
def mxfp8_quantize_hip(
    x: torch.Tensor,
    rowwise: bool = False,
    colwise: bool = True,
    scale_dim_x: int = 1,
    scale_dim_y: int = 32,
    fp8_format: str = "e4m3",
    scaling_mode: str = "rceil",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to MXFP8 format using HIP kernels."""
    # Use custom op if available, otherwise fall back to direct implementation
    output_colwise, scales_colwise = torch.ops.torchao.mxfp8_quantize_hip(x, scale_dim_y)
    
    # Return empty tensors for rowwise to match expected API
    output_rowwise = torch.empty(0, dtype=torch.float8_e4m3fnuz, device=x.device)
    scales_rowwise = torch.empty(0, dtype=torch.float8_e8m0fnu, device=x.device)
    
    return output_rowwise, output_colwise, scales_rowwise, scales_colwise
