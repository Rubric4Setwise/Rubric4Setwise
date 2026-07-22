
from typing import Union, List, Optional, Tuple
import torch

def get_device(
        device: Optional[Union[str, torch.device]],
        no_mps: bool = False,
    ) -> Union[str, torch.device]:
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and not no_mps:
                device = "mps"
            else:
                device = "cpu"
        return device

def get_dtype(
        dtype: Optional[Union[str, torch.dtype]],
        device: Optional[Union[str, torch.device]],
        verbose: int = 1,
    ) -> torch.dtype:
        if dtype is None:
            print("No dtype set")
        if device == "cpu":
            dtype = torch.float32
        if not isinstance(dtype, torch.dtype):
            if dtype == "fp16" or "float16":
                dtype = torch.float16
            elif dtype == "bf16" or "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        return dtype