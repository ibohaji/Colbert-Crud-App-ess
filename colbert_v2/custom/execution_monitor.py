import time
import inspect
import os
import pynvml
import torch
import numpy as np
from ..config import MetaData  
import time
import pynvml
import torch
import os
import numpy as np

def monitor_gpu(func):
    """Simple decorator to monitor GPU memory usage."""
    def wrapper(*args, **kwargs):
        gpu_memory_before = print_gpu_utilisation()
        function_name = func.__name__

        start_time = time.time()
        result = func(*args, **kwargs)  
        end_time = time.time()

        gpu_memory_after = print_gpu_utilisation()
        gpu_utilization = MetaData().gpu_utilization
        gpu_utilization.append({function_name: gpu_memory_after})
        
        MetaData().update(
                 gpu_utilization = gpu_utilization               
        )

        print(f"GPU Memory Before: {gpu_memory_before}")
        print(f"GPU Memory After: {gpu_memory_after}")
        print(f"Execution Time: {end_time - start_time} seconds")

        return result
    return wrapper

def print_gpu_utilisation():
    """Get GPU memory utilization using PyNVML"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        torch_gpu_id = torch.cuda.current_device()
        pynvml.nvmlInit() 
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        nvml_gpu_id = int(devices[torch_gpu_id]) 
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_gpu_id)
        gpu_name = pynvml.nvmlDeviceGetName(handle)

        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info_used = info.used // 1024 ** 2  # Convert to MB
        info_total = info.total // 1024 ** 2  # Convert to MB

        gpu_memory_info = {
            "name": gpu_name,
            "used": info_used,
            "total": info_total,
            "percentage": np.round((info_used * 100) / info_total, 2)
        }

        print(f"GPU {nvml_gpu_id} memory occupied: {info_used}/{info_total} MB = {gpu_memory_info['percentage']}%.")
        pynvml.nvmlShutdown()

        return gpu_memory_info
    else:
        print("CUDA_VISIBLE_DEVICES not set. Skipping GPU memory measurement.")
        return {"used": 0, "total": 0, "percentage": 0}
    