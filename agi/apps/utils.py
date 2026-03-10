

def pick_free_device(threshold_ratio: float = 0.6):
    """挑一张空闲卡, threshold_ratio 表示空闲率阈值 (默认显存占用 <90% 才认为是空闲)"""
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return torch.device("cpu")

    best_device = None
    max_free = -1
    for i in range(num_gpus):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        free_ratio = free_mem / total_mem
        if free_ratio > threshold_ratio and free_mem > max_free:
            best_device = i
            max_free = free_mem

    if best_device is None:  # 没找到合适的，退回 cuda:0
        best_device = 0
    return torch.device(f"cuda:{best_device}")

def best_torch_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32  # CPU 上只能用 float32

    device = torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(device)

    if major >= 8:
        return torch.bfloat16  # Ampere及以上，支持 BF16，优先使用
    elif major >= 7:
        return torch.float16   # Volta/Turing 支持 FP16
    else:
        return torch.float32   # 老架构，回退到 FP32