
import torch

def pick_free_device(threshold_ratio: float = 0.6):
    """挑一张空闲卡, threshold_ratio 表示空闲率阈值 (默认显存占用 <90% 才认为是空闲)"""
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
