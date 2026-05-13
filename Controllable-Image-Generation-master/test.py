import torch

# 检查是否有可用的 CUDA GPU
cuda_available = torch.cuda.is_available()
print("CUDA 可用:", cuda_available)

# 当前默认设备
device = torch.device("cuda" if cuda_available else "cpu")
print("当前使用设备:", device)

# 查看当前 CUDA 设备编号
if cuda_available:
    print("当前 CUDA 设备:", torch.cuda.current_device())
    print("CUDA 设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("总显存 (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)