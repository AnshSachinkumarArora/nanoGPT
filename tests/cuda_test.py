import torch

print("Torch Version:   ", torch.__version__)
print("CUDA Version:    ", torch.version.cuda)
print("Supported Archs: ", torch.cuda.get_arch_list())
print("Device Name:     ", torch.cuda.get_device_properties(0).name)
print("Tensor Test:     ", torch.randn(1).cuda())
