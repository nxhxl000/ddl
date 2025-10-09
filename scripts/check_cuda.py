import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("Built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())
    print("cuDNN:", torch.backends.cudnn.version())