import sys, torch
print("python executable:", sys.executable)
print("torch.__version__:", getattr(torch, '__version__', None))
print("torch.version.cuda:", getattr(torch.version, 'cuda', None))
print("torch.backends.cudnn.version():", getattr(torch.backends.cudnn, 'version', lambda: None)())
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))