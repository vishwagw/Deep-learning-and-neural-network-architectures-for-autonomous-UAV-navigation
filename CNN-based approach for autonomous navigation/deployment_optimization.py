# Convert to TensorRT for NVIDIA Jetson
from torch2trt import torch2trt

model.eval()
example_input = torch.rand(1, 15, 224, 224).cuda()  # 5 frames * 3 channels
model_trt = torch2trt(model, [example_input], fp16_mode=True)

# Save optimized model
torch.save(model_trt.state_dict(), 'drone_navigation_trt.pth')