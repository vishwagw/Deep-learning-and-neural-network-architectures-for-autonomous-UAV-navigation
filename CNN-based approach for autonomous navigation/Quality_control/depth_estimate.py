# Using MiDaS (requires torch)
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
depth_map = midas(torch.from_numpy(frame).unsqueeze(0))