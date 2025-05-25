import numpy as np

# dimensions: (time, x, y, z, rgba)
SIZE = 16
tensor = np.zeros((SIZE, SIZE, SIZE, 4), dtype=np.float32)

# set full‐opacity alpha
tensor[..., 3] = 1.0

# frame 0: pink (1,0,1)
tensor[..., :3] = [1.0, 0.0, 0.7]

np.save("tensors/pink_orange_voxel.npy", tensor)
