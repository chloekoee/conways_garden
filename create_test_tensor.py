import numpy as np

# dimensions: (time, x, y, z, rgba)
SIZE = 16
tensor = np.zeros((2, SIZE, SIZE, SIZE, 4), dtype=np.float32)

# set full‐opacity alpha
tensor[..., 3] = 0.7

# frame 0: pink (1,0,1)
tensor[0, ..., :3] = [1.0, -0.0, 0.7]

# frame 1: orange (1,0.5,0)
tensor[1, ..., :3] = [1.0, 0.5, 0.0]

np.save("tensors/pink_orange_voxel.npy", tensor)
