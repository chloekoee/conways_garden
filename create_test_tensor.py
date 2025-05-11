import numpy as np

tensor = np.zeros((2, 1, 1, 1, 4), dtype=np.float32)
tensor[0, 0, 0, 0] = [1, 0, 1, 1]
tensor[1, 0, 0, 0] = [1, 0.5, 0, 1]
np.save("pink_orange_voxel.npy", tensor)