import numpy as np
## for metal
sobelX = np.array(
    [
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
    ],
    dtype=np.float32,
)
sobelY = -np.rot90(sobelX, k=-1, axes=(0, 1)) 
sobelZ = np.transpose(sobelX, (2,1,0))

identity = np.zeros((3, 3, 3), dtype=np.float32)
identity[1, 1, 1] = 1.0
identity.ravel()

# Sobel X (z = 0 plane, then z = 1, then z = 2)
SOBEL_X = [
    -1.0,
    -2.0,
    -1.0,
    -2.0,
    -4.0,
    -2.0,
    -1.0,
    -2.0,
    -1.0,  # z = 0
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,  # z = 1
    1.0,
    2.0,
    1.0,
    2.0,
    4.0,
    2.0,
    1.0,
    2.0,
    1.0,  # z = 2
]

# Sobel Y (manually “minus‐rotate” SOBEL_X around the XY plane)
SOBEL_Y = [
    1.0,
    2.0,
    1.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    -2.0,
    -1.0,  # z = 0
    2.0,
    4.0,
    2.0,
    0.0,
    0.0,
    0.0,
    -2.0,
    -4.0,
    -2.0,  # z = 1
    1.0,
    2.0,
    1.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    -2.0,
    -1.0,  # z = 2
]

# Sobel Z (permute SOBEL_X so that original z‐axis becomes x)
SOBEL_Z = [
    -1.0,
    0.0,
    1.0,
    -2.0,
    0.0,
    2.0,
    -1.0,
    0.0,
    1.0,  # z = 0 (actually x‐slice)
    -2.0,
    0.0,
    2.0,
    -4.0,
    0.0,
    4.0,
    -2.0,
    0.0,
    2.0,  # z = 1
    -1.0,
    0.0,
    1.0,
    -2.0,
    0.0,
    2.0,
    -1.0,
    0.0,
    1.0,  # z = 2
]

# Identity (passes through center voxel for each z,y,x)
IDENTITY = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,  # z = 0
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,  # z = 1
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,  # z = 2
]
