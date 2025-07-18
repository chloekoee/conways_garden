import numpy as np
import matplotlib.pyplot as plt
import math
from opensimplex import noise2

X_SIZE = 30
Y_SIZE = 40
Z_SIZE = 30

cx, cz = (X_SIZE / 2.0), (Z_SIZE / 2.0)


def get_height(x, z):

    # amplitude
    a1 = Y_SIZE
    a2, a4, a8 = a1 * 0.5, a1 * 0.25, a1 * 0.125

    # frequency
    f1 = 0.005
    f2, f4, f8 = f1 * 2, f1 * 4, f1 * 8

    if noise2(0.1 * x, 0.1 * z) < 0:
        a1 /= 1.07

    height = 0
    dx = x - cx
    dz = z - cz
    height += noise2(dx * f1, dz * f1) * a1 + a1
    height += noise2(dx * f2, dz * f2) * a2 - a2
    height += noise2(dx * f4, dz * f4) * a4 + a4
    height += noise2(dx * f8, dz * f8) * a8 - a8

    height = max(height, noise2(x * f8, z * f8) + 2)

    return min(height, Y_SIZE)


island = np.zeros((X_SIZE, Y_SIZE * 2, Z_SIZE, 4), dtype=np.float32)
R = min(X_SIZE, Z_SIZE) / 2.0

cx, cz = (X_SIZE - 1) / 2.0, (Z_SIZE - 1) / 2.0
R = min(X_SIZE, Z_SIZE) / 2.0

bottom_col = np.array([41, 28, 99])
bottom_col = bottom_col / 255
top_col = np.array([129, 149, 181])
top_col = top_col / 255
noise_amp = 0.05
alpha_min = 0.3
f_noise = 0.1

for x in range(X_SIZE):
    for z in range(Z_SIZE):
        dx, dz = x - cx, z - cz
        r = math.hypot(dx, dz)
        raw = Y_SIZE * (1 - r / R)
        height = int(np.clip(raw, 0, None))

        simplex_multiplier = noise2(x, z)
        height *= 1 + 0.5 * simplex_multiplier
        for y in range(int(height)):
            h = y / float(Y_SIZE)  # do 1- to invert the colouring

            # sample noise in [0,1]
            n = (noise2(x * f_noise, z * f_noise) + 1) / 2

            # interpolate colour and add jitter
            col = bottom_col + (top_col - bottom_col) * h
            col += (n - 0.5) * noise_amp
            col = np.clip(col, 0, 1)

            # alpha falls off toward tip, plus a bit of noise
            alpha = 1 - h * (1 - alpha_min)
            alpha += (n - 0.5) * (noise_amp * 0.5)
            alpha = float(np.clip(alpha, alpha_min, 1))

            island[x, y, z, :] = [col[0], col[1], col[2], alpha]

# invert cone to make island
island = island[:, ::-1, :, :]
np.save("islands/island.npy", island)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect(island.shape[0:3] / np.max(island.shape[0:3]))

ax.cla()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.voxels(
    filled=(island[:, :, :, 3] > 0.1),
    facecolors=np.clip(island[:, :, :, :4], 0, 1),
)

plt.show()
plt.close()
