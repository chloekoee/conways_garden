import numpy as np
import matplotlib.pyplot as plt
import math
from opensimplex import noise2

X_SIZE = 60
Y_SIZE = 40
Z_SIZE = 60

cx, cz = (X_SIZE / 2.0), (Z_SIZE / 2.0)

TIP_COLOR = np.array([179 / 255, 61 / 255, 120 / 255])
MOSS_COLOUR = np.array([47 / 255, 64 / 255, 20 / 255])
BASE_COLOR = np.array([64 / 255, 17 / 255, 0 / 255])

earth_base = np.array([0.5, 0.4, 0.3])  # earthy brown
blend_factor = 0.4  # 0 = original, 1 = earth only

darken = 0.2  # 0 = black, 1 = unchanged

TIP_COLOR = TIP_COLOR * (1 - blend_factor) + earth_base * blend_factor
BASE_COLOR = BASE_COLOR * (1 - blend_factor) + earth_base * blend_factor
MOSS_COLOUR = MOSS_COLOUR * (1 - blend_factor) + earth_base * blend_factor

TIP_COLOR *= darken
BASE_COLOR *= darken
MOSS_COLOUR *= darken
# 55, 230, 21
noise_amp = 0.05
alpha_min = 0.4
f_noise = 0.1


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
            col = BASE_COLOR + (TIP_COLOR - BASE_COLOR) * h
            n_green = (noise2(x * 0.18, z * 0.18)) / 1.5
            # Interpolate between base color and green using n_green
            col = (1 - n_green) * col + n_green * MOSS_COLOUR

            col += (n - 0.5) * noise_amp
            col = np.clip(col, 0, 1)

            # vary r with the radius
            alpha = 1 - r * (1 - alpha_min)
            alpha += (n - 0.5) * (noise_amp * 0.5)
            alpha = float(np.clip(alpha, alpha_min, 1))

            island[x, y, z, :] = [col[0], col[1], col[2], alpha]

# invert cone to make island
island = island[:, ::-1, :, :]
np.save("islands/island.npy", island)

## Showing 3D volumetric plot - takes very long
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.set_box_aspect(island.shape[0:3] / np.max(island.shape[0:3]))

# ax.cla()
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# ax.voxels(
#     filled=(island[:, :, :, 3] > 0.1),
#     facecolors=np.clip(island[:, :, :, :4], 0, 1),
# )

# plt.show()
# plt.close()

## Showing slice of top and side

# Top face (largest y)
y_top = island.shape[1] - 1
z_side = island.shape[2] // 2
birdseye = island[:, y_top, :, :3]
sideon = island[:, :, z_side, :3]

# Plot top face
plt.imshow(birdseye)
plt.axis("off")
plt.show()

plt.imshow(sideon)
plt.axis("off")
plt.show()
