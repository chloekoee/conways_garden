from numba import njit
import numpy as np
import glm
import math

# resolution
WIN_RES = glm.vec2(1600, 900)

# WIN_RES = glm.vec2(1920, 900)

# ray casting
MAX_RAY_DIST = 6

# chunk dimensions
# CHUNK_SIZE = 32
# H_CHUNK_SIZE = CHUNK_SIZE // 2
# CHUNK_AREA = CHUNK_SIZE * CHUNK_SIZE
# CHUNK_VOL = CHUNK_AREA * CHUNK_SIZE

# # world
# WORLD_W, WORLD_H, WORLD_D = 1, 1, 1  ##10, 3, 10
# WORLD_AREA = WORLD_W * WORLD_D
# WORLD_VOL = WORLD_AREA * WORLD_H

# # world centre
# CENTER_XZ = WORLD_W * H_CHUNK_SIZE
# CENTRE_Y = WORLD_H * H_CHUNK_SIZE

# camera
ASPECT_RATIO = WIN_RES.x / WIN_RES.y
FOV_DEG = 75
V_FOV = glm.radians(FOV_DEG)  # vertical FOV
H_FOV = 2 * math.atan(math.tan(V_FOV * 0.5) * ASPECT_RATIO)  # horizontal FOV
NEAR = 0.1
FAR = 2000.0
PITCH_MAX = glm.radians(89)

# player
PLAYER_SPEED = 0.01  # 0.005
PLAYER_ROT_SPEED = 0.003
# PLAYER_POS = glm.vec3(CENTER_XZ, WORLD_H * CHUNK_SIZE, CENTER_XZ)
PLAYER_POS = glm.vec3(7.5,17,7.5)
MOUSE_SENSITIVITY = 0.002

# colors
BG_COLOR = glm.vec3(0.1, 0.16, 0.25)

# offets for calculating vertex position
TOP = np.array(
    [
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1],
    ]
)

BOTTOM = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])

RIGHT = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])

LEFT = np.array([[1, 0], [0, 0], [0, 1], [1, 1]])

FRONT = np.array(
    [
        [0, 1],
        [0, 0],
        [1, 0],
        [1, 1],
    ]
)

BACK = np.array([[1, 1], [1, 0], [0, 0], [0, 1]])


# ambient occlusion offsets
# if we fix one axis, then iterate through all neighbors
# anti-clockwise, the offsets we use to obtain these neighbours are
# as below
AO_NEIGHBOURHOOD_EVEN = np.array(
    [[1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]]
)

AO_NEIGHBOURHOOD_ODD = np.array(
    [
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0],
    ]
)
