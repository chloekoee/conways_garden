from numba import njit
import numpy as np
import glm
import math

# resolution
WIN_RES = glm.vec2(1920, 900)

# ray casting
MAX_RAY_DIST = 6

# camera
ASPECT_RATIO = WIN_RES.x / WIN_RES.y
FOV_DEG = 75
V_FOV = glm.radians(FOV_DEG)  # vertical FOV
H_FOV = 2 * math.atan(math.tan(V_FOV * 0.5) * ASPECT_RATIO)  # horizontal FOV
NEAR = 0.1
FAR = 2000.0
PITCH_MAX = glm.radians(89)

# player
PLAYER_SPEED = 0.01
PLAYER_ROT_SPEED = 0.003
## TODO: dynamically calculate this based on the NCA size
PLAYER_POS = glm.vec3(7.5, 17, 7.5)
MOUSE_SENSITIVITY = 0.002

# colors
BG_COLOR = glm.vec3(0.1, 0.16, 0.25)

# nca simulation
SECONDS_PER_STEP = 2
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
    [[1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0]]
)
