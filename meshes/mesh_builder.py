from constants.settings import *
from numba import uint8
from numba import njit

"""
FACE ID is an enum:
0: top
1: bottom
2: right
3: left
4: front
5: back
"""


@njit
def to_uint8(x, y, z, r, g, b, a, face_id, ao_id):
    return (
        uint8(x),
        uint8(y),
        uint8(z),
        uint8(r),
        uint8(g),
        uint8(b),
        uint8(a),
        uint8(face_id),
        uint8(ao_id),
    )


@njit
def is_void(position, nca_tensor, shape):
    ## TODO: make more elegant way to see inner blocks
    return True
    x, y, z = position
    x_dim, y_dim, z_dim = shape
    if 0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim:
        # if in bounds and the neighbouring alpha value is false
        # if nca_tensor[x, y, z, 3] == 0:
        if nca_tensor[x, y, z, 3] < 1:

            return True
        # otherwise neighbouring alpha value is alive then don't render this face
        return False
    # if out of bounds (no alpha value) then this is an outer face
    return True


@njit
def add_data(vertex_data, index, *vertices):
    for vertex in vertices:
        for attr in vertex:
            vertex_data[index] = attr
            index += 1
    return index


@njit
def get_ao(position, nca_tensor, shape, fixed_axis, even=True, flip=False):
    x, y, z = position
    neighbours = []

    row = (fixed_axis + 2) % 3 if flip else (fixed_axis + 1) % 3
    col = (fixed_axis + 1) % 3 if flip else (fixed_axis + 2) % 3

    neighbourhood = AO_NEIGHBOURHOOD_EVEN if even else AO_NEIGHBOURHOOD_ODD

    for n in neighbourhood:
        v = [0, 0, 0]
        v[row] = n[0]
        v[col] = n[1]
        neighbours.append(is_void((x + v[0], y + v[1], z + v[2]), nca_tensor, shape))

    a = sum([neighbours[7]] + neighbours[0:2])
    b = sum(neighbours[1:4])
    c = sum(neighbours[3:6])
    d = sum(neighbours[5:8])

    return (a, b, c, d)


@njit
def add_face(
    fixed_axis: np.int64,
    base_vector: np.ndarray,
    offsets: np.ndarray,
    vertex_data: np.ndarray,
    index: np.int64,
    rgba: np.ndarray,
    face_id,
    ao_values,
) -> np.int64:
    """
    Adds vertices for a quad face defined by a fixed axis and a base coordinate.
    
              v0 *---* v1
                /     \
            v3 *-------*  v2

    The function creates a quad from 4 vertices using offsets for the unfixed axes and
    then breaks the quad into 2 triangles by calling add_data.
    """
    v = []
    k = [i for i in range(3) if i != fixed_axis]
    for i in range(4):
        vertex = base_vector.copy()
        vertex[k[0]] += offsets[i, 0]
        vertex[k[1]] += offsets[i, 1]
        v.append(
            to_uint8(
                vertex[0],
                vertex[1],
                vertex[2],
                rgba[0],
                rgba[1],
                rgba[2],
                rgba[3],
                face_id,
                ao_values[i],
            )
        )

    # first triangle uses v0, v1, v2,  second triangle uses v0, v2, v3
    return add_data(vertex_data, index, v[0], v[1], v[2], v[0], v[2], v[3])


@njit
def build_nca_mesh(nca_tensor: np.ndarray, format_size: int):
    shape = nca_tensor.shape[:3]
    x_dim, y_dim, z_dim = shape
    nca_volume = x_dim * y_dim * z_dim

    # Allocate a vertex data array.
    ## we account for the case where all faces are visible
    # vertex_data = np.empty(nca_volume * 18 * format_size, dtype="uint8")
    vertex_data = np.empty(nca_volume * 36 * format_size, dtype="uint8")

    index = 0

    ## Visit voxels from centre of region and spiral outwards, such that the 
    # outermost voxels are visited last
    cx, cy, cz = x_dim // 2, y_dim // 2, z_dim // 2
    R = max(cx, x_dim - 1 - cx, cy, y_dim - 1 - cy, cz, z_dim - 1 - cz)
    for d in range(0, R + 1):
        for x in range(cx - d, cx + d + 1):
            if x < 0 or x >= x_dim:
                continue
            for y in range(cy - d, cy + d + 1):
                if y < 0 or y >= y_dim:
                    continue
                for z in range(cz - d, cz + d + 1):
                    if z < 0 or z >= z_dim:
                        continue

                    # check if still in bounds
                    if max(abs(x - cx), abs(y - cy), abs(z - cz)) != d:
                        continue
                    
                    # Skip if voxel is empty (alpha channel, index 3, is zero)
                    if nca_tensor[x, y, z, 3] < MIN_ALPHA:
                        continue

                    rgba = nca_tensor[x, y, z, :]

                    # Right face (face_id 2): sample neighbor at x+1
                    if is_void((x + 1, y, z), nca_tensor, shape):
                        ao_values = get_ao(
                            (x + 1, y, z), nca_tensor, shape, fixed_axis=0
                        )
                        index = add_face(
                            fixed_axis=0,
                            base_vector=np.array([x + 1, y, z], dtype=np.int64),
                            offsets=RIGHT,
                            vertex_data=vertex_data,
                            index=index,
                            rgba=(rgba),
                            face_id=2,
                            ao_values=ao_values,
                        )

                    # Top face (face_id 0): sample neighbor at y+1
                    if is_void((x, y + 1, z), nca_tensor, shape):
                        ao_values = get_ao(
                            (x, y + 1, z), nca_tensor, shape, fixed_axis=1
                        )
                        index = add_face(
                            fixed_axis=1,
                            base_vector=np.array([x, y + 1, z], dtype=np.int64),
                            offsets=TOP,
                            vertex_data=vertex_data,
                            index=index,
                            rgba=(rgba),  # supply your rgba tuple
                            face_id=0,
                            ao_values=ao_values,
                        )

                    # Front face (face_id 5): sample neighbor at z+1
                    if is_void((x, y, z + 1), nca_tensor, shape):
                        ao_values = get_ao(
                            (x, y, z + 1),
                            nca_tensor,
                            shape,
                            fixed_axis=2,
                            even=False,
                            flip=True,
                        )
                        index = add_face(
                            fixed_axis=2,
                            base_vector=np.array([x, y, z + 1], dtype=np.int64),
                            offsets=FRONT,
                            vertex_data=vertex_data,
                            index=index,
                            rgba=(rgba),
                            face_id=5,
                            ao_values=ao_values,
                        )

                    # Left face (face_id 3): sample neighbor at x-1
                    if is_void((x - 1, y, z), nca_tensor, shape):
                        ao_values = get_ao(
                            (x - 1, y, z), nca_tensor, shape, fixed_axis=0, even=False
                        )
                        index = add_face(
                            fixed_axis=0,
                            base_vector=np.array([x, y, z], dtype=np.int64),
                            offsets=LEFT,
                            vertex_data=vertex_data,
                            index=index,
                            rgba=(rgba),
                            face_id=3,
                            ao_values=ao_values,
                        )

                    # Bottom face (face_id 1): sample neighbor at y-1
                    if is_void((x, y - 1, z), nca_tensor, shape):
                        ao_values = get_ao(
                            (x, y - 1, z), nca_tensor, shape, fixed_axis=1, even=False
                        )
                        index = add_face(
                            fixed_axis=1,
                            base_vector=np.array([x, y, z], dtype=np.int64),
                            offsets=BOTTOM,
                            vertex_data=vertex_data,
                            index=index,
                            rgba=(rgba),
                            face_id=1,
                            ao_values=ao_values,
                        )

                    # Back face (face_id 4): sample neighbor at z-1
                    if is_void((x, y, z - 1), nca_tensor, shape):
                        ao_values = get_ao(
                            (x, y, z - 1),
                            nca_tensor,
                            shape,
                            fixed_axis=2,
                            even=True,
                            flip=True,
                        )
                        index = add_face(
                            fixed_axis=2,
                            base_vector=np.array([x, y, z], dtype=np.int64),
                            offsets=BACK,
                            vertex_data=vertex_data,
                            index=index,
                            rgba=(rgba),
                            face_id=4,
                            ao_values=ao_values,
                        )

    return vertex_data[: index + 1]


@njit
def cube_to_uint8(x, y, z):
    return (uint8(x), uint8(y), uint8(z))


@njit
def add_cube_face(
    fixed_axis: np.int64,
    base_vector: np.ndarray,
    offsets: np.ndarray,
    vertex_data: np.ndarray,
    index: np.int64,
) -> np.int64:
    v = []
    k = [i for i in range(3) if i != fixed_axis]
    for i in range(4):
        vertex = base_vector.copy()
        vertex[k[0]] += offsets[i, 0]
        vertex[k[1]] += offsets[i, 1]
        v.append(
            cube_to_uint8(
                vertex[0],
                vertex[1],
                vertex[2],
            )
        )

    # first triangle uses v0, v1, v2,  second triangle uses v0, v2, v3
    return add_data(vertex_data, index, v[0], v[1], v[2], v[0], v[2], v[3])


@njit
def build_cube_mesh(x, y, z):
    vertex_data = np.empty(36 * 3, dtype="uint8")
    index = 0

    # Right face (face_id 2): sample neighbor at x+1
    index = add_cube_face(
        fixed_axis=0,
        base_vector=np.array([x + 1, y, z], dtype=np.int64),
        offsets=RIGHT,
        vertex_data=vertex_data,
        index=index,
    )

    # Top face (face_id 0): sample neighbor at y+1
    index = add_cube_face(
        fixed_axis=1,
        base_vector=np.array([x, y + 1, z], dtype=np.int64),
        offsets=TOP,
        vertex_data=vertex_data,
        index=index,
    )

    # Front face (face_id 5): sample neighbor at z+1
    index = add_cube_face(
        fixed_axis=2,
        base_vector=np.array([x, y, z + 1], dtype=np.int64),
        offsets=FRONT,
        vertex_data=vertex_data,
        index=index,
    )

    # Left face (face_id 3): sample neighbor at x-1
    index = add_cube_face(
        fixed_axis=0,
        base_vector=np.array([x, y, z], dtype=np.int64),
        offsets=LEFT,
        vertex_data=vertex_data,
        index=index,
    )

    # Bottom face (face_id 1): sample neighbor at y-1
    index = add_cube_face(
        fixed_axis=1,
        base_vector=np.array([x, y, z], dtype=np.int64),
        offsets=BOTTOM,
        vertex_data=vertex_data,
        index=index,
    )

    # Back face (face_id 4): sample neighbor at z-1
    index = add_cube_face(
        fixed_axis=2,
        base_vector=np.array([x, y, z], dtype=np.int64),
        offsets=BACK,
        vertex_data=vertex_data,
        index=index,
    )

    return vertex_data[: index + 1]
