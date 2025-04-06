from settings import *
from numba import uint8

"""
Vertex Attributes: [x, y, z, face_id, voxel_id] (each an int 8 --> 1 byte)
At any given angle we can only see three faces of a voxel
18 vertices forms 6 triangles which form three faces of one voxel.

Therefore each VISIBLE voxel will have at most 3 faces which must be rendered 
We construct vertex data - which is our VBO storing the vertices that constitute the 
chunks visible outward facing faces 

QUESTION: check if vertex data contains indices for all vertices in the chunk or
just the outward facing ones 

ANSWER: vertex_data contains all vertices which we render in no particular order, compressed to save 
buffer space

The maximum size of vertex_data = np.empty(CHUNK_VOL * 18 * format_size, dtype='uint8')
CHUNK_VOL = number of voxels in chunk
18 = number of vertices that form the visible 3 faces
format_size = number of floats needed to describe a vertex's attributes: [x, y, z, face_id, voxel_id]

FACE ID is an enum:
0: top
1: bottom
2: right
3: left
4: back
5: front
"""

@njit
def to_uint8(x, y, z, voxel_id, face_id):
    return uint8(x), uint8(y), uint8(z), uint8(voxel_id), uint8(face_id)

@njit
def get_chunk_index(world_voxel_pos):
    """
    At the world's boundary, we may return -1
    """
    wx, wy, wz = world_voxel_pos
    cx = wx // CHUNK_SIZE
    cy = wy // CHUNK_SIZE
    cz = wz // CHUNK_SIZE

    if not (0 <= cx < WORLD_W and 0 <= cy < WORLD_H and 0 <= cz < WORLD_D):
        return -1

    index = cx + WORLD_W * cz + WORLD_AREA * cy
    return index

@njit
def is_void(local_voxel_pos, world_voxel_pos, world_voxels):
    chunk_index = get_chunk_index(world_voxel_pos)
    ## when checking at the world's boundary, these voxels are always void so should always render their in world neighbours
    if chunk_index == -1: 
        return False
    chunk_voxels = world_voxels[chunk_index]

    ## obtaining the voxel index relative to within the chunk
    x, y, z = local_voxel_pos
    voxel_index = x % CHUNK_SIZE + z % CHUNK_SIZE * CHUNK_SIZE + y % CHUNK_SIZE * CHUNK_AREA

    if chunk_voxels[voxel_index]:
        return False

    return True

@njit
def add_data(vertex_data, index, *vertices):
    """
    QUESTION: Why does this not multiply by 5*18 (the format size * number of vertices) it seems
    to add vertex data at vertex_data[vertex_id] rather than vertex_data[vertex_id*format_data]
    *vertices: list of vertex data tuples holding the (x,y,z,voxel_id, face_id)

    ANSWER: The index which we insert the 18 vertices into is just a running total of vertices we have
    inserted. The vertex buffer contains these vertices in no particular order - just compressed
    as much as possible in order to preserve space.
    """
    for vertex in vertices:
        for attr in vertex:
            vertex_data[index] = attr
            index += 1

    return index

@njit
def build_chunk_mesh(chunk_voxels, format_size, chunk_pos, world_voxels):
    """
    Constructs hollow representation of the chunk, returning vertex_data,
    a 1D array of all vertex information (in x, y, z, voxel_id, face_id)

    An inefficiency we optimise for is rendering faces of voxels within chunk A, when chunk A is adjacent to chunk B and thus
    these voxels will never be visible to the camera 
    """

    ## Initialise 1D array for vertices holding vertex data in format: x, y, z, voxel_id, face_id
    # with the maximum amount of space (each voxel has 3 faces visible)
    vertex_data = np.empty(CHUNK_VOL * 18 * format_size, dtype="uint8")
    index = 0

    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):

                ## TODO: WHAT IS THIS
                cx, cy, cz = chunk_pos
                wx = x + cx * CHUNK_SIZE
                wy = y + cy * CHUNK_SIZE
                wz = z + cz * CHUNK_SIZE


                ## unique identifier for voxel within chunk
                voxel_id = chunk_voxels[x + CHUNK_SIZE * z + CHUNK_AREA * y]
                if not voxel_id:
                    continue

                ## check to see if neighbouring voxels exist for each face of the current voxel
                #  - if they don't then add vertices representing the current voxel face (which is projected
                # 1 unit away from the voxels x,y,z, in that face direction, not projected if bottom, left or back)
                # note that this face is also represented by two triangles


                # top face
                if is_void((x, y + 1, z), (wx, wy + 1, wz), world_voxels):
                    # format: x, y, z, voxel_id, face_id
                    v0 = to_uint8(x    , y + 1, z    , voxel_id, 0)
                    v1 = to_uint8(x + 1, y + 1, z    , voxel_id, 0)
                    v2 = to_uint8(x + 1, y + 1, z + 1, voxel_id, 0)
                    v3 = to_uint8(x    , y + 1, z + 1, voxel_id, 0)

                    index = add_data(vertex_data, index, v0, v3, v2, v0, v2, v1)

                # bottom face
                if is_void((x, y - 1, z), (wx, wy - 1, wz), world_voxels):

                    v0 = to_uint8(x    , y, z    , voxel_id, 1)
                    v1 = to_uint8(x + 1, y, z    , voxel_id, 1)
                    v2 = to_uint8(x + 1, y, z + 1, voxel_id, 1)
                    v3 = to_uint8(x    , y, z + 1, voxel_id, 1)

                    index = add_data(vertex_data, index, v0, v2, v3, v0, v1, v2)

                # right face
                if is_void((x + 1, y, z), (wx + 1, wy, wz), world_voxels):

                    v0 = to_uint8(x + 1, y    , z    , voxel_id, 2)
                    v1 = to_uint8(x + 1, y + 1, z    , voxel_id, 2)
                    v2 = to_uint8(x + 1, y + 1, z + 1, voxel_id, 2)
                    v3 = to_uint8(x + 1, y    , z + 1, voxel_id, 2)

                    index = add_data(vertex_data, index, v0, v1, v2, v0, v2, v3)

                # left face
                if is_void((x - 1, y, z), (wx - 1, wy, wz), world_voxels):

                    v0 = to_uint8(x, y    , z    , voxel_id, 3)
                    v1 = to_uint8(x, y + 1, z    , voxel_id, 3)
                    v2 = to_uint8(x, y + 1, z + 1, voxel_id, 3)
                    v3 = to_uint8(x, y    , z + 1, voxel_id, 3)

                    index = add_data(vertex_data, index, v0, v2, v1, v0, v3, v2)

                # back face
                if is_void((x, y, z - 1), (wx, wy, wz - 1), world_voxels):

                    v0 = to_uint8(x,     y,     z, voxel_id, 4)
                    v1 = to_uint8(x,     y + 1, z, voxel_id, 4)
                    v2 = to_uint8(x + 1, y + 1, z, voxel_id, 4)
                    v3 = to_uint8(x + 1, y,     z, voxel_id, 4)

                    index = add_data(vertex_data, index, v0, v1, v2, v0, v2, v3)

                # front face
                if is_void((x, y, z + 1), (wx, wy, wz + 1), world_voxels):

                    v0 = to_uint8(x    , y    , z + 1, voxel_id, 5)
                    v1 = to_uint8(x    , y + 1, z + 1, voxel_id, 5)
                    v2 = to_uint8(x + 1, y + 1, z + 1, voxel_id, 5)
                    v3 = to_uint8(x + 1, y    , z + 1, voxel_id, 5)

                    index = add_data(vertex_data, index, v0, v2, v1, v0, v3, v2)

    return vertex_data[:index + 1]
