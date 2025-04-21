from settings import *
from numba import uint8

"""
Vertex Attributes: [x, y, z, face_id, voxel_id] (each an np.int64 8 --> 1 byte)
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
4: front
5: back
"""

@njit
def to_uint8(x, y, z, voxel_id, face_id, ao_id):
    return uint8(x), uint8(y), uint8(z), uint8(voxel_id), uint8(face_id), uint8(ao_id)

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
def get_ao_corrections(fixed_axis: np.int64, world_voxels, local_voxel_pos, world_voxel_pos, flip = False):
    neighbours = []

    x, y, z = local_voxel_pos
    wx, wy, wz = world_voxel_pos

    col = (fixed_axis + 1)%3 if flip else (fixed_axis + 2)%3
    row = (fixed_axis + 2)%3 if flip else (fixed_axis + 1)%3

    for neighbour in AO_NEIGHBOURHOOD:
        v = [0,0,0]
        v[row] = neighbour[1]
        v[col] = neighbour[0]
        neighbours.append(is_void((x+v[0], y+v[1], z+v[2]), (wx+v[0], wy+v[1],wz+v[2]), world_voxels))

    a = sum([neighbours[7]] + neighbours[0:2])
    b = sum(neighbours[1:4])
    c = sum(neighbours[3:6])
    d = sum(neighbours[5:8])

    return a,b,c,d


@njit
def add_face(fixed_axis: np.int64, base_vector: np.ndarray, is_right_group: bool, vertex_data: np.ndarray, index:np.int64, voxel_id, face_id, ao_values) -> np.int64:
    '''
    Adds vertices for a quad face defined by a fixed axis and a base coordinate.
    
              v0 *---* v1
                /     \
            v3 *-------*  v2
    

    Parameters:
        fixed_axis: The index (0, 1, or 2) of the coordinate (x,y,z) that remains fixed.
        fixed_value: the base coordinate (1,0,0) if adding right face, (0,1,0) if adding top face
        is_right_group: Determines which offset set is used (TODO: figure out mathematical term for this)
        vertex_data: 1D array vertex buffer.
        index: Running index of vertex_data buffer.
        voxel_id: The voxel id to pack into each vertex.
        face_id: The face id (enum) to pack into each vertex.

    The function creates a quad from 4 vertices using offsets for the unfixed axes and
    then breaks the quad into 2 triangles by calling add_data.
    '''
    k = fixed_axis
    offsets = RH_OFFSETS if is_right_group else LH_OFFSET
    packed_vertices = []

    # here the 4 is hardcoded and thats bad coding practice mr chloe 
    for i in range(4):
        vertex = base_vector.copy()
        vertex[(k+1)%3] += offsets[i,0]
        vertex[(k+2)%3] += offsets[i,1]
        packed_vertices.append(to_uint8(vertex[0], vertex[1], vertex[2], voxel_id, face_id, ao_values[i]))

    ## the two triangles forming the face.
    # the first triangle uses v0, v1, v2
    # the second triangle uses v0, v2, v3
    index = add_data(vertex_data, index, packed_vertices[0], packed_vertices[1], packed_vertices[2], packed_vertices[0], packed_vertices[2], packed_vertices[3])
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
                
                ## Calculate global position
                cx, cy, cz = chunk_pos
                wx = x + cx * CHUNK_SIZE
                wy = y + cy * CHUNK_SIZE
                wz = z + cz * CHUNK_SIZE


                ## unique identifier for voxel within chunk
                voxel_id: int = chunk_voxels[x + CHUNK_SIZE * z + CHUNK_AREA * y]
                if not voxel_id:
                    continue

                ## check to see if neighbouring voxels exist for each face of the current voxel
                #  - if they don't then add vertices representing the current voxel face (which is projected
                # 1 unit away from the voxels x,y,z, in that face direction, not projected if bottom, left or back)
                # note that this face is also represented by two triangles

                # right face (face_id 2): x is fixed at x+1.
                if is_void((x+1, y, z), (wx+1, wy, wz), world_voxels):
                    ao_values = get_ao_corrections(0, world_voxels, (x+1, y, z), (wx+1, wy, wz))
                    index = add_face(
                        fixed_axis=0,
                        base_vector=np.array([x+1, y, z], dtype=np.int64),
                        is_right_group=True,
                        vertex_data=vertex_data,
                        index=index,
                        voxel_id=voxel_id,
                        face_id=2,
                        ao_values = ao_values
                    )
                
                # top face (face_id 0): y is fixed at y+1.
                if is_void((x, y+1, z), (wx, wy+1, wz), world_voxels):
                    ao_values = get_ao_corrections(1, world_voxels, (x, y+1, z), (wx, wy+1, wz))

                    index = add_face(
                        fixed_axis=1,
                        base_vector=np.array([x, y+1, z], dtype=np.int64),
                        is_right_group=True,
                        vertex_data=vertex_data,
                        index=index,
                        voxel_id=voxel_id,
                        face_id=0,
                        ao_values = ao_values
                    )

                # front face (face_id 4): z is fixed at z+1.
                if is_void((x, y, z+1), (wx, wy, wz+1), world_voxels):
                    ao_values = get_ao_corrections(2, world_voxels, (x, y+1, z), (wx, wy+1, wz))
                    index = add_face(
                        fixed_axis=2,
                        base_vector=np.array([x, y, z+1], dtype=np.int64),
                        is_right_group=True,
                        vertex_data=vertex_data,
                        index=index,
                        voxel_id=voxel_id,
                        face_id=5,
                        ao_values=ao_values
                    )

                # left face (face_id 3): x is fixed at x.
                if is_void((x-1, y, z), (wx-1, wy, wz), world_voxels):
                    ao_values = get_ao_corrections(0, world_voxels,(x-1, y, z), (wx-1, wy, wz), flip=True)
                    index = add_face(
                        fixed_axis=0,
                        base_vector=np.array([x, y, z], dtype=np.int64),
                        is_right_group=False,
                        vertex_data=vertex_data,
                        index=index,
                        voxel_id=voxel_id,
                        face_id=3,
                        ao_values=ao_values
                    )

                # bottom face (face_id 1): y is fixed at y.
                if is_void((x, y-1, z), (wx, wy-1, wz), world_voxels):
                    ao_values = get_ao_corrections(1, world_voxels,(x, y-1, z), (wx, wy-1, wz))
                    index = add_face(
                        fixed_axis=1,
                        base_vector=np.array([x, y, z], dtype=np.int64),
                        is_right_group=False,
                        vertex_data=vertex_data,
                        index=index,
                        voxel_id=voxel_id,
                        face_id=1,
                        ao_values=ao_values
                    )

                # back face (face_id 5): z is fixed at z.
                if is_void((x, y, z-1), (wx, wy, wz-1), world_voxels):
                    ao_values = get_ao_corrections(2, world_voxels,(x, y, z-1), (wx, wy, wz-1), flip=True)
                    index = add_face(
                        fixed_axis=2,
                        base_vector=np.array([x, y, z], dtype=np.int64),
                        is_right_group=False,
                        vertex_data=vertex_data,
                        index=index,
                        voxel_id=voxel_id,
                        face_id=4,
                        ao_values=ao_values
                    )



    return vertex_data[:index + 1]
