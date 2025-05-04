from settings import *
from meshes.chunk_mesh import ChunkMesh

"""
Class holding the 3D renderable object 
Now comprising of a single 1-unit sized ube
"""


class Chunk:
    def __init__(self, world, position):
        self.app = world.app
        self.world = world
        self.position = position
        self.m_model = self.get_model_matrix()
        self.voxels: np.array = self.build_voxels()
        self.mesh: ChunkMesh = None
        self.is_empty = True

    def get_model_matrix(self):
        """
        Gets the model matrix: where the 1x1x1 cube is relative to the entire world
        """
        return glm.translate(glm.mat4(), glm.vec3(self.position) * CHUNK_SIZE)

    def build_mesh(self):
        self.mesh = ChunkMesh(self)

    def set_uniform(self):
        ## uploads model matrix as a uniform variable in GPU memory - to be used by the shaders
        self.mesh.program["m_model"].write(self.m_model)

    def render(self):
        if not self.is_empty:
            self.set_uniform()
            self.mesh.render()

    def build_voxels(self):
        """
        TODO: Why does this require several voxels
        """
        # empty chunk
        voxels = np.zeros(CHUNK_VOL, dtype="uint8")

        cx, cy, cz = glm.ivec3(self.position) * CHUNK_SIZE

        # fill chunk
        for x in range(CHUNK_SIZE):
            wx = x + cx
            for z in range(CHUNK_SIZE):
                wz = z + cz
                world_height = int(
                    glm.simplex(glm.vec2(wx, wz) * 0.01) * CHUNK_SIZE + CHUNK_SIZE
                )
                local_height = min(world_height - cy, CHUNK_SIZE)

                for y in range(local_height):
                    wy = y + cy
                    voxels[x + CHUNK_SIZE * z + CHUNK_AREA * y] = wy + 1

        if np.any(voxels):
            self.is_empty = False

        return voxels
