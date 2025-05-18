from settings import *
from meshes.cube_mesh import CubeMesh


class VoxelMarker:
    def __init__(self, voxel_handler):
        self.app = voxel_handler.app
        self.handler = voxel_handler
        self.position = glm.vec3(0)
        self.mesh = CubeMesh(self.app)

    def update(self):
        if self.handler.target_found:
            self.position = self.handler.voxel_position

    def render(self):
        if self.handler.target_found:
            self.mesh.render()
