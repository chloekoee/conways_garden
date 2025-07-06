from constants.settings import *
from meshes.cube_mesh import CubeMesh


class VoxelMarker:
    def __init__(self, voxel_handler):
        self.app = voxel_handler.app
        self.handler = voxel_handler
        self.position = glm.vec3(0)
        self.mesh = CubeMesh(self)

    def update(self):
        if self.handler.target_found and self.position != self.handler.voxel_position:
            print(
                f"changing target from {self.position} to {self.handler.voxel_position}"
            )
            # self.handler.voxel_position
            self.position = self.handler.voxel_position
            self.mesh.rebuild()

    def render(self):
        # if self.handler.target_found:
        self.mesh.render()
