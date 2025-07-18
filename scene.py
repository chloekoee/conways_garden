from constants.settings import *
from objects.nca import NCA
from objects.island import Island
from objects.crosshair import CrossHair
from handlers.voxel_handler import VoxelHandler


class Scene:
    def __init__(self, app):
        self.app = app
        self.nca = NCA(self.app)
        self.voxel_handler = VoxelHandler(self)
        self.island = Island(self.app)
        self.crosshair = CrossHair(self.app)
        self.set_model_matrices()

    def update(self):
        self.voxel_handler.update()

    def render(self):
        self.nca.render()
        self.crosshair.render()
        self.island.render()

    def set_model_matrices(self):
        ix, iy, iz = self.island.get_shape()
        nx, ny, nz = 10, 10, 10
        nx, ny, nz = self.nca.get_shape()

        offset = glm.vec3(ix // 2 - nx // 2, iy, iz // 2 - nz // 2)

        island_origin = glm.vec3(0, 0, 0)
        nca_origin = offset + island_origin

        self.island.set_model_matrix(glm.translate(glm.mat4(), island_origin))
        self.nca.set_model_matrix(glm.translate(glm.mat4(), nca_origin))
