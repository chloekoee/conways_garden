from constants.settings import *
from objects.nca import NCA
from objects.crosshair import CrossHair
from handlers.voxel_handler import VoxelHandler

import moderngl as mgl


class Scene:
    def __init__(self, app):
        self.app = app
        self.nca = NCA(self.app)
        self.voxel_handler = VoxelHandler(self)
        self.crosshair = CrossHair(self.app)

    def update(self):
        self.voxel_handler.update()

    def render(self):
        self.nca.render()
        self.crosshair.render()
