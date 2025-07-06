from constants.settings import *
from objects.nca import NCA
from objects.voxel_marker import VoxelMarker
from objects.crosshair import CrossHair
from handlers.voxel_handler import VoxelHandler

import moderngl as mgl


class Scene:
    def __init__(self, app):
        self.app = app
        self.nca = NCA(self.app)
        self.voxel_handler = VoxelHandler(self)
        self.voxel_marker = VoxelMarker(self.voxel_handler)
        self.crosshair = CrossHair(self.app)

    def update(self):
        self.voxel_marker.update()
        self.voxel_handler.update()

    def render(self):
        self.nca.render()
        self.voxel_marker.render()
        self.app.ctx.disable(mgl.CULL_FACE)
        self.crosshair.render()
