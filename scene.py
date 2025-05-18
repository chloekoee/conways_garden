from settings import *
from world_objects.nca import NCA
from world_objects.voxel_marker import VoxelMarker
from voxel_handler import VoxelHandler

import moderngl as mgl


class Scene:
    def __init__(self, app):
        self.app = app
        self.nca = NCA(self.app)
        self.voxel_handler = VoxelHandler(self)
        self.voxel_marker = VoxelMarker(self.voxel_handler)

    def update(self):
        self.voxel_marker.update()
        self.voxel_handler.update()

    def render(self):
        self.nca.render()
        self.voxel_marker.render()
        self.app.ctx.disable(mgl.CULL_FACE)
