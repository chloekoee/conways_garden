from settings import *
from world_objects.nca import NCA
from world_objects.voxel_marker import VoxelMarker
import moderngl as mgl


class Scene:
    def __init__(self, app):
        self.app = app
        self.nca = NCA(self.app)
        self.voxel_marker = VoxelMarker(self.nca.voxel_handler)

    def update(self):
        self.nca.update()
        self.voxel_marker.update()

    def render(self):
        self.nca.render()
        self.voxel_marker.render()

        ## By disabling face culling, we can see within the chunk (all voxels which are rendered are the ones which face outward to the camera)
        ## instead of just having 3 faces rendered, all their faces are rendered - so faces that point to the outer world are rendered
        self.app.ctx.disable(mgl.CULL_FACE)
