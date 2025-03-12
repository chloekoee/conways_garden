from settings import *
import moderngl as mgl
from world_objects.chunk import Chunk


class Scene:
    def __init__(self, app):
        self.app = app
        self.chunk = Chunk(self.app)

    def update(self):
        pass

    def render(self):
        self.chunk.render()

        ## By disabling face culling, we can see within the chunk (all voxels which are rendered are the ones which face outward to the camera)
        ## this doesn't change - but instead of just having 3 faces rendered, all their faces are rendered ???
        self.app.ctx.disable(mgl.CULL_FACE)
