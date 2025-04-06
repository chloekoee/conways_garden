from settings import *
import moderngl as mgl
from world import World

class Scene:
    def __init__(self, app):
        self.app = app
        self.world = World(self.app)

    def update(self):
        self.world.update()

    def render(self):
        self.world.render()

        ## By disabling face culling, we can see within the chunk (all voxels which are rendered are the ones which face outward to the camera)
        ## instead of just having 3 faces rendered, all their faces are rendered - so faces that point to the outer world are rendered 
        # self.app.ctx.disable(mgl.CULL_FACE)
