from constants.settings import *
from meshes.crosshair_mesh import CrosshairMesh


class CrossHair:
    def __init__(self, app):
        self.app = app
        self.mesh = CrosshairMesh(self.app)

    def render(self):
        self.mesh.render()
