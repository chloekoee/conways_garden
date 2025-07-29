from constants.settings import *
from meshes.sky_mesh import SkyMesh


class Sky:
    def __init__(self, app):
        self.app = app
        self.build_mesh()

    def build_mesh(self):
        self.mesh = SkyMesh(self.app)

    def render(self):
        self.mesh.render()
