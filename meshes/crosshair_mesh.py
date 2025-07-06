from constants.settings import *
from meshes.base_mesh import BaseMesh
import moderngl


class CrosshairMesh(BaseMesh):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.ctx = self.app.ctx
        self.program = self.app.shader_program.crosshair

        self.vbo_format = "2f2"
        self.attrs = ("cross_position",)
        self.vao = self.get_vao()

    def render(self):
        self.vao.render(mode=moderngl.LINES)

    def get_vertex_data(self):

        vertices = [
            (-0.01, 0.0),
            (0.01, 0.0),
            (0.0, -0.02),
            (0.0, 0.02),
        ]

        data = np.array(vertices, dtype=np.float16)
        return data
