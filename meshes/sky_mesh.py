from meshes.base_mesh import BaseMesh
import numpy as np


class SkyMesh(BaseMesh):
    def __init__(self, app):
        super().__init__()
        self.app = app

        self.ctx = self.app.ctx
        self.program = self.app.shader_program.sky
        self.vbo_format = "2f"
        self.attrs = ("in_position",)  # extra comma to create tuple
        self.vao = self.get_vao()

    def get_vertex_data(self):
        vertex_data = np.array(
            [-1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1], dtype="float32"
        )
        return vertex_data
