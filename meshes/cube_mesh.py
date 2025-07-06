from constants.settings import *
from meshes.base_mesh import BaseMesh
from meshes.mesh_builder import build_cube_mesh


class CubeMesh(BaseMesh):
    def __init__(self, voxel_marker):
        super().__init__()
        self.app = voxel_marker.app
        self.voxel_marker = voxel_marker
        self.ctx = self.app.ctx
        self.program = self.app.shader_program.voxel_marker

        self.vbo_format = "3u1"
        self.attrs = ("in_position",)
        self.vao = self.get_vao()

    def rebuild(self):
        self.vao = self.get_vao()

    def get_vertex_data(self):
        (
            x,
            y,
            z,
        ) = self.voxel_marker.position
        mesh = build_cube_mesh(x, y, z)
        return mesh
