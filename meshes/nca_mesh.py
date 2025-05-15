from meshes.base_mesh import BaseMesh
from meshes.nca_builder import build_nca_mesh


class NCAMesh(BaseMesh):
    def __init__(self, nca):
        super().__init__()
        self.app = nca.app
        self.nca = nca
        self.ctx = self.app.ctx
        self.program = self.app.shader_program.nca

        self.vbo_format = "3u1 4u1 1u1 1u1"
        self.format_size = 3 + 4 + 1 + 1
        self.attrs = ("in_position", "rgba", "face_id", "ao_id")
        self.vao = self.get_vao()

    def rebuild(self):
        self.vao = self.get_vao()

    def get_vertex_data(self):
        mesh = build_nca_mesh(
            nca_tensor=self.nca.state,
            format_size=self.format_size,
        )
        return mesh
