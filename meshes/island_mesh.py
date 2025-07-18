from meshes.base_mesh import BaseMesh
from meshes.mesh_builder import build_nca_mesh


class IslandMesh(BaseMesh):
    def __init__(self, island):
        super().__init__()
        self.app = island.app
        self.island = island
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
            nca_tensor=self.island.state,
            format_size=self.format_size,
        )
        return mesh

    def get_vao(self):
        vertex_data = self.get_vertex_data()
        vbo = self.ctx.buffer(vertex_data)
        vao = self.ctx.vertex_array(
            self.program, [(vbo, self.vbo_format, *self.attrs)], skip_errors=True
        )
        return vao

    def render(self):
        self.vao.render()
