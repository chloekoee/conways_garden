from constants.settings import *
from meshes.island_mesh import IslandMesh


class Island:
    def __init__(self, app):
        self.app = app
        self.state = np.load(f"islands/island.npy")
        self.state[..., :4] = np.rint(self.state[..., :4] * 255).astype(np.uint8)
        self.shape = self.state.shape[:3]
        position = np.where(self.state[..., 3] > 0)
        self.seed_position = position[0][0], position[1][0], position[2][0]
        self.mesh: IslandMesh = None
        self.build_mesh()

    def build_mesh(self):
        self.mesh = IslandMesh(self)

    def render(self):
        self.set_uniform()
        self.mesh.render()

    def set_uniform(self):
        self.mesh.program["m_model"].write(self.m_model)

    def set_model_matrix(self, model):
        self.m_model = model

    def get_model_matrix(self):
        return self.m_model

    def get_shape(self):
        return self.shape
