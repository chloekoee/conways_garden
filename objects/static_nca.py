from constants.settings import *
from meshes.nca_mesh import NCAMesh
import numpy as np


class StaticNCA:
    def __init__(self, app, nca_name="cherry_blossom"):
        self.app = app
        self.frames = np.load(f"state/static/{nca_name}.npy")

        self.lifespan, self.x, self.y, self.z, self.c = self.frames.shape
        self.total_voxels = self.x * self.y * self.z

        self.step = 0
        self.frozen = False
        self.refresh = False

        optimised_state = self.frames[0]
        optimised_state = np.rint(optimised_state * 255).astype(np.uint8)
        self.state = optimised_state.astype(np.uint8)

        position = np.where(self.state[..., 3] > 0)
        x, y, z = [len(s) // 2 for s in position]
        self.seed_position = position[0][x], position[1][y], position[2][z]

        self.mesh: NCAMesh = None

        self.direction = 1
        self.build_mesh()

    def get_shape(self):
        return self.x, self.y, self.z

    def set_model_matrix(self, model):
        self.m_model = model
        self.inv_m_model = glm.inverse(model)

    def get_model_matrix(self):
        return self.m_model

    def toggle_freeze(self):
        self.frozen = not self.frozen

    def delete_voxel(self, x, y, z):
        self.state[x, y, z, 3] = 0
        self.frames[self.step, x, y, z, 3] = 0
        self.build_mesh()

    def take_step(self):
        self.step += self.direction
        if self.step == self.lifespan - 1:
            self.direction = -1
        elif self.step == 0:
            self.direction = 1

        next_state = self.frames[self.step]
        self.state = np.rint(next_state * 255).astype(np.uint8)
        self.build_mesh()

    def build_mesh(self):
        self.mesh = NCAMesh(self)

    def render(self):
        self.set_uniform()
        self.mesh.render()

    def set_uniform(self):
        self.mesh.program["m_model"].write(self.m_model)

    def set_refresh(self):
        self.refresh = True
