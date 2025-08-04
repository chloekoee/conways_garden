from constants.settings import *
from meshes.nca_mesh import NCAMesh
import numpy as np
from handlers.metal_handler import MetalHandler


class NCA:
    def __init__(self, app, uses_learnable_perception=False, nca_name="nyan_sakura_masked"):
        self.app = app
        self.metal = MetalHandler(nca_name, uses_learnable_perception)
        self.x, self.y, self.z, self.c = self.metal.get_shape()
        self.total_voxels = self.x * self.y * self.z
        self.step = 0
        self.frozen = False
        self.refresh = False

        ## set initial state for mesh generation s
        optimised_state = self.metal.get_seed()[..., :4]
        optimised_state = np.rint(optimised_state * 255).astype(np.uint8)
        self.state = optimised_state.astype(np.uint8)

        position = np.where(self.state[..., 3] > 0)
        self.seed_position = position[0][0], position[1][0], position[2][0]

        self.mesh: NCAMesh = None

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
        self.metal.overwrite_voxel(x, y, z)
        self.build_mesh()

    def take_step(self):
        next_state = self.metal.compute_next_state()
        next_state = next_state[..., :4]
        next_state = np.rint(next_state * 255)
        next_state[..., :] = np.clip(next_state[..., :], 0, 255)

        self.state = next_state.astype(np.uint8)
        self.build_mesh()
        self.step += 1

    def build_mesh(self):
        self.mesh = NCAMesh(self)

    def render(self):
        self.set_uniform()
        self.mesh.render()

    def set_uniform(self):
        self.mesh.program["m_model"].write(self.m_model)

    def set_refresh(self):
        self.refresh = True
