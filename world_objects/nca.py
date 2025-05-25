from settings import *
from meshes.nca_mesh import NCAMesh
import moderngl as mgl
import numpy as np


class NCA:
    def __init__(self, app):
        self.app = app
        self.compute_shader = app.shader_program.compute

        ## load in seed/initial state
        seed_state = np.load(f"tensors/pink_orange_voxel.npy")

        ## normalise all rgba values to integers between 0 and 255 for memory
        seed_state[..., :4] = np.rint(seed_state[..., :4] * 255).astype(np.uint8)
        seed_state = seed_state.astype(np.uint8)

        ## set initial state
        self.state = seed_state
        self.x, self.y, self.z, _ = seed_state.shape

        ## initialise textures for the current and next state
        self.current_state = self.app.ctx.texture3d(
            size=(self.x, self.y, self.z),
            components=4,
            dtype="u1",
            data=seed_state.tobytes(),
        )

        self.next_state = self.app.ctx.texture3d(
            size=(self.x, self.y, self.z),
            components=4,
            dtype="u1",
            data=None,
        )

        ## set the textures interpolation to nearest neighbour (no linear interpolation)
        self.current_state.filter = (mgl.NEAREST, mgl.NEAREST)
        self.next_state.filter = (mgl.NEAREST, mgl.NEAREST)

        ## to ping pong between each texture
        self.state_textures = [self.current_state, self.next_state]

        self.step = 0
        self.frozen = False

        self.m_model = self.get_model_matrix()
        self.mesh: NCAMesh = None

        self.build_mesh()

    def toggle_freeze(self):
        self.frozen = not self.frozen
    def take_step(self):
        curr_i, next_i = (self.step % 2), ((self.step + 1) % 2)

        # bind textures into global image slots
        self.state_textures[curr_i].use(
            location=6
        )  ## current state sampler3D (shares namespace with textures)
        self.state_textures[next_i].bind_to_image(
            0, read=False, write=True
        )  ## next state image3D

        ## dispatch enough groups to cover the full (self.x,self.y,self.z) volume
        gx = (self.x + 7) // 8
        gy = (self.y + 7) // 8
        gz = (self.z + 7) // 8
        self.app.shader_program.compute.run(gx, gy, gz)

        ## read back for mesh building
        raw = self.state_textures[next_i].read()
        self.state = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.x, self.y, self.z, 4)
        )

        ## generate mesh for next state
        self.build_mesh()
        self.step += 1

    def build_mesh(self):
        self.mesh = NCAMesh(self)

    def render(self):
        self.set_uniform()
        self.mesh.render()

    def set_uniform(self):
        self.mesh.program["m_model"].write(self.m_model)

    def get_model_matrix(self):
        return glm.mat4(1.0)
