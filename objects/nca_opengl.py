from constants.settings import *
from meshes.nca_mesh import NCAMesh
import moderngl as mgl
import numpy as np


class NCA:
    def __init__(self, app):
        self.app = app
        self.compute_shader = app.shader_program.compute

        ## load in seed/initial state
        init_state = np.load(f"tensors/pink_orange_voxel.npy").astype(np.float32)

        self.x, self.y, self.z, self.c = init_state.shape
        self.total_voxels = self.x * self.y * self.z

        ## initialise ssbo for compute shader
        byte_length = self.total_voxels * self.c * 4  ## as using float32 not uint8
        self.buffers = [
            app.ctx.buffer(init_state.tobytes()),  ## current state
            app.ctx.buffer(reserve=byte_length),  ## next state
        ]
        self.buffers[0].bind_to_storage_buffer(0)
        self.buffers[1].bind_to_storage_buffer(1)

        self.compute_shader["X"].value = self.x
        self.compute_shader["Y"].value = self.y
        self.compute_shader["Z"].value = self.z

        self.step = 0
        self.frozen = False
        self.refresh = False

        ## set initial state for mesh generation
        optimised_state = init_state[..., :4]
        optimised_state = np.rint(optimised_state * 255).astype(np.uint8)
        self.state = optimised_state.astype(np.uint8)

        self.m_model = self.get_model_matrix()
        self.mesh: NCAMesh = None

        self.build_mesh()

    def toggle_freeze(self):
        self.frozen = not self.frozen

    def offset(self, x, y, z):
        return (((x * self.y) + y) * self.z + z) * self.c * 4 - 1  ## float byte offset

    def delete_voxel(self, x, y, z):
        curr_i, next_i = (self.step % 2), ((self.step + 1) % 2)
        # partial_voxel = self.state[x,y,z]
        voxel = np.zeros((1, 1, 1, self.c))
        # voxel[...,:4] = partial_voxel
        ## writing out to both, incase next_state has already been computed
        current_state = self.buffers[curr_i]
        next_state = self.buffers[next_i]

        idx = self.offset(x, y, z)
        current_state.write(bytes(voxel), offset=idx)
        next_state.write(bytes(voxel), offset=idx)
        self.build_mesh()

    def take_step(self):
        curr_i, next_i = (self.step % 2), ((self.step + 1) % 2)

        # bind ssbos into global image slots
        self.buffers[0].bind_to_storage_buffer(curr_i)
        self.buffers[1].bind_to_storage_buffer(next_i)

        ## dispatch enough groups to cover the full (self.x,self.y,self.z) volume
        gx = (self.x + 7) // 8
        gy = (self.y + 7) // 8
        gz = (self.z + 7) // 8
        self.app.shader_program.compute.run(gx, gy, gz)

        ## read back for mesh building
        raw = bytearray(self.buffers[next_i].read())
        state = np.frombuffer(
            raw, dtype=np.float32
        ).reshape(  ## TODO: why did we need that reshape
            (self.x, self.y, self.z, 16)
        )
        ##  crop off the hidden channels, convert all values to 255 for efficient rendering
        ## normalise all rgba values to integers between 0 and 255 for memory
        state = state[..., :4]
        state = np.rint(state * 255).astype(np.uint8)
        self.state = state.astype(np.uint8)
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

    def set_refresh(self):
        self.refresh = True
