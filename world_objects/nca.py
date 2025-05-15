from settings import *
from nca_voxel_handler import NCAVoxelHandler
from meshes.nca_mesh import NCAMesh


class NCA:
    def __init__(self, app):
        self.app = app
        self.voxel_handler = NCAVoxelHandler(self)

        ## TODO: switch to dynamic loading of voxels
        simulation_tensor = np.load(f"tensors/pink_orange_voxel.npy")

        # Normalise all rgba values to integers between 0 and 255 for memory
        simulation_tensor[..., :4] = np.rint(simulation_tensor[..., :4] * 255).astype(
            np.uint8
        )
        self.simulation = simulation_tensor
        self.step = 0
        self.lifespan = self.simulation.shape[0]
        self.state = self.simulation[self.step]
        self.frozen = False

        self.m_model = self.get_model_matrix()
        self.mesh: NCAMesh = None

        self.build_mesh()

    def update(self):
        self.voxel_handler.update()

    def toggle_freeze(self):
        self.frozen = not self.frozen

    def take_step(self):
        self.step = (self.step + 1) % self.lifespan
        self.state = self.simulation[self.step, :, :, :, :]
        self.build_mesh()

    def build_mesh(self):
        self.mesh = NCAMesh(self)

    def render(self):
        self.set_uniform()
        self.mesh.render()

    def set_uniform(self):
        self.mesh.program["m_model"].write(self.m_model)

    def get_model_matrix(self):
        return glm.mat4(1.0)
