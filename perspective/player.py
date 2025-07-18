from __future__ import annotations
from perspective.camera import Camera
from constants.settings import *


class Player(Camera):
    def __init__(self, app, position=PLAYER_POS, yaw=0, pitch=0):
        self.app = app
        super().__init__(position, yaw, pitch, self.app.aspect_ratio)
        self.uninitialized = True

    def update(self):
        if self.uninitialized is True:
            self.reset_view()
            self.uninitialized = False
        super().update()

    def reset_view(self) -> None:
        sx, sy, sz = self.app.scene.nca.seed_position
        ## Obtain eye and reference vectors
        if self.uninitialized:
            M = glm.mat4(
                self.app.scene.nca.get_model_matrix()
            )  # mat3 drops translation row/column
            self.position = glm.vec3(M[3])  ## want the translation row
            # TODO: put these into constants
            self.position.x + 50
            self.position.y + 20
            self.position.z = 50
        nca_seed_position: glm.vec3 = glm.vec3(sx, sy, sz)

        ## Derive vector pointing from eye to reference
        direction: glm.make_vec3 = glm.normalize(nca_seed_position - self.position)

        ## Derive the yaw and pitch
        self.pitch = glm.asin(glm.clamp(direction.y, -1.0, 1.0))
        self.yaw = glm.atan(direction.z, direction.x)
