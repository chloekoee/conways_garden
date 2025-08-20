from __future__ import annotations
from perspective.camera import Camera
from constants.settings import *
import pygame as pg


class Player(Camera):
    def __init__(self, app, position=PLAYER_POS, yaw=0, pitch=0):
        self.app = app
        super().__init__(position, yaw, pitch, self.app.aspect_ratio)

    def update(self):
        super().update()

    def reset_view(self, position=None):
        """
        Calculates pitch and yaw of camera so player spawns looking at NCA,
        look at direction = (nca.seed - player.postion) translated to global coordinates
        """
        pg.mouse.set_pos((self.app.cx, self.app.cy))
        ## Translation from local nca -> global coordinates
        M = glm.mat4(self.app.scene.nca.get_model_matrix())
        self.position = position if position else (STARTING_POSITION + glm.vec3(M[3]))
        nca_position = glm.vec3(*self.app.scene.nca.seed_position) + glm.vec3(M[3])

        ## Derive vector pointing from eye to reference
        direction: glm.make_vec3 = glm.normalize(nca_position - self.position)

        ## Derive the yaw and pitch
        self.pitch = glm.asin(direction.y)
        self.yaw = glm.atan(direction.z, direction.x)
