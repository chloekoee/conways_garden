from __future__ import annotations
from typing import Optional
import pygame as pg
from camera import Camera
from settings import *


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
        w, h, d = self.app.scene.nca.state.shape[0:3]

        ## Obtain eye and reference vectors
        if self.uninitialized:
            self.position: glm.vec3 = glm.vec3(w * 2, h // 2, d // 2)
        nca_centre: glm.vec3 = glm.vec3(w // 2, h // 2, d // 2)

        ## Derive vector pointing from eye to reference
        direction: glm.make_vec3 = glm.normalize(nca_centre - self.position)

        ## Derive the yaw and pitch
        self.pitch = glm.asin(glm.clamp(direction.y, -1.0, 1.0))
        self.yaw = glm.atan(direction.z, direction.x)
