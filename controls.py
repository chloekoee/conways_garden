import pygame as pg
from settings import PLAYER_SPEED, MOUSE_SENSITIVITY, SECONDS_PER_STEP


class Controls:
    def __init__(self, app):
        self.app = app
        self.prev_keys = pg.key.get_pressed()

    def poll(self):
        """
        - Pull in all events once
        - Capture mouse delta
        - Capture key state
        """
        self.events = pg.event.get()
        self.mouse_dx, self.mouse_dy = pg.mouse.get_rel()
        self.keys = pg.key.get_pressed()

    def handle_global(self):
        """
        Things like quit or pause that live at the app level
        """
        for e in self.events:
            if e.type == pg.QUIT:
                self.app.is_running = False
            elif e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                self.app.paused = not self.app.paused
                pg.mouse.set_visible(self.app.paused)
                pg.event.set_grab(not self.app.paused)

    def apply(self, player):
        """
        Dispatch both continuous and discrete actions on the Player (or scene)
        """
        # 1) Discrete events
        for e in self.events:
            if e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                player.app.scene.voxel_handler.remove_voxel()

        # 2) “Toggle‐on‐hold” actions
        if self.keys[pg.K_k]:
            player.app.scene.nca.toggle_freeze()

        # 3) Mouse‐look
        if self.mouse_dx:
            player.rotate_yaw(self.mouse_dx * MOUSE_SENSITIVITY)
        if self.mouse_dy:
            player.rotate_pitch(self.mouse_dy * MOUSE_SENSITIVITY)

        # 4) Keyboard movement
        vel = PLAYER_SPEED * player.app.delta_time
        if self.keys[pg.K_w]:
            player.move_forward(vel)
        if self.keys[pg.K_s]:
            player.move_back(vel)
        if self.keys[pg.K_d]:
            player.move_right(vel)
        if self.keys[pg.K_a]:
            player.move_left(vel)
        if self.keys[pg.K_SPACE]:
            player.move_up(vel)
        if self.keys[pg.K_LSHIFT]:
            player.move_down(vel)
