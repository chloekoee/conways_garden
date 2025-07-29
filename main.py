from controls import Controls
from constants.settings import *
import moderngl as mgl
import pygame as pg
import sys
from perspective.player import Player
from scene import Scene
from shader_program import ShaderProgram
from textures import Textures


class Engine:
    def __init__(self):
        pg.init()

        ## Dynamically calculate display size
        info = pg.display.Info()
        screen_w, screen_h = info.current_w, info.current_h
        self.resolution = glm.vec2(int(screen_w * 1.0), int(screen_h * 1.0))
        self.cx, self.cy = screen_w // 2, screen_h // 2
        self.aspect_ratio = self.resolution.x / self.resolution.y

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(
            pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE
        )
        ## Set value of 24 bits for depth buffer
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)

        ## Create open GL context
        pg.display.set_mode(self.resolution, flags=pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.BLEND | mgl.CULL_FACE)
        self.ctx.depth_func = "<="
        ## Enable automatic garbage collection of unused Open GL objects
        self.ctx.gc_mode = "auto"

        ## Time keeping
        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.time = 0
        self.time_since_last_step = 0

        cx, cy = self.resolution.x // 2, self.resolution.y // 2
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        pg.mouse.set_pos((self.cx, self.cy))

        self.is_running = True
        self.paused = False
        self.on_init()

        pg.display.set_caption(f"{self.scene.nca.step :.0f}")

    def on_init(self):
        self.textures = Textures(self)
        self.controls = Controls(self)
        self.player = Player(self)
        self.shader_program = ShaderProgram(self)
        self.scene = Scene(self)

    def update(self):
        self.shader_program.update()
        self.scene.update()
        self.player.update()

        self.delta_time = self.clock.tick()
        self.time = pg.time.get_ticks() * 0.001

        if not self.scene.nca.frozen:
            if self.time - self.time_since_last_step > SECONDS_PER_STEP:

                self.time_since_last_step = self.time
                self.scene.nca.take_step()
                pg.display.set_caption(f"{self.scene.nca.step :.0f}")

    def on_render(self):
        self.ctx.clear()
        self.scene.render()
        pg.display.flip()

    def run(self):
        while self.is_running:
            self.controls.poll()
            self.controls.handle_global()
            if not self.paused:
                self.controls.apply(self.player)
                self.update()
            self.on_render()
        pg.quit()
        sys.exit()


if __name__ == "__main__":
    app = Engine()
    app.run()
