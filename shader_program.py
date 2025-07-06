from constants.settings import *
from constants.sobel import *


class ShaderProgram:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.player = app.player
        self.controls = app.controls
        self.nca = self.get_program(shader_name="nca/nca")
        self.crosshair = self.get_program(shader_name="crosshair/crosshair")

        # self.compute = self.get_compute_shader(shader_name="compute")
        self.set_uniforms_on_init()

    def set_uniforms_on_init(self):
        self.nca["m_proj"].write(self.player.m_proj)
        self.nca["m_model"].write(glm.mat4())
        self.nca["face_textures"].value = (0, 1, 2, 3, 4, 5)

        # self.compute["sobelX"].value    = SOBEL_X
        # self.compute["sobelY"].value    = SOBEL_Y
        # self.compute["sobelZ"].value    = SOBEL_Z
        # self.compute["identity"].value = IDENTITY

    def update(self):
        self.nca["m_view"].write(self.player.m_view)

    def get_program(self, shader_name):
        with open(f"shaders/{shader_name}.vert") as file:
            vertex_shader = file.read()

        with open(f"shaders/{shader_name}.frag") as file:
            fragment_shader = file.read()

        program = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )
        return program

    def get_compute_shader(self, shader_name):
        with open(f"shaders/{shader_name}.glsl", encoding="utf-8") as file:
            compute_shader = file.read()

        compute_program = self.ctx.compute_shader(compute_shader)
        return compute_program
