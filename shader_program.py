from settings import *
from constants.sobel import *
import numpy as np
try:
    import objc  # type: ignore
    import Metal  # type: ignore
except ImportError:
    Metal = None


class ShaderProgram:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.player = app.player
        self.nca = self.get_program(shader_name="nca")
        self.voxel_marker = self.get_program(shader_name="voxel_marker")
        self.compute = self.get_compute_shader(shader_name="compute")
        self.set_uniforms_on_init()

        # if Metal is available, set up the compute pipeline
        if Metal is not None:
            self._setup_metal_compute()

    def set_uniforms_on_init(self):
        self.nca["m_proj"].write(self.player.m_proj)
        self.nca["m_model"].write(glm.mat4())
        self.nca["face_textures"].value = (0, 1, 2, 3, 4, 5)

        self.voxel_marker["m_proj"].write(self.player.m_proj)
        self.voxel_marker["u_texture_0"] = 0

        # self.compute["sobelX"].value    = SOBEL_X
        # self.compute["sobelY"].value    = SOBEL_Y
        # self.compute["sobelZ"].value    = SOBEL_Z
        self.compute["identity"].value = IDENTITY

    def update(self):
        self.nca["m_view"].write(self.player.m_view)
        self.voxel_marker["m_view"].write(self.player.m_view)

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

    def _setup_metal_compute(self):
        """Initialize PyObjC Metal compute pipeline from precompiled metallib."""
        from constants.sobel import SOBEL_X, SOBEL_Y, SOBEL_Z, IDENTITY
        # combine kernels
        kernels = np.array(SOBEL_X + SOBEL_Y + SOBEL_Z + IDENTITY, dtype=np.float32)
        dims = np.array([self.compute["X"].value,
                         self.compute["Y"].value,
                         self.compute["Z"].value], dtype=np.uint32)

        # device and pipeline
        self.metal_device = Metal.MTLCreateSystemDefaultDevice()
        lib, _ = self.metal_device.newLibraryWithFile_error_("metal/compute.metallib", None)
        fn = lib.newFunctionWithName_("compute_main")
        self._metal_pipeline, _ = self.metal_device.newComputePipelineStateWithFunction_error_(fn, None)
        self._metal_queue = self.metal_device.newCommandQueue()
        self._metal_kernels = kernels
        self._metal_dims = dims

    def run_metal_compute(self, current_state: bytes) -> bytes:
        """Run the Metal compute shader, return raw bytes of next state."""
        if Metal is None:
            raise RuntimeError("Metal compute not available")

        # create buffers
        buf_in = self.metal_device.newBufferWithBytes_length_options_(current_state, len(current_state), 0)
        buf_out = self.metal_device.newBufferWithLength_options_(len(current_state), 0)
        buf_kern = self.metal_device.newBufferWithBytes_length_options_(self._metal_kernels, self._metal_kernels.nbytes, 0)
        buf_dims = self.metal_device.newBufferWithBytes_length_options_(self._metal_dims, self._metal_dims.nbytes, 0)

        # encode commands
        cmd_buf = self._metal_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()
        encoder.setComputePipelineState_(self._metal_pipeline)
        encoder.setBuffer_offset_atIndex_(buf_in,  0, 0)
        encoder.setBuffer_offset_atIndex_(buf_out, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_kern,0, 2)
        encoder.setBuffer_offset_atIndex_(buf_dims,0, 3)
        w, h, d = ((self._metal_dims + 7) // 8).tolist()
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            Metal.MTLSize(w, h, d), Metal.MTLSize(8, 8, 8)
        )
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # read back
        ptr = buf_out.contents().as_buffer(len(current_state))
        return bytes(ptr)
