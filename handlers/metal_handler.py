import numpy as np
import Metal
from constants.sobel import identity, sobelX, sobelY, sobelZ


class MetalHandler:
    storage_mode = Metal.MTLResourceStorageModeShared

    def __init__(self, nca_name, uses_learnable_perception):
        self.static = False if uses_learnable_perception else True

        self.load_kernel()
        self.load_state(nca_name)
        self.bind_static_resources()

    def load_kernel(self):
        self.dev = Metal.MTLCreateSystemDefaultDevice()
        file_name = (
            "shaders/nca/static_compute.metal"
            if self.static
            else "shaders/nca/dynamic_compute.metal"
        )

        src = open(file_name, encoding="utf-8").read()

        lib, _ = self.dev.newLibraryWithSource_options_error_(src, None, None)

        self.func = lib.newFunctionWithName_("convolutionKernel")
        # pipeline state object holding gpu data - like argument layout
        self.pso = self.dev.newComputePipelineStateWithFunction_error_(self.func, None)[
            0
        ]

    def load_state(self, nca_name):
        state = np.load(f"state/{nca_name}.npy", allow_pickle=True).item()

        if self.static:
            perception_arrays = [sobelX, sobelY, sobelZ, identity]
            update_arrays = [
                state["layers.0.weight"],
                state["layers.0.bias"],
                state["layers.2.weight"],
            ]
        else:
            perception_arrays = [state["perception_layer.conv.weight"]]
            update_arrays = [
                state["update_network.conv1.weight"],
                state["update_network.conv1.bias"],
                state["update_network.conv2.weight"],
                state["update_network.conv2.bias"],
                state["update_network.conv3.weight"],
            ]

        perception_bufs = []
        for arr in perception_arrays:
            a32 = np.ascontiguousarray(arr.astype(np.float32))
            perception_bufs.append(self.mtl_buf(a32.ravel()))

        update_bufs = []
        for arr in update_arrays:
            a32 = np.ascontiguousarray(arr.astype(np.float32))
            update_bufs.append(self.mtl_buf(a32.ravel()))
        seed = state["seed"].squeeze(0)  # get rid of batch dimension
        seed = np.moveaxis(seed, 0, 3)
        self.seed = seed
        seed = np.ascontiguousarray(seed)
        X, Y, Z, C = self.seed.shape
        shape_buf = self.mtl_buf(np.array([X, Y, Z], dtype=np.uint32))
        self.shape = self.seed.shape
        self.NUM_VOXELS = X * Y * Z
        self.nbytes = seed.nbytes

        self.static_buffers = [*perception_bufs, shape_buf, *update_bufs]

        self.current_buffer = self.mtl_buf(seed)
        self.next_buffer = self.mtl_buf(np.zeros_like(seed, dtype=np.float32))

    def bind_static_resources(self):
        arg_enc = self.func.newArgumentEncoderWithBufferIndex_(0)
        self.static_arg_buffer = self.dev.newBufferWithLength_options_(
            arg_enc.encodedLength(), Metal.MTLResourceStorageModeShared
        )
        arg_enc.setArgumentBuffer_offset_(self.static_arg_buffer, 0)
        for i, b in enumerate(self.static_buffers):
            arg_enc.setBuffer_offset_atIndex_(b, 0, i)

    def compute_next_state(self):
        commandQueue = self.dev.newCommandQueue()
        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        # tells encoder, use this pipeline for all following dispatches
        computeEncoder.setComputePipelineState_(self.pso)

        ## encode static and dynamic buffers
        computeEncoder.setBuffer_offset_atIndex_(self.static_arg_buffer, 0, 0)
        computeEncoder.setBuffer_offset_atIndex_(self.current_buffer, 0, 1)
        computeEncoder.setBuffer_offset_atIndex_(self.next_buffer, 0, 2)

        grd, grp = self.compute_workgroup_sizes()
        computeEncoder.dispatchThreads_threadsPerThreadgroup_(grd, grp)

        ## submit and execute the command defined
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        ## read output back out to CPU
        output_buffer = self.next_buffer.contents().as_buffer(self.nbytes)
        output_array_flat = np.frombuffer(output_buffer, dtype=np.float32)
        output_array = output_array_flat.reshape(self.shape)

        ## ping pong buffers
        self.current_buffer, self.next_buffer = self.next_buffer, self.current_buffer

        return output_array

    def compute_workgroup_sizes(self):
        threads_per_grid = Metal.MTLSizeMake(self.NUM_VOXELS, 1, 1)
        # 32 threads per group ? depends on architecture ~ 32?
        threads_per_group = Metal.MTLSizeMake(16, 1, 1)
        return threads_per_grid, threads_per_group

    def mtl_buf(self, input_array):
        return self.dev.newBufferWithBytes_length_options_(
            input_array, input_array.nbytes, self.storage_mode
        )

    def overwrite_voxel(self, x, y, z):
        ptr = self.current_buffer.contents().as_buffer(self.nbytes)
        float_view = np.frombuffer(ptr, dtype=np.float32)
        idx = self.offset(x, y, z)
        C = self.shape[3]
        float_view[idx : idx + C] = 0.0

    def offset(self, x, y, z):
        _, Y, Z, C = self.shape
        # return (((x * Y) + y) * Z + z) * C * 4 - 1  ## float byte offset
        return (((x * Y) + y) * Z + z) * C  ## float byte offset

    def get_shape(self):
        return self.shape

    def get_seed(self):
        return self.seed
