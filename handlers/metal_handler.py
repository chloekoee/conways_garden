import numpy as np
# import Metal
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
            l1_w, l1_b, l2_w = (
                state["layers.0.weight"],
                state["layers.0.bias"],
                state["layers.2.weight"],
            )

            perception_params = [
                self.mtl_buf(sobelX.ravel()),
                self.mtl_buf(sobelY.ravel()),
                self.mtl_buf(sobelZ.ravel()),
                self.mtl_buf(identity.ravel()),
            ]

            update_params = [l1_w, l1_b, l2_w]
        else:
            pw, l1_w, l1_b, l2_w, l2_b, l3_w = (
                state["perception_layer.conv.weight"],
                state["update_network.conv1.weight"],
                state["update_network.conv1.bias"],
                state["update_network.conv2.weight"],
                state["update_network.conv2.bias"],
                state["update_network.conv3.weight"],
            )
            perception_params = [pw]
            update_params = [l1_w, l1_b, l2_w, l2_b, l3_w]

        update_params = map(lambda p: self.mtl_buf(p.ravel()), update_params)

        seed = state["seed"].squeeze(0)  # get rid of batch dimension
        seed = np.swapaxes(seed, 2, 3)  # swap z and y
        seed = np.moveaxis(seed, 0, 3)
        self.seed = seed
        seed = np.ascontiguousarray(seed)

        X, Y, Z, C = seed.shape
        self.shape = seed.shape
        self.NUM_VOXELS = X * Y * Z
        self.nbytes = seed.nbytes

        # make buffers
        self.static_buffers = [
            *perception_params,
            self.mtl_buf(np.array([X, Y, Z], dtype=np.uint32)),
            *update_params,
        ]

        # current and next states
        self.current_buffer = self.mtl_buf(seed)
        self.next_buffer = self.mtl_buf(np.zeros_like(seed))

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
