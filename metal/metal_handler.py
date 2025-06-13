import numpy as np
import Metal
from constants.sobel import identity, sobelX, sobelY, sobelZ


class MetalHandler():
    storage_mode = Metal.MTLResourceStorageModeShared

    def __init__(self, nca_name = "potted_flower"):
        self.load_kernel()
        self.load_state(nca_name)
        self.bind_static_resources()

    def mtl_buf(self, input_array):
        return self.dev.newBufferWithBytes_length_options_(
            input_array, input_array.nbytes, self.storage_mode
        )
    
    def load_kernel(self):
        self.dev = Metal.MTLCreateSystemDefaultDevice()
        src = open("metal/compute_kernel.metal", encoding="utf-8").read()
        lib, _ = self.dev.newLibraryWithSource_options_error_(src, None, None)
        self.func = lib.newFunctionWithName_("convolutionKernel")

    def load_state(self, nca_name):
        state = np.load(f"state/{nca_name}.npy", allow_pickle=True).item()
        seed = state["seed"]
        l1_w, l1_b, l2_w = (
            state["layers.0.weight"],
            state["layers.0.bias"],
            state["layers.2.weight"],
        )
        
        _, C, X, Y, Z = seed.shape
        self.shape = (X, Y, Z, C)
        self.num_voxels = X * Y * Z
        self.num_elements = X * Y * Z * C
        shape = np.array([X, Y, Z], dtype=np.uint32)

        current_array = np.ones(self.num_elements, dtype=np.float32)
        ## TODO: use the seed here 
        self.next_array = np.zeros_like(current_array)

        self.static_buffers = [
            self.mtl_buf(sobelX.ravel()),
            self.mtl_buf(sobelY.ravel()),
            self.mtl_buf(sobelZ.ravel()),
            self.mtl_buf(identity.ravel()),
            self.mtl_buf(shape),
            self.mtl_buf(l1_w.ravel()),
            self.mtl_buf(l1_b.ravel()),
            self.mtl_buf(l2_w.ravel()),
        ]

        self.current_buffer = self.mtl_buf(current_array)
        self.next_buffer = self.mtl_buf(self.next_array)

    def bind_static_resources(self):
        commandQueue = self.dev.newCommandQueue()
        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        # pipeline state object holding gpu data - like argument layout
        pso = self.dev.newComputePipelineStateWithFunction_error_(self.func, None)[0]

        # tells encoder, use this pipeline for all following dispatches
        computeEncoder.setComputePipelineState_(pso)  # TODO

        # create argument encoder for static resources
        # we define an encoder which knows how to pack the struct declared in slot 0 in the kernel, into an MTL buffer
        arg_enc = self.func.newArgumentEncoderWithBufferIndex_(0)
        static_arg_buffer = self.dev.newBufferWithLength_options_(
            arg_enc.encodedLength(), Metal.MTLResourceStorageModeShared
        )
        arg_enc.setArgumentBuffer_offset_(static_arg_buffer, 0)
        for i, b in enumerate(self.static_buffers):
            arg_enc.setBuffer_offset_atIndex_(b, 0, i)

        computeEncoder.setBuffer_offset_atIndex_(static_arg_buffer, 0, 0)

    def compute_next_state(self, step):
        curr_i, next_i = (step % 2), ((step + 1) % 2)

        commandQueue = self.dev.newCommandQueue()
        commandBuffer = commandQueue.commandBuffer()
        computeEncoder = commandBuffer.computeCommandEncoder()

        # pipeline state object holding gpu data - like argument layout
        pso = self.dev.newComputePipelineStateWithFunction_error_(self.func, None)[0]

        # tells encoder, use this pipeline for all following dispatches
        computeEncoder.setComputePipelineState_(pso)  # TODO

        # create argument encoder for static resources
        # we define an encoder which knows how to pack the struct declared in slot 0 in the kernel, into an MTL buffer
        arg_enc = self.func.newArgumentEncoderWithBufferIndex_(0)
        static_arg_buffer = self.dev.newBufferWithLength_options_(
            arg_enc.encodedLength(), Metal.MTLResourceStorageModeShared
        )
        arg_enc.setArgumentBuffer_offset_(static_arg_buffer, 0)
        for i, b in enumerate(self.static_buffers):
            arg_enc.setBuffer_offset_atIndex_(b, 0, i)

        computeEncoder.setBuffer_offset_atIndex_(static_arg_buffer, 0, 0)

        # directly encode dynamic buffers
        computeEncoder.setBuffer_offset_atIndex_(self.current_buffer, 0, curr_i+1)
        computeEncoder.setBuffer_offset_atIndex_(self.next_buffer, 0, next_i+1)

        grd, grp = self.compute_workgroup_sizes()
        computeEncoder.dispatchThreads_threadsPerThreadgroup_(grd, grp)

        ## submit and execute the command defined
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        ## return next state as numpy array 
        output_buffer = self.next_buffer.contents().as_buffer(next_array.nbytes)
        output_array = np.frombuffer(output_buffer, dtype=np.float32)
        return output_array.reshape(self.shape)

    def compute_workgroup_sizes(self):
        threads_per_grid = Metal.MTLSizeMake(self.num_voxels, 1, 1)
        # 32 threads per group ? depends on architecture ~ 32?
        threads_per_group = Metal.MTLSizeMake(16, 1, 1)
        return threads_per_grid, threads_per_group