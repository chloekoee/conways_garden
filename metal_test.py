#!/usr/bin/env python3
import numpy as np
import Metal
from constants.sobel import identity, sobelX, sobelY, sobelZ

VOXEL_PATH_NAME = "potted_flower"
## load the metal kernel
# ----------------------------------------------------
dev = Metal.MTLCreateSystemDefaultDevice()
src = open("metal/compute_kernel.metal", encoding="utf-8").read()
lib, _ = dev.newLibraryWithSource_options_error_(src, None, None)
func = lib.newFunctionWithName_("convolutionKernel")

## prepare input arguments
# ----------------------------------------------------
storage_mode = Metal.MTLResourceStorageModeShared


def mtl_buf(input_array):
    return dev.newBufferWithBytes_length_options_(
        input_array, input_array.nbytes, storage_mode
    )


# load state
state = np.load(f"state/{VOXEL_PATH_NAME}.npy", allow_pickle=True).item()
l1_w, l1_b, l2_w = (
    state["layers.0.weight"],
    state["layers.0.bias"],
    state["layers.2.weight"],
)
seed = state["seed"]

# hardcoded dimensions for now
_, C, X, Y, Z = seed.shape
NUM_VOXELS = X * Y * Z
NUM_ELEMENTS = X * Y * Z * C
shape = np.array([X, Y, Z], dtype=np.uint32)

# current and next states
current_array = np.ones(NUM_ELEMENTS, dtype=np.float32)
next_array = np.zeros_like(current_array)

# make buffers
static_buffers = [
    mtl_buf(sobelX.ravel()),
    mtl_buf(sobelY.ravel()),
    mtl_buf(sobelZ.ravel()),
    mtl_buf(identity.ravel()),
    mtl_buf(shape),
    mtl_buf(l1_w.ravel()),
    mtl_buf(l1_b.ravel()),
    mtl_buf(l2_w.ravel()),
]

current_buffer = mtl_buf(current_array)
next_buffer = mtl_buf(next_array)

## create commands to run kernel
# ----------------------------------------------------
commandQueue = dev.newCommandQueue()
commandBuffer = commandQueue.commandBuffer()
computeEncoder = commandBuffer.computeCommandEncoder()

# pipeline state object holding gpu data - like argument layout
pso = dev.newComputePipelineStateWithFunction_error_(func, None)[0]

# tells encoder, use this pipeline for all following dispatches
computeEncoder.setComputePipelineState_(pso)  # TODO

# create argument encoder for static resources
# we define an encoder which knows how to pack the struct declared in slot 0 in the kernel, into an MTL buffer
arg_enc = func.newArgumentEncoderWithBufferIndex_(0)
static_arg_buffer = dev.newBufferWithLength_options_(
    arg_enc.encodedLength(), Metal.MTLResourceStorageModeShared
)
arg_enc.setArgumentBuffer_offset_(static_arg_buffer, 0)
for i, b in enumerate(static_buffers):
    arg_enc.setBuffer_offset_atIndex_(b, 0, i)

computeEncoder.setBuffer_offset_atIndex_(static_arg_buffer, 0, 0)

# directly encode dynamic buffers
computeEncoder.setBuffer_offset_atIndex_(current_buffer, 0, 1)
computeEncoder.setBuffer_offset_atIndex_(next_buffer, 0, 2)

## defining work group sizes
# TODO: can try making it a 3D grid - but may be non uniform size, and this could be unsupported?
threads_per_grid = Metal.MTLSizeMake(NUM_VOXELS, 1, 1)
# 32 threads per group ? depends on architecture ~ 32?
threads_per_group = Metal.MTLSizeMake(16, 1, 1)
computeEncoder.dispatchThreads_threadsPerThreadgroup_(
    threads_per_grid, threads_per_group
)

## submit and execute the command defined
computeEncoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

## read output back out to CPU
# ----------------------------------------------------
output_buffer = next_buffer.contents().as_buffer(next_array.nbytes)
output_array = np.frombuffer(output_buffer, dtype=np.float32)
print("Result:", output_array.reshape((X, Y, Z, C)))
