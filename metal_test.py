#!/usr/bin/env python3
import numpy as np
import Metal
from constants.sobel import identity, sobelX, sobelY, sobelZ
from Foundation import NSMakeRange

## load the metal kernel 
# ----------------------------------------------------
dev = Metal.MTLCreateSystemDefaultDevice()
src = open('metal/compute_kernel.metal', encoding='utf-8').read()
lib, _ = dev.newLibraryWithSource_options_error_(src, None, None)
func = lib.newFunctionWithName_("convolutionKernel")

## prepare input arguments
# ----------------------------------------------------
storage_mode = Metal.MTLResourceStorageModeShared

def mtl_buf(input_array):    
    return dev.newBufferWithBytes_length_options_(input_array, input_array.nbytes, storage_mode)

# hardcoded dimensions for now 
X, Y, Z, C= 2, 1, 1, 16
NUM_VOXELS = X*Y*Z
NUM_ELEMENTS = X*Y*Z*C
shape = np.array([X, Y, Z], dtype=np.uint32)

# loaded weights 
w = np.load(f"weights/potted_flower.npy", allow_pickle=True).item()
l1_w, l1_b, l2_w = w['layers.0.weight'], w['layers.0.bias'], w['layers.2.weight']

# current and next states 
cur = np.ones(NUM_ELEMENTS, dtype=np.float32)
nxt = np.zeros_like(cur)

# make buffers
bufs = [
    mtl_buf(cur),
    mtl_buf(nxt),
    mtl_buf(sobelX.ravel()),
    mtl_buf(sobelY.ravel()),
    mtl_buf(sobelZ.ravel()),
    mtl_buf(identity.ravel()),
    mtl_buf(shape),
    mtl_buf(l1_w.ravel()),
    mtl_buf(l1_b.ravel()),
    mtl_buf(l2_w.ravel()),
]
offsets = [0] * len(bufs)
bindings = NSMakeRange(0, len(bufs))

## create commands to run kernel 
# ----------------------------------------------------
commandQueue = dev.newCommandQueue()
commandBuffer = commandQueue.commandBuffer()
computeEncoder = commandBuffer.computeCommandEncoder()

pso = dev.newComputePipelineStateWithFunction_error_(func, None)[0]
computeEncoder.setComputePipelineState_(pso)  # set kernel to call

computeEncoder.setBuffers_offsets_withRange_(bufs, offsets, bindings)

## defining work group sizes
# TODO: can try making it a 3D grid - but may be non uniform size, and this could be unsupported?
threads_per_grid = Metal.MTLSizeMake(NUM_VOXELS, 1, 1)  
# 32 threads per group ? depends on architecture ~ 32? 
threads_per_group = Metal.MTLSizeMake(16, 1, 1)
computeEncoder.dispatchThreads_threadsPerThreadgroup_(threads_per_grid, threads_per_group)

## submit and execute the command defined 
computeEncoder.endEncoding() 
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

## read output back out to CPU 
# ----------------------------------------------------
output_buffer_ptr = bufs[1].contents()
output_buffer = output_buffer_ptr.as_buffer(nxt.nbytes)
output_array = np.frombuffer(output_buffer, dtype=np.float32)
print("Result:", output_array.reshape((X,Y,Z,C)))