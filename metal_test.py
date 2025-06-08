import metalcompute as mc
import os
import numpy as np

NCA_NAME = "potted_flower"
## import metal shader
this_dir = os.path.dirname(__file__)
shader_path = os.path.join(this_dir, "metal/compute_kernel.metal")
with open(shader_path, "r") as f:
    shader_source = f.read()

## import weights and biases
weights = np.load(f"weights/{NCA_NAME}.npy", allow_pickle = True).item()
layer_1_weight = weights['layers.0.weight']
layer_1_bias = weights['layers.0.bias']
layer_2_weight = weights['layers.2.weight']

## create metal device to run compute_shader_kernel with 
dev = mc.Device()

## create compiled kernel to run
run_compute_shader = dev.kernel(shader_source).function("convolutionKernel")

## make buffers for inputs/outputs
X, Y, Z = 2, 1, 1
channels = 16
total_voxels = X * Y * Z
flattened_size = X * Y * Z * channels

# current_np  = np.ones(flattened_size).astype(np.float32)
current_np  = np.zeros(flattened_size).astype(np.float32)
next_np     = np.zeros(flattened_size).astype(np.float32) 
shape = np.array([X, Y, Z]).astype(np.uint32)

sobelX_np = np.array(
    [
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
    ],
    dtype=np.float32,
)
sobelY_np = -np.rot90(sobelX_np, k=-1, axes=(0, 1)) 
sobelZ_np = np.transpose(sobelX_np, (2,1,0))

identity_np = np.zeros((3, 3, 3), dtype=np.float32)
identity_np[1, 1, 1] = 1.0
identity_np.ravel()

## create MTLBuffers
buf_current  = dev.buffer(current_np)
buf_next     = dev.buffer(next_np)
buf_sobelX   = dev.buffer(sobelX_np)
buf_sobelY   = dev.buffer(sobelX_np)
buf_sobelZ   = dev.buffer(sobelX_np)
buf_identity = dev.buffer(identity_np)
buf_shape = dev.buffer(shape)
buf_l1w = dev.buffer(layer_1_weight)
buf_l1b = dev.buffer(layer_1_bias)
buf_l2w = dev.buffer(layer_2_weight)

## run kernel (count = total_voxels seems to not matter)
run_compute_shader(total_voxels, buf_current, buf_next, buf_sobelX, buf_sobelY, buf_sobelZ, buf_identity, buf_shape, buf_l1w, buf_l1b, buf_l2w)

## write out results from gpu -> cpu 
print(np.frombuffer(buf_next, dtype = 'f'))