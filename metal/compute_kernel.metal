#include <metal_stdlib>
using namespace metal;

inline uint computeIndex(uint xi, uint yi, uint zi, uint c,
                        uint X, uint Y, uint Z) {
    uint vol  = 16 * Z * Y;
    uint area = 16 * Z;
    return (xi * vol) + (yi * area) + (zi * 16) + c;
}

inline float convolve3D(uint xi, uint yi, uint zi, uint c,
                        device const float* current,
                        constant const float* convKernel,
                        uint X, uint Y, uint Z) {
    float sum = 0.0f;
    for (uint ox = 0; ox < 3; ox++) {
        for (uint oy = 0; oy < 3; oy++) {
            for (uint oz = 0; oz < 3; oz++) {
                uint nx = (xi + ox - 1 + X) % X;
                uint ny = (yi + oy - 1 + Y) % Y;
                uint nz = (zi + oz - 1 + Z) % Z;
                uint idx = computeIndex(nx, ny, nz, c, X, Y, Z);
                uint kIdx = ox * 9 + oy * 3 + oz;
                sum += current[idx] * convKernel[kIdx];
            }
        }
    }
    return sum;
}

kernel void convolutionKernel(
    device const float*    current     [[buffer(0)]],
    device       float*    next        [[buffer(1)]],
    constant   float*      sobelX      [[buffer(2)]],
    constant   float*      sobelY      [[buffer(3)]],
    constant   float*      sobelZ      [[buffer(4)]],
    constant   float*      identity    [[buffer(5)]],
    constant   uint*        shape       [[buffer(6)]],
    constant   float*      l1w      [[buffer(7)]],
    constant   float*      l1b      [[buffer(8)]],
    constant   float*      l2w      [[buffer(9)]],
    uint                   id        [[thread_position_in_grid]]
) {
    uint X = shape[0];
    uint Y = shape[1];
    uint Z = shape[2];

    uint x = uint(id % X);
    uint y = uint((id / X) % Y);
    uint z = uint((id / (X*Y)) % Z);
    
    if (id >= X*Y*Z) {
       return;
    }

    // TODO: stack these into single array 
    float perceptionVector[64];
    for (int c = 0; c < 16; c++) {
        perceptionVector[(16*0) + c] = convolve3D(x, y, z, c, current, identity, X, Y, Z);
    }

    for (int c = 0; c < 16; c++) {
        perceptionVector[(16*1) + c] = convolve3D(x, y, z, c, current, sobelX, X, Y, Z);
    }
    
    for (int c = 0; c < 16; c++) {
        perceptionVector[(16*2) + c] = convolve3D(x, y, z, c, current, sobelY, X, Y, Z);
    }

    for (int c = 0; c < 16; c++) {
        perceptionVector[(16*3) + c] = convolve3D(x, y, z, c, current, sobelZ, X, Y, Z);
    }

    for (int c = 0; c < 64; c++) {
        next[id*64 + c] = perceptionVector[c];
    }

    // Pass through FCN
    
    // float hiddenVector[16];
    // for (int c = 0; c < 16; c++){
    //     float sum = 0.0;
    //     for (int p = 0; p < 64; p++){
    //         sum += l1_w[h * 16 + p] * perceptions[p];
    //     }
    //     // do relu and add biases 
    // }

    

    //  next[(16*id) + c] = convolve3D(x, y, z, c, current, identity, X, Y, Z);
    // next[id] = id;
    // for (uint c = 0; c < 16; c++) {
        
    //     uint idx = ( (x * (Y * Z) + y * Z + z) * 16 ) + c;
    //     idx = id;
    //     next[idx] = 9;//current[idx];
    // }
    // Only using 'identity' here; swap in sobelX/sobelY/sobelZ as needed
    // next = current;
    
}

