#include <metal_stdlib>
using namespace metal;

inline int computeIndex(int xi, int yi, int zi, int c,
                        int X, int Y, int Z) {
    int vol  = 16 * Z * Y;
    int area = 16 * Z;
    return (xi * vol) + (yi * area) + (zi * 16) + c;
}

inline float convolve3D(int xi, int yi, int zi, int c,
                        device const float* current,
                        constant const float* convKernel,
                        int X, int Y, int Z) {
    float sum = 0.0f;
    for (int ox = 0; ox < 3; ox++) {
        for (int oy = 0; oy < 3; oy++) {
            for (int oz = 0; oz < 3; oz++) {
                int nx = (xi + ox - 1 + X) % X;
                int ny = (yi + oy - 1 + Y) % Y;
                int nz = (zi + oz - 1 + Z) % Z;
                int idx = computeIndex(nx, ny, nz, c, X, Y, Z);
                int kIdx = ox * 9 + oy * 3 + oz;
                sum += current[idx] * convKernel[kIdx];
            }
        }
    }
    return sum;
}


// TODO: write in code for FCN 

// make array to hold hidden channel (size 16)
for (int c = 0; c < 16; c++){
    float sum = 0.0f;
    for (int p = 0; p < 64; p++){
        sum += l1_w[h * 16 + p] * perceptions[p];
    }
    // do relu and add biases 
}


// TODO: put this code back for the convolutions 
for (int c = 0; c < 16; c++) {
    int i = computeIndex(x, y, z, c, X, Y, Z);
    next[i] = current[i];
    // next[i] = convolve3D(x, y, z, c, current, identity, X, Y, Z);
}


kernel void convolutionKernel(
    device const float*    current     [[buffer(0)]],
    device       float*    next        [[buffer(1)]],
    constant   float*      sobelX      [[buffer(2)]],
    constant   float*      sobelY      [[buffer(3)]],
    constant   float*      sobelZ      [[buffer(4)]],
    constant   float*      identity    [[buffer(5)]],
    constant   int&        X           [[buffer(6)]],
    constant   int&        Y           [[buffer(7)]],
    constant   int&        Z           [[buffer(8)]],
    uint                   id        [[thread_position_in_grid]]
) {
    int x = int(id % X);
    int y = int((id / X) % Y);
    int z = int(id / (X*Y));

    //if (x >= X || y >= Y || z >= Z) {
    //    return;
    // }

    // suppose volume has 16 channels:
    for (int c = 0; c < 16; c++) {
        int idx = id;
        // ( (x * (Y * Z) + y * Z + z) * 16 ) + c;
        next[idx] = 9;//current[idx];
    }
    // Only using 'identity' here; swap in sobelX/sobelY/sobelZ as needed
    // next = current;
    
}
"""
# for (int c = 0; c < 16; c++) {
#         int i = computeIndex(x, y, z, c, X, Y, Z);
#         next[i] = current[i];
#         // next[i] = convolve3D(x, y, z, c, current, identity, X, Y, Z);
#     }