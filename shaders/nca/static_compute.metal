#include <metal_stdlib>
using namespace metal;

struct StaticResources {
    constant   float*      sobelX      [[id(0)]];
    constant   float*      sobelY      [[id(1)]];
    constant   float*      sobelZ      [[id(2)]];
    constant   float*      identity    [[id(3)]];
    constant   uint3&       shape       [[id(4)]];
    constant   float*      l1_w        [[id(5)]];
    constant   float*      l1_b        [[id(6)]];
    constant   float*      l2_w        [[id(7)]];
};

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
                uint nx = (xi + ox + X - 1) % X;
                uint ny = (yi + oy + Y - 1) % Y;
                uint nz = (zi + oz + Z - 1) % Z;
                uint idx = computeIndex(nx, ny, nz, c, X, Y, Z);
                uint kIdx = ox * 9 + oy * 3 + oz;
                sum += current[idx] * convKernel[kIdx];
            }
        }
    }
    return sum;
}

inline int aliveMask(uint xi, uint yi, uint zi,
                        device const float* current,
                        uint X, uint Y, uint Z) {
    // check that in all neighbours, there exist at least one neighbour w alpha > 0.1
    for (uint ox = 0; ox < 3; ox++) {
        for (uint oy = 0; oy < 3; oy++) {
            for (uint oz = 0; oz < 3; oz++) {
                uint nx = (xi + ox - 1 + X) % X;
                uint ny = (yi + oy - 1 + Y) % Y;
                uint nz = (zi + oz - 1 + Z) % Z;
                uint idx = computeIndex(nx, ny, nz, 3, X, Y, Z);
                if (current[idx] > 0.1){
                    return 0;
                }
            }
        }
    }
    return 1;
}

kernel void convolutionKernel(
    constant        StaticResources&    R           [[buffer(0)]],
    device const    float*              current     [[buffer(1)]],
    device          float*              next        [[buffer(2)]],
    uint                                id          [[thread_position_in_grid]]
) {
    uint X = R.shape[0];
    uint Y = R.shape[1];
    uint Z = R.shape[2];

    uint z = uint(id % Z);
    uint y = uint((id / Z) % Y);
    uint x = uint((id / (Z*Y)) % X);
    
    if (id >= X*Y*Z) {
       return;
    }

    // TODO: stack these into single array 
    float perceptionVector[64];
    for (int c = 0; c < 16; c++) {
        perceptionVector[(16*0) + c] = convolve3D(x, y, z, c, current, R.identity, X, Y, Z);
    }

    for (int c = 0; c < 16; c++) {
        perceptionVector[(16*1) + c] = convolve3D(x, y, z, c, current, R.sobelX, X, Y, Z);
    }
    
    for (int c = 0; c < 16; c++) {
        perceptionVector[(16*2) + c] = convolve3D(x, y, z, c, current, R.sobelY, X, Y, Z);
    }

    for (int c = 0; c < 16; c++) {
        perceptionVector[(16*3) + c] = convolve3D(x, y, z, c, current, R.sobelZ, X, Y, Z);
    }

    // Pass through FCN
    float hiddenLayer[16];
    for (int c = 0; c < 16; c++){
        float sum = 0.0;
        for (int p = 0; p < 64; p++){
            sum += R.l1_w[c * 64 + p] * perceptionVector[p];
        }
        hiddenLayer[c] = max(0.0, sum + R.l1_b[c]);
    }

    for (int c = 0; c < 16; c++){
        float sum = 0.0;
        for (int h = 0; h < 16; h++) {
            sum += R.l2_w[c * 16 + h] * hiddenLayer[h];
        }
        int i = computeIndex(x, y, z, c, X, Y, Z);
        next[i] = current[i] + sum;
    }

    if (aliveMask(x, y, z, current, X, Y, Z) == 1){
        for (int c = 0; c < 16; c++){
            next[id*16 + c] = 0; 
        }
    }
}

