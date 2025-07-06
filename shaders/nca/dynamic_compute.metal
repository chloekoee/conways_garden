// inline float convolve3D(uint xi, uint yi, uint zi, uint c,
//                         uint pIdx,
//                         constant const float* pw,
//                         device   const float* current,
//                         uint X, uint Y, uint Z) {
//     float sum = 0.0f;
//     for (uint ox = 0; ox < 3; ox++) {
//       for (uint oy = 0; oy < 3; oy++) {
//         for (uint oz = 0; oz < 3; oz++) {
//           // compute signed neighbor coords
//           int nx = int(xi) + int(ox) - 1;
//           int ny = int(yi) + int(oy) - 1;
//           int nz = int(zi) + int(oz) - 1;
//           // zero‐pad: skip any out‐of‐bounds neighbor
//           if (nx < 0 || nx >= int(X) ||
//               ny < 0 || ny >= int(Y) ||
//               nz < 0 || nz >= int(Z)) {
//             continue;
//           }
//           uint idx = computeIndex(uint(nx), uint(ny), uint(nz), c, X, Y, Z);
//           uint outChan = c*3 + pIdx;
//           uint kIdx    = outChan*27 + ox*9 + oy*3 + oz;
//           sum += current[idx] * pw[kIdx];
//         }
//       }
//     }
//     return sum;
// }
#include <metal_stdlib>
using namespace metal;

struct StaticResources {
    constant   float*      pw          [[id(0)]];
    constant   uint3&      shape       [[id(1)]];
    constant   float*      l1_w        [[id(2)]];
    constant   float*      l1_b        [[id(3)]];
    constant   float*      l2_w        [[id(4)]];
    constant   float*      l2_b        [[id(5)]];
    constant   float*      l3_w        [[id(6)]];
};

// TODO: old indexing for X Y Z C
// inline uint computeIndex(uint xi, uint yi, uint zi, uint c,
//                         uint X, uint Y, uint Z) {
//     uint vol  = 16 * Z * Y;
//     uint area = 16 * Z;
//     return (xi * vol) + (yi * area) + (zi * 16) + c;
// }

// indexing for C X Y Z
inline uint computeIndex(uint xi, uint yi, uint zi, uint ci,
                         uint X, uint Y, uint Z) {
    // flatten (x,y,z,channel) in row‐major (channel last)
     return ((ci * X + xi) * Y + yi) * Z + zi;
}

inline float convolve3D(uint xi, uint yi, uint zi, uint c,
                        uint pIdx, 
                        constant const float* pw, 
                        device const float* current,
                        uint X, uint Y, uint Z) {
    float sum = 0.0f;
    for (uint ox = 0; ox < 3; ox++) {
        for (uint oy = 0; oy < 3; oy++) {
            for (uint oz = 0; oz < 3; oz++) {
                uint nx = (xi + ox + X - 1) % X;
                uint ny = (yi + oy + Y - 1) % Y;
                uint nz = (zi + oz + Z - 1) % Z;
                uint idx = computeIndex(nx, ny, nz, c, X, Y, Z);
                uint kIdx = (pIdx * 27) + ox * 9 + oy * 3 + oz;
                sum += current[idx] * pw[kIdx];
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

    // TODO: check c 
    float perceptionVector[48];
    for (int c = 0; c < 16; c++) {
        for (int p = 0; p < 3; p++){
            int pIdx = (3*c) + p;
            perceptionVector[pIdx] = convolve3D(x, y, z, c, pIdx, R.pw, current, X, Y, Z);
        }
    }

    // Pass through FCN
    float firstLayer[16];
    for (int c = 0; c < 16; c++){
        float sum = 0.0;
        for (int p = 0; p < 48; p++){
            sum += R.l1_w[c * 48 + p] * perceptionVector[p];
        }
        firstLayer[c] = max(0.0, sum + R.l1_b[c]);
    }
    
    float secondLayer[16];
    for (int c = 0; c < 16; c++){
        float sum = 0.0;
        for (int p = 0; p < 16; p++){
            sum += R.l2_w[c * 16 + p] * firstLayer[p];
        }
        secondLayer[c] = max(0.0, sum + R.l2_b[c]);
    }

    for (int c = 0; c < 16; c++){
        float sum = 0.0;
        for (int h = 0; h < 16; h++) {
            sum += R.l3_w[c * 16 + h] * secondLayer[h];
        }
        int i = computeIndex(x, y, z, c, X, Y, Z);
        next[i] = current[i] + sum;
    }

    if (aliveMask(x, y, z, current, X, Y, Z) == 1){
        for (int c = 0; c < 16; c++){
            int i = computeIndex(x, y, z, c, X, Y, Z);
            next[i] = 0; 
        }
    }
}

