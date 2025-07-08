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
inline uint computeIndex(uint xi, uint yi, uint zi, uint c,
                        uint X, uint Y, uint Z) {
    uint vol  = 16 * Z * Y;
    uint area = 16 * Z;
    return (xi * vol) + (yi * area) + (zi * 16) + c;
}

// indexing for C X Y Z
// inline uint computeIndex(uint xi, uint yi, uint zi, uint ci,
//                          uint X, uint Y, uint Z) {
//     // flatten (x,y,z,channel) in row‐major (channel last)
//      return ((ci * X + xi) * Y + yi) * Z + zi;
// }

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
                // ox * 9 + oy * 3 + oz indexes with size 27 3D kernel 
                // pIdx indexes which of 48 conv kernels we are using
                uint kIdx = (pIdx * 27) + ox * 9 + oy * 3 + oz;
                sum += current[idx] * pw[kIdx];
            }
        }
    }
    return sum;
}

inline int aliveMask(uint xi, uint yi, uint zi,
                     device const float* current,
                     uint X, uint Y, uint Z)
{
    for (int ox = -1; ox <= 1; ox++) {
        int sx = int(xi) + ox;
        if (sx < 0 || sx >= int(X))           continue;
        for (int oy = -1; oy <= 1; oy++) {
            int sy = int(yi) + oy;
            if (sy < 0 || sy >= int(Y))       continue;
            for (int oz = -1; oz <= 1; oz++) {
                int sz = int(zi) + oz;
                if (sz < 0 || sz >= int(Z))   continue;

                uint idx = computeIndex(uint(sx),
                                        uint(sy),
                                        uint(sz),
                                        /* channel = */ 3,
                                        X, Y, Z);
                if (current[idx] > 0.1f) 
                    return 1;
            }
        }
    }
    return 0;
}
// inline int aliveMask(uint xi, uint yi, uint zi,
//                         device const float* current,
//                         uint X, uint Y, uint Z) {
//     // check that in all neighbours, there exist at least one neighbour w alpha > 0.1
//     for (int ox = -1; ox < 2; ox++) {
//         int nx = (xi + ox - 1 + X);
//         if (nx >= 0 && nx < X){ 
//             for (int oy = -1; oy < 2; oy++) {
//                 int ny = (yi + oy - 1 + Y);
//                 if (ny >= 0 && ny < Y){ 
//                     for (int oz = -1; oz < 2; oz++) {
//                         int nz = (zi + oz - 1 + Z);

//                         if (nz >= 0 && nz < Z){
//                             uint idx = computeIndex(uint(nx), uint(ny), uint(nz), 3, X, Y, Z);          
//                             if (current[idx] > 0.1){
//                                 return 1;
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     return 0;
// }

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

    // calcualte pre alive mask 
    // 1 means neighbourhood index = 1 (neighbours alive)
    uint aliveBefore = aliveMask(x, y, z, current, X, Y, Z);

    // TODO: ALIVE MASKING PRESTEP
    // store boolean matrix for alive cell positions

    // TODO: check if channels are contiguous or perceptions
    // channels contiguous
    // float perceptionVector[48];
    // for (int c = 0; c < 16; c++) {
    //     for (int p = 0; p < 3; p++){
    //         int pIdx = (3*c) + p;
    //         perceptionVector[pIdx] = convolve3D(x, y, z, c, pIdx, R.pw, current, X, Y, Z);
    //     }
    // }

    // perceptions contiguous 
    float perceptionVector[48];
    for (int p = 0; p < 3; p++){
        for (int c = 0; c < 16; c++) {
            int pIdx = (16*p) + c;
            perceptionVector[pIdx] = convolve3D(x, y, z, c, pIdx, R.pw, current, X, Y, Z);
        }
    }

    // Going from size 48 perception to size 16 hidden layer 1
    float firstLayer[16];
    for (int c = 0; c < 16; c++){
        float sum = 0.0;
        for (int p = 0; p < 48; p++){
            sum += R.l1_w[c * 48 + p] * perceptionVector[p];
        }
        firstLayer[c] = max(0.0, sum + R.l1_b[c]);
    }
    
    // Going from size 16 hidden layer 1 to size 16 hidden layer 2
    float secondLayer[16];
    for (int c = 0; c < 16; c++){
        float sum = 0.0;
        for (int p = 0; p < 16; p++){
            sum += R.l2_w[c * 16 + p] * firstLayer[p];
        }
        secondLayer[c] = max(0.0, sum + R.l2_b[c]);
    }

    // Going from size 16 hidden layer 2 to size 16 output layer
    uint aliveAfter = 0;
    for (int c = 0; c < 16; c++){
        float sum = 0.0;
        for (int h = 0; h < 16; h++) {
            sum += R.l3_w[c * 16 + h] * secondLayer[h];
        }
        int i = computeIndex(x, y, z, c, X, Y, Z);
        next[i] = current[i] + sum;
        if (next[i] > 0.1){ // &  == 3 
            uint aliveAfter = 1;
        }
    }

    // ALIVE MASKING POST STEP
    // need to store boolean matrix for alive cell positions

    // calculate post alive mask 
  //#aliveMask(x, y, z, next, X, Y, Z);


    if (aliveAfter == 0){ // aliveBefore + aliveAfter < 2
        for (int c = 0; c < 16; c++){
            next[id*16 + c] = 0; 
        }
    }

    // ALIVE MASKING COMBINE PRE AND POST STEP TO ENSURE ONLY ALIVE CELLS PERSIST
    // i.e, we do not have random growth appear from no where, encourage growth from 
    // neighbouring cells (including those wrapped around due to circular padding)


    // if (aliveMask(x, y, z, current, X, Y, Z) == 1){
    //     for (int c = 0; c < 16; c++){
    //         int i = computeIndex(x, y, z, c, X, Y, Z);
    //         next[i] = 0; 
    //     }
    // }
}

