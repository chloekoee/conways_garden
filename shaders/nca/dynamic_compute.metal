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

// indexing for X Y Z C
inline uint computeIndex(uint xi, uint yi, uint zi, uint c,
                        uint X, uint Y, uint Z) {
    uint vol  = 16 * Z * Y;
    uint area = 16 * Z;
    return (xi * vol) + (yi * area) + (zi * 16) + c;
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
                                        3,
                                        X, Y, Z);
                if (current[idx] > 0.1) 
                    return 1;
            }
        }
    }
    return 0;
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

    // Calculate pre-alive mask 
    uint aliveBefore = aliveMask(x, y, z, current, X, Y, Z);

    // Calculate perception vector 
    float perceptionVector[48];
    for (int c = 0; c < 16; c++) {
        for (int p = 0; p < 3; p++){
            int pIdx = (3*c) + p;
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
        
        // Calculate alive post mask 
        if (next[i] > 0.1){
            aliveAfter = 1;
        }
    }

    // Cell must be alive before and after to be alive in this step  
    if (aliveBefore + aliveAfter < 2){ 
        for (int c = 0; c < 16; c++){
            next[id*16 + c] = 0; 
        }
    }
}

