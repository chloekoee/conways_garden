#version 430

layout(local_size_x = 8,
       local_size_y = 8,
       local_size_z = 8) in;

layout(std430, binding = 0) buffer CurrentState { 
    float current[]; 
};

layout(std430, binding = 1) buffer NextState { 
    float next[]; 
};

uniform float sobelX [27];
uniform float sobelY [27];
uniform float sobelZ [27];
uniform float identity [27];

uniform int X;
uniform int Y;
uniform int Z; 


int index(int x, int y, int z, int c) {
    int vol = 16*Z*Y;
    int area = 16*Z;
    return (x*vol) + (y*area) + (z*16) + c;
}

float convolve(int x, int y, int z, int c, in float kernel[27]) {
    float  sum = 0;
    for (int xi = 0; xi < 3; xi++) {
        for (int yi = 0; yi < 3; yi++) {
            for (int zi = 0; zi < 3; zi++) {
                int nx = (x + xi -1 + X) % X;
                int ny = (y + yi -1 + Y) % Y;
                int nz = (z + zi -1 + Z) % Z;
                int i = index(nx, ny, nz, c);
                int ki = xi * 9 + yi * 3 + zi;
                sum += current[i] * kernel[ki];
            }
        }
    }
    return sum;
}


void main() {

    // obtain the 3D uv coordinate for this invocation
    ivec3 p = ivec3(gl_GlobalInvocationID);
     int x = p.x, y = p.y, z = p.z;

    // guard agaisnt out of bound work groups 
    if (x >= X || y >= Y || z >= Z) {
    return;
    }

    // only uses identity currently
    for (int c = 0; c < 16; c++){
        int i = index(x,y,z,c);
        // next[i] = convolve(x,y,z,c, sobelX);
        // next[i] = convolve(x,y,z,c, sobelY);
        // next[i] = convolve(x,y,z,c, sobelZ);
        next[i] = convolve(x,y,z,c, identity);
    }

}
    // construct the perception vector
    // float[16] perception = new int[16];
    // for (int c = 0; c < 16; c++){
    //     perception[c] = convolve(p, c);
    // }
    
