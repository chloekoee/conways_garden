#version 430

layout(local_size_x = 8,
       local_size_y = 8,
       local_size_z = 8) in;

layout(binding = 6) uniform usampler3D currentState;
layout(binding = 0, rgba8ui) writeonly uniform uimage3D nextState;

void main() {
    // obtain the 3D uv coordinate for this invocation
    ivec3 p = ivec3(gl_GlobalInvocationID);

    // fetch the current values from the sampler
    uvec4 v = texelFetch(currentState, p, 0);

    // increase alpha slightly
    v.a = (v.a < 5u) ? 0u : (v.a - 25u);
    imageStore(nextState, p, v);

    imageStore(nextState, p, v);
}
