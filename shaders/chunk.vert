#version 330 core

// Input is a single vertex data in form: x y, z, voxel_id, face_id
layout (location = 0) in ivec3 in_position;
layout (location = 1) in int voxel_id;
layout (location = 2) in int face_id;
layout (location = 3) in int ao_id;

// Uniform variables are the MVP matrices 
uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

// Computed voxel colour and uv coordinate to index into texture 
out vec3 voxel_color;
out vec2 uv;
out float shading; 

flat out int f_face_id;

const float face_shading[6] = float[6](
    1.0, 0.5, 
    0.5, 0.8,
    0.5, 0.8
);

const float ao_values[4] = float[4](0.1, 0.25, 0.5, 1.0);

// to facilitate simple assignment of triangle corners to texture (square) corners,
// let us define all possible uv coordinates to assing here
const vec2 uv_coords[4] = vec2[4](
    vec2(0,0), vec2(0, 1),
    vec2(1,1), vec2(1, 0)
);

const int uv_indices[6] = int[6](
    0, 1, 2, 0, 2, 3
);


vec3 hash31(float p) {
    vec3 p3 = fract(vec3(p * 21.2) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx) + 0.05;
}


void main() {
    int uv_index = gl_VertexID % 6;
    uv = uv_coords[uv_indices[uv_index]];
    voxel_color = hash31(voxel_id);
    shading = ao_values[ao_id]*face_shading[face_id];
    f_face_id = face_id;
    // additional 1.0 used to convert 3D position into 4D homogeneous coordinate
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
}