#version 330 core

layout (location = 0) in ivec3 in_position;
layout (location = 1) in ivec4 rgba;
layout (location = 2) in int face_id;
layout (location = 3) in int ao_id;

// Uniform variables are the MVP matrices 
uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

// Computed voxel colour and uv coordinate to index into texture 
out vec4 voxel_color;
out vec2 uv;
out float shading; 

flat out int f_face_id;

const float face_shading[6] = float[6](
    1.0, 0.5, 
    0.5, 0.8,
    0.5, 0.8
);

const float ao_values[4] = float[4](0.1, 0.25, 0.5, 1.0);

const vec2 uv_coords[4] = vec2[4](
    vec2(1,0), vec2(1,1),
    vec2(0,1), vec2(0,0)
);

const int uv_indices[6] = int[6](
    0, 1, 2, 0, 2, 3
);

void main() {
    int uv_index = gl_VertexID % 6;
    uv = uv_coords[uv_indices[uv_index]];
    // convert back to value between 0 and 1, doesn't matter as frag shader casts everything to white
    voxel_color = vec4(rgba) / 255.0;
    shading = ao_values[ao_id]*face_shading[face_id];
    f_face_id = face_id;
    // additional 1.0 used to convert 3D position into 4D homogeneous coordinate
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);

}