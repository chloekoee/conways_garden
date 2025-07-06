#version 330 core

layout (location = 0) in ivec3 in_position;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

out vec4 marker_color;
out vec2 uv;

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
    marker_color = vec4(0.9765, 0.2784, 0.698, 1.0); // hardcoded pink here as only one mode
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
}