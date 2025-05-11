#version 330 core

layout (location = 0) out vec4 fragColor;

const vec3 gamma = vec3(2.2);
const vec3 inv_gamma = 1.0 / gamma;

uniform sampler2D face_textures[6];

in vec4 voxel_color; 
in vec2 uv;
in float shading;

flat in int f_face_id;

void main() {
    vec3 tex_col = texture(face_textures[f_face_id], uv).rgb;
    tex_col = pow(tex_col, gamma);

    tex_col *= voxel_color.rgb;
    tex_col *= shading;

    tex_col = pow(tex_col, inv_gamma);
    fragColor = vec4(tex_col, voxel_color.a);
}