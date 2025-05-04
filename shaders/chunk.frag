#version 330 core

layout (location = 0) out vec4 fragColor;

const vec3 gamma = vec3(2.2);
const vec3 inv_gamma = 1 / gamma;

// uniform sampler2D u_texture_0;
uniform sampler2D face_textures[6];

in vec3 voxel_color;
in vec2 uv;
in float shading;

flat in int f_face_id;


void main() {
    // vec3 tex_col = texture(u_texture_0, uv).rgb;
    vec3 tex_col = texture(face_textures[f_face_id], uv).rgb;
    tex_col = pow(tex_col, gamma);

    tex_col.rgb *= voxel_color;
    //colouring everything white
    // tex_col = tex_col * 0.001 + vec3(1);
    tex_col *= shading;

    tex_col = pow(tex_col, inv_gamma);
    fragColor = vec4(tex_col, 1);
}