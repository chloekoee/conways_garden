#version 330 core

layout (location = 0) out vec4 fragColor;

in vec3 marker_color;
in vec2 uv;

uniform sampler2D u_texture_0;

void main() {
    // sampling from the texture sampler array with texture unit 0
    fragColor = texture(u_texture_0, uv);
    fragColor.rgb += marker_color;
    fragColor.a = 0.25;// (fragColor.r + fragColor.b > 1.0) ? 0.0 : 1.0;
}
