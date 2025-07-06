#version 330 core

layout (location = 0) out vec4 fragColor;

void main() {
    fragColor.rgb = vec3(0.9765, 0.2784, 0.698);;
    fragColor.a = 1.0;
}
