#version 330 core

layout(location = 0) in vec2 cross_position;

void main() {
  gl_Position = vec4(cross_position, 0, 1);
}