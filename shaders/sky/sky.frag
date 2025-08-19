#version 460 core
layout(location = 0) out vec4 fragColor;

uniform vec2 u_resolution;
uniform mat3 rot3;
uniform vec3 pos3;
uniform float u_focal;
uniform float u_aspect;

// uniform float u_time;

const float PI = acos(-1.0);
const int NUM_OCTAVES = 7;

float noise(vec2 p) {
    return sin(p[0]) + sin(p[1]);
}

mat2 rot(float a) {
    float sa = sin(a);
    float ca = cos(a);
    return mat2(ca, -sa, sa, ca);
}

float fbm(vec2 p) {
    float res = 0.0;
    float amp = 0.3; // controls how much of other color incorporated
    float freq = 1.95; // controls thinness of lines higher = more thin
    for( int i = 0; i < NUM_OCTAVES; i++) {
        res += amp * noise(p);
        amp *= 0.5;
        p = p * freq * rot(PI / 4.0) - res * 0.4;
    }
    return res;
}

vec3 lightPos = vec3(250.0, 100.0, -300.0) * 4.0;

vec3 getSky(vec3 p, vec3 rd) {
    vec3 col = vec3(0.1373, 0.1059, 0.3176);
    float sun = 0.01 / (1.0 - dot(rd, normalize(lightPos)));
    col = mix(col, vec3(0.5804, 0.2431, 0.349), 2.0 * fbm(vec2(10.5 * length(rd.xz), rd.y)));
    col += sun * 0.1;
    return col;
}

vec3 render(vec2 uv) {
    vec3 col = vec3(0.4118, 0.2824, 0.6353);
    vec3 rayVS = normalize(vec3(uv * vec2(u_aspect, 1.0), -1.0) * u_focal);    
    vec3 rd    = rot3 * rayVS;

    vec3 skyCol = getSky(pos3, rd);    

    float sunFactor = 0.01 / (1.0 - dot(rd, normalize(lightPos)));
    vec3  sunColor  = vec3(1.0, 0.3412, 0.102) * 1.0;

    // mask out the sun from skyCol
    vec3 sunTerm = sunColor * sunFactor;
    vec3 skyOnly = skyCol - sunTerm;

    vec3 finalCol = skyOnly * 0.5   // darker sky
                + sunTerm;       // full-bright sun

    return finalCol;
}

void main() {
    // unrolled transformation of pixel coords -> space in world 
    vec2 uv = (2.0*gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;
    vec3 color = render(uv);
    fragColor = vec4(color, 1.0);

}
