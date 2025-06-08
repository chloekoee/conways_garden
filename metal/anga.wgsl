
@group(0) @binding(0) var<uniform> shape: ParameterShape;
@group(0) @binding(1) var<storage> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> next_state: array<f32>;
@group(0) @binding(3) var<storage> l1_w: array<f32>;
@group(0) @binding(4) var<storage> l1_b: array<f32>;
@group(0) @binding(5) var<storage> l2_w: array<f32>;
@group(0) @binding(6) var<uniform> seed: u32;

// Constants
const CHANNELS = ${channels};
const CONVOLUTIONS = ${convolutions.length + 1};
const HIDDEN_CHANNELS = ${hiddenChannels};
const PERCEPTION_VECTOR = CHANNELS * CONVOLUTIONS;

// Perception kernels
${convolutions.map((conv, i) => `const PERCEPTION_${i}: mat3x3f = mat3x3f(${conv.map(v => v.toFixed(1)).join(', ')});`).join('\n')}

@workgroup_size(8, 8)
@compute
fn compute_main(@builtin(global_invocation_id) pos: vec3u) {
  let size = shape.size;

  // Compute grid position
  let x = pos.x;
  let y = pos.y;

  // Ensure valid grid bounds
  if (x >= size || y >= size) {
    return;
  }

  // Perception convolution
  var perceptions: array<f32, PERCEPTION_VECTOR>;

  // Copy identity convolution directly from state
  for (var c = 0u; c < CHANNELS; c++) {
    perceptions[c${stripe ? ' * CONVOLUTIONS' : ''}] = state[index(c, x, y)];
  }

  // Compute convolutions
  for (var c = 0u; c < CHANNELS; c++) {
    ${convolutions.map((_, i) => `perceptions[c ${stripe ? '* CONVOLUTIONS +' : '+ CHANNELS *'} ${i + 1}] = convolve(c, x, y, PERCEPTION_${i});`).join('\n')}
  }

  // Fully connected layers
  var hidden: array<f32, HIDDEN_CHANNELS>;
  for (var h = 0u; h < HIDDEN_CHANNELS; h++) {
    var sum: f32 = 0.0;
    for (var p = 0u; p < PERCEPTION_VECTOR; p++) {
      sum += l1_w[h * PERCEPTION_VECTOR + p] * perceptions[p];
    }
    // ReLU(x) = max(0, x)
    hidden[h] = max(0.0, sum + l1_b[h]);
  }

  // Output layer (next state computation)
  for (var c = 0u; c < CHANNELS; c++) {
    var sum: f32 = 0.0;
    for (var h = 0u; h < HIDDEN_CHANNELS; h++) {
      sum += l2_w[c * HIDDEN_CHANNELS + h] * hidden[h];
    }

    let i = index(c, x, y);
    next_state[i] = state[i] + sum * mask(i);
  }

  // Alive masking
  alive_mask(x, y);
}





// Alive masking
fn alive_mask(x: u32, y: u32) {
  let size = shape.size;

  for (var ky = 0u; ky < 3u; ky++) {
    for (var kx = 0u; kx < 3u; kx++) {
      // Find neighbours with circular padding
      let dx = (x + kx - 1 + size) % size;
      let dy = (y + ky - 1 + size) % size;
      // Check alpha > 0.1 
      if (next_state[index(3, dx, dy)] > 0.1) {
        return;
      }
    }
  }

  // Write zeros
  for (var c = 0u; c < 16; c++) {
    next_state[index(c, x, y)] = 0;
  }
}