---
name: rltools-web-visualization
description: >
  Web-based visualization and browser simulation — Canvas 2D / Three.js 3D render functions, WASM compilation, Chart.js dashboards, gamepad support.
layer: L2
domain: [general-rl]
source-project: rl-tools-framework
depends-on: [rltools-build]
tags: [web, wasm, visualization, threejs]
---

# RL-Tools Web Visualization

## Overview

Build web-based visualization, interactive dashboards, and browser-based simulations for rl-tools environments. Covers environment render functions, WebAssembly compilation, and the ExTrack UI component library.

## When to Use This Skill

- User wants to create a 2D or 3D render function for an environment
- User wants to compile C++ to WebAssembly for browser inference
- User wants to build interactive visualization dashboards
- User wants to use Chart.js for learning curve plots
- User wants to add gamepad support for interactive control

## Web Tools Ecosystem

| Tool | Location | Purpose |
|------|----------|---------|
| zoo.rl.tools | `/home/ai/source/rl-tools-framework/zoo.rl.tools/` | Model zoo frontend |
| studio.rl.tools | `/home/ai/source/rl-tools-framework/studio.rl.tools/` | Environment UI designer |
| l2f-studio | `/home/ai/source/rl-tools-framework/l2f-studio/` | Browser drone simulator |
| raptor.rl.tools | `/home/ai/source/rl-tools-framework/raptor.rl.tools/` | Trajectory analysis |
| extrack-ui-lib | `/home/ai/source/rl-tools-framework/extrack-ui-lib/` | Reusable JS components |
| rl-tools.github.io | `/home/ai/source/rl-tools-framework/rl-tools.github.io/` | Landing page + demos |

## Environment Render Functions

Every rl-tools environment can provide a browser visualization via the `get_ui()` function in `operations_cpu.h`. This returns an ES6 module string.

### 2D Canvas Pattern

```cpp
template <typename DEVICE, typename SPEC>
std::string get_ui(DEVICE& device, MyEnv<SPEC>& env) {
    return R"RL_TOOLS_LITERAL(
export async function init(canvas, options) {
    return { ctx: canvas.getContext('2d') };
}
export async function render(ui_state, parameters, state, action) {
    const ctx = ui_state.ctx;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Draw environment elements using Canvas 2D API
    const centerX = w / 2;
    const centerY = h / 2;

    // Example: draw an agent
    ctx.beginPath();
    ctx.arc(centerX + state.x * 50, centerY, 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#7DB9B6';
    ctx.fill();
    ctx.stroke();
}
    )RL_TOOLS_LITERAL";
}
```

### 3D Three.js Pattern

```cpp
template <typename DEVICE, typename SPEC>
std::string get_ui(DEVICE& device, MyEnv<SPEC>& env) {
    return R"RL_TOOLS_LITERAL(
export async function init(canvas, options) {
    const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, canvas.width / canvas.height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas });
    renderer.setSize(canvas.width, canvas.height);
    camera.position.z = 5;

    // Create objects
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshPhongMaterial({ color: 0x7DB9B6 });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    // Lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5, 5, 5);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x404040));

    return { THREE, scene, camera, renderer, mesh };
}
export async function render(ui_state, parameters, state, action) {
    const { scene, camera, renderer, mesh } = ui_state;
    mesh.position.set(state.x, state.y, state.z);
    mesh.rotation.set(state.roll, state.pitch, state.yaw);
    renderer.render(scene, camera);
}
    )RL_TOOLS_LITERAL";
}
```

### Interactive Development

Use **studio.rl.tools** for interactive render function development:
- `/home/ai/source/rl-tools-framework/studio.rl.tools/`
- ACE editor for live JavaScript editing
- Canvas preview with hot-reload
- Import/export render functions

## WebAssembly Compilation

Compile C++ environments and inference to run in the browser:

```bash
# Using Emscripten
emcmake cmake -B build_wasm -DCMAKE_BUILD_TYPE=Release
cmake --build build_wasm
```

Device for WASM: `rlt::devices::wasm32`

The l2f-studio demonstrates the full pattern:
- C++ inference compiled to WASM
- Three.js rendering
- JavaScript ↔ WASM data marshaling

## Chart.js Integration

For learning curve visualization:

```javascript
import Chart from 'chart.js/auto';

// Load return data
const response = await fetch('experiments/.../return.json');
const data = await response.json();

new Chart(canvas, {
    type: 'line',
    data: {
        labels: data.steps,
        datasets: [{
            label: 'Return',
            data: data.returns,
            borderColor: '#4A90D9',
            fill: false
        }]
    },
    options: {
        scales: {
            x: { title: { display: true, text: 'Training Steps' } },
            y: { title: { display: true, text: 'Episode Return' } }
        }
    }
});
```

## ExTrack UI Library

**Location**: `/home/ai/source/rl-tools-framework/extrack-ui-lib/`

Reusable components:
- **Chart.js wrapper** — Learning curves with seed aggregation
- **Three.js wrapper** — 3D scene management
- **ACE editor** — In-browser code editing
- **pako** — zlib decompression for `.json.gz` trajectory files

## Gamepad Support

For interactive control (used in raptor.rl.tools):

```javascript
window.addEventListener('gamepadconnected', (e) => {
    const gamepad = e.gamepad;
    function update() {
        const gp = navigator.getGamepads()[gamepad.index];
        const axes = gp.axes;  // Stick positions [-1, 1]
        // Map to control inputs
        requestAnimationFrame(update);
    }
    update();
});
```

## UI Server (C++ Real-Time)

For real-time visualization during training:

**Location**: `/home/ai/source/rl-tools-framework/rl-tools/include/rl_tools/ui_server/`

WebSocket-based server that pushes state updates to connected browser clients during training.

## Key Dependencies

| Library | Purpose | CDN |
|---------|---------|-----|
| Chart.js | Learning curves | `chart.js/auto` |
| Three.js | 3D rendering | `three@0.160.0` |
| ACE | Code editor | `ace-builds` |
| pako | zlib decompression | `pako` |
