// Unity Cube: the whole Numerai v6.0 dataset as a GPU-resident uint8 volume, viewed as three slices.
//   X = ticker, +Y = feature, -Y = target, Z = date. Every cell is one byte on a shared colour ramp.

const $ = (id) => document.getElementById(id);
const DATA = "../data/cube/";
const TEX_W = 8192;                      // texture row width (the guaranteed WebGPU 2D limit)
const STAGE = 64 << 20;                  // upload staging size
const BOX = { WX: 4, WF: 1.2, WT: 0.18, WZ: 2 };  // world extents: tickers, features, targets, dates
const BIN_BYTES = [0, 64, 127, 191, 254];
const RAMP = ["#7fb0ee", "#3a73bd", "#4a4a47", "#bd4744", "#f0948d"];  // low -> high, grey midpoint
const NULL_FG = "#898781", NULL_BG = "#2a2a28", EMPTY = "#222220", LOADING = "#141413";
const BG = [0x0d / 255, 0x0d / 255, 0x0d / 255];

const setStatus = (s) => { $("status").textContent = s; };
const setProgress = (f) => { $("bar").firstElementChild.style.width = `${(f * 100).toFixed(1)}%`; };
function fail(html) {
  $("err").style.display = "block";
  $("err").innerHTML = html;
  setStatus("error");
  throw new Error(html);
}

// ---------- WebGPU device ----------
if (!navigator.gpu) fail("WebGPU is not available in this browser. On Linux Chrome try <code>--enable-unsafe-webgpu --enable-features=Vulkan</code>.");
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) fail("No WebGPU adapter found.");

const meta = await (await fetch(DATA + "meta.json")).json();
// ?dates=N loads only the first N dates (for testing on small or software GPUs)
const dateParam = +new URLSearchParams(location.search).get("dates");
const T = meta.tickers.length, D = dateParam > 0 ? Math.min(dateParam, meta.dates.length) : meta.dates.length, NF = meta.features.length, NT = meta.targets.length;
const NC = NF + NT, L = Math.ceil(T / TEX_W), offs = meta.row_offsets;
const columns = meta.columns.map((c) => c.name);

// Pack whole dates into storage buffers no larger than the device allows to bind.
const bindLimit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize, 2 ** 31);
const chunkLimit = Math.floor(bindLimit / 4) * 4;
const dateChunk = new Uint8Array(D), dateRow0 = new Uint32Array(D), dateEnd = new Float64Array(D);
const chunks = [];
for (let d = 0, cur = null; d < D; d++) {
  const rows = offs[d + 1] - offs[d], bytes = rows * NC;
  if (!cur || cur.bytes + bytes > chunkLimit) chunks.push(cur = { bytes: 0, rows: 0 });
  dateChunk[d] = chunks.length - 1;
  dateRow0[d] = cur.rows;
  cur.rows += rows;
  cur.bytes += bytes;
  dateEnd[d] = cur.bytes;
}
const K = chunks.length;
if (K > adapter.limits.maxStorageBuffersPerShaderStage) {
  fail(`Needs ${K} storage buffers of ≤${(chunkLimit / 2 ** 30).toFixed(2)} GiB but the GPU allows ${adapter.limits.maxStorageBuffersPerShaderStage}.`);
}
const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
  },
});
device.lost.then((info) => fail(`GPU device lost: ${info.message}`));
device.addEventListener("uncapturederror", (e) => console.error("WebGPU:", e.error.message));
const gpuName = adapter.info?.description || adapter.info?.architecture || "GPU";

// ---------- row index: (ticker, date) -> chunk<<24 | row within chunk, -1 = no row ----------
setStatus("building index…");
const rowTicker = new Uint16Array(await (await fetch(DATA + "row_ticker.u16")).arrayBuffer());
const index = new Int32Array(TEX_W * D * L).fill(-1);
for (let d = 0; d < D; d++) {
  const c = dateChunk[d], r0 = dateRow0[d], base = offs[d];
  for (let r = base; r < offs[d + 1]; r++) {
    const t = rowTicker[r];
    index[(((t / TEX_W) | 0) * D + d) * TEX_W + (t % TEX_W)] = (c << 24) | (r0 + r - base);
  }
}
const indexAt = (t, d) => index[(((t / TEX_W) | 0) * D + d) * TEX_W + (t % TEX_W)];
const idxTex = device.createTexture({ size: [TEX_W, D, L], format: "r32sint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
device.queue.writeTexture({ texture: idxTex }, index, { bytesPerRow: TEX_W * 4, rowsPerImage: D }, [TEX_W, D, L]);

// ---------- ticker orderings (position on X -> ticker id) ----------
const permTex = device.createTexture({ size: [TEX_W, L], format: "r32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
let perm = new Uint32Array(TEX_W * L), inv = new Uint32Array(T);
function setOrder(kind) {
  const ids = Array.from({ length: T }, (_, i) => i);  // tickers are stored alphabetically
  const fs = meta.first_seen, co = meta.country;
  if (kind === "first") ids.sort((a, b) => (fs[a] < fs[b] ? -1 : fs[a] > fs[b] ? 1 : a - b));
  if (kind === "country") ids.sort((a, b) => (co[a] < co[b] ? -1 : co[a] > co[b] ? 1 : fs[a] < fs[b] ? -1 : fs[a] > fs[b] ? 1 : a - b));
  perm = new Uint32Array(TEX_W * L);
  ids.forEach((id, pos) => { perm[pos] = id; inv[id] = pos; });
  device.queue.writeTexture({ texture: permTex }, perm, { bytesPerRow: TEX_W * 4 }, [TEX_W, L]);
}

// ---------- value buffers ----------
device.pushErrorScope("out-of-memory");
const chunkBufs = chunks.map((c) => device.createBuffer({
  size: Math.ceil(c.bytes / 4) * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
}));
const oom = await device.popErrorScope();
if (oom) fail(`Not enough GPU memory for ${(chunks.reduce((s, c) => s + c.bytes, 0) / 2 ** 30).toFixed(1)} GiB of cells: ${oom.message}`);

// ---------- colours ----------
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const colors = new Float32Array(260 * 4);
for (let v = 0; v <= 254; v++) {
  let i = 0;
  while (i < 3 && v > BIN_BYTES[i + 1]) i++;
  const f = (v - BIN_BYTES[i]) / (BIN_BYTES[i + 1] - BIN_BYTES[i]);
  const a = hex(RAMP[i]), b = hex(RAMP[i + 1]);
  colors.set([0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * f).concat(1), v * 4);
}
[[255, NULL_FG], [256, EMPTY], [257, NULL_BG], [258, LOADING]].forEach(([i, h]) => colors.set([...hex(h), 1], i * 4));

// ---------- shaders ----------
const UNIFORM_WGSL = /* wgsl */`
struct U {
  viewProj: mat4x4f,
  box: vec4f,     // WX, WF, WT, WZ
  slice: vec4f,   // ticker pos centre, signed Y cell centre (targets < 0), date centre
  dims: vec4u,    // T, NF, NT, D
  misc: vec4u,    // NC, flags (1 hide empty, 2/4/8 date/feature/ticker plane visible), max samples, loaded dates
  hover: vec4i,   // ticker pos, column, date, active
  colors: array<vec4f, 260>,  // 0..254 ramp, 255 null, 256 empty, 257 null background, 258 loading
};
@group(0) @binding(0) var<uniform> u: U;
`;
const chunkDecls = chunks.map((_, k) => `@group(0) @binding(${3 + k}) var<storage, read> c${k}: array<u32>;`).join("\n");
const chunkSwitch = chunks.map((_, k) => `if (c == ${k}u) { return c${k}[w]; }`).join(" else ") + " return 0u;";

const planeWGSL = UNIFORM_WGSL + /* wgsl */`
@group(0) @binding(1) var idxTex: texture_2d_array<i32>;
@group(0) @binding(2) var permTex: texture_2d<u32>;
${chunkDecls}

fn word(c: u32, w: u32) -> u32 { ${chunkSwitch} }

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) @interpolate(flat) plane: u32,
};

fn sliceWorld() -> vec3f {
  let y = select(u.slice.y / f32(u.dims.z) * u.box.z, u.slice.y / f32(u.dims.y) * u.box.y, u.slice.y >= 0.0);
  return vec3f(u.slice.x / f32(u.dims.x) * u.box.x, y, u.slice.z / f32(u.dims.w) * u.box.w);
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) plane: u32) -> VOut {
  var corner = array<vec2f, 6>(vec2f(0, 0), vec2f(1, 0), vec2f(0, 1), vec2f(0, 1), vec2f(1, 0), vec2f(1, 1));
  let a = corner[vi].x;
  let b = corner[vi].y;
  let s = sliceWorld();
  let y = mix(-u.box.z, u.box.y, select(b, a, plane == 2u));
  var p: vec3f;
  if (plane == 0u) { p = vec3f(a * u.box.x, y, s.z); }
  else if (plane == 1u) { p = vec3f(a * u.box.x, s.y, b * u.box.w); }
  else { p = vec3f(s.x, y, b * u.box.w); }
  var o: VOut;
  o.pos = u.viewProj * vec4f(p, 1.0);
  o.world = p;
  o.plane = plane;
  return o;
}

// world -> continuous cell coords (ticker pos, signed Y cell, date)
fn toCell(w: vec3f) -> vec3f {
  let y = select(w.y / u.box.z * f32(u.dims.z), w.y / u.box.y * f32(u.dims.y), w.y >= 0.0);
  return vec3f(w.x / u.box.x * f32(u.dims.x), y, w.z / u.box.w * f32(u.dims.w));
}

fn cellIds(q: vec3f) -> vec3u {
  let tp = u32(clamp(floor(q.x), 0.0, f32(u.dims.x - 1u)));
  var col: u32;
  if (q.y >= 0.0) { col = u32(min(floor(q.y), f32(u.dims.y - 1u))); }
  else { col = u.dims.y + u32(min(floor(-q.y), f32(u.dims.z - 1u))); }
  let d = u32(clamp(floor(q.z), 0.0, f32(u.dims.w - 1u)));
  return vec3u(tp, col, d);
}

// 0..255 = stored byte, 256 = no row for this ticker/date, 258 = not loaded yet
fn cellValue(id: vec3u) -> u32 {
  if (id.z >= u.misc.w) { return 258u; }
  let t = textureLoad(permTex, vec2i(i32(id.x % 8192u), i32(id.x / 8192u)), 0).r;
  let packed = textureLoad(idxTex, vec2i(i32(t % 8192u), i32(id.z)), i32(t / 8192u), 0).r;
  if (packed < 0) { return 256u; }
  let c = u32(packed) >> 24u;
  let b = (u32(packed) & 0xFFFFFFu) * u.misc.x + id.y;
  return (word(c, b >> 2u) >> ((b & 3u) * 8u)) & 0xFFu;
}

fn colorOf(v: u32, q: vec3f) -> vec3f {
  if (v == 255u) {  // null value: diagonal hatch
    return select(u.colors[257].rgb, u.colors[255].rgb, fract((q.x + q.y + q.z) * 2.0) < 0.35);
  }
  return u.colors[v].rgb;
}

@fragment fn fs(i: VOut) -> @location(0) vec4f {
  var q = toCell(i.world);
  if (i.plane == 0u) { q.z = u.slice.z; } else if (i.plane == 1u) { q.y = u.slice.y; } else { q.x = u.slice.x; }
  let dx = dpdx(q);
  let dy = dpdy(q);
  // Average n x n samples over the pixel footprint when a pixel covers many cells.
  let n = u32(clamp(ceil(max(length(dx), length(dy))), 1.0, f32(u.misc.z)));
  var acc = vec3f(0.0);
  var filled = 0.0;
  for (var a = 0u; a < n; a++) {
    for (var b = 0u; b < n; b++) {
      let o = (vec2f(f32(a), f32(b)) + 0.5) / f32(n) - 0.5;
      let qs = q + dx * o.x + dy * o.y;
      let v = cellValue(cellIds(qs));
      acc += colorOf(v, qs);
      filled += select(1.0, 0.0, v == 256u);
    }
  }
  let nn = f32(n * n);
  if ((u.misc.y & 1u) != 0u && filled < nn * 0.5) { discard; }
  var col = acc / nn;
  let gx = length(vec2f(dx.x, dy.x));
  let gy = length(vec2f(dx.y, dy.y));
  let gz = length(vec2f(dx.z, dy.z));
  if (u.hover.w != 0 && all(vec3i(cellIds(q)) == u.hover.xyz)) {
    // Outline the hovered cell when it is big on screen, otherwise tint it.
    let f = fract(q);
    let px = min(f, 1.0 - f) / max(vec3f(gx, gy, gz), vec3f(1e-9));  // pixels to the cell border per axis
    let size = 1.0 / max(vec3f(gx, gy, gz), vec3f(1e-9));            // cell size in pixels per axis
    var edge: f32; var cell: f32;
    if (i.plane == 0u) { edge = min(px.x, px.y); cell = min(size.x, size.y); }
    else if (i.plane == 1u) { edge = min(px.x, px.z); cell = min(size.x, size.z); }
    else { edge = min(px.y, px.z); cell = min(size.y, size.z); }
    if (cell < 8.0) { col = mix(col, vec3f(1.0), 0.6); }
    else if (edge < 2.0) { col = vec3f(1.0); }
  }
  // Lines where the other visible slices cut this one.
  var line = false;
  if (i.plane != 2u && (u.misc.y & 8u) != 0u && abs(q.x - u.slice.x) < 0.8 * gx) { line = true; }
  if (i.plane != 1u && (u.misc.y & 4u) != 0u && abs(q.y - u.slice.y) < 0.8 * gy) { line = true; }
  if (i.plane != 0u && (u.misc.y & 2u) != 0u && abs(q.z - u.slice.z) < 0.8 * gz) { line = true; }
  if (line) { col = mix(col, vec3f(0.94, 0.94, 0.92), 0.7); }
  return vec4f(col, 1.0);
}
`;

const lineWGSL = UNIFORM_WGSL + /* wgsl */`
@vertex fn vs(@location(0) p: vec3f) -> @builtin(position) vec4f { return u.viewProj * vec4f(p, 1.0); }
@fragment fn fs() -> @location(0) vec4f { return vec4f(0.22, 0.22, 0.21, 1.0); }
`;

const canvas = $("gpu");
const ctx = canvas.getContext("webgpu");
const format = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format, alphaMode: "opaque" });
const depth = { format: "depth32float", depthWriteEnabled: true, depthCompare: "greater" };  // reversed Z

const planeModule = device.createShaderModule({ code: planeWGSL });
const planePipe = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: planeModule, entryPoint: "vs" },
  fragment: { module: planeModule, entryPoint: "fs", targets: [{ format }] },
  primitive: { topology: "triangle-list", cullMode: "none" },
  depthStencil: depth,
});
const lineModule = device.createShaderModule({ code: lineWGSL });
const linePipe = device.createRenderPipeline({
  layout: "auto",
  vertex: { module: lineModule, entryPoint: "vs", buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] }] },
  fragment: { module: lineModule, entryPoint: "fs", targets: [{ format }] },
  primitive: { topology: "line-list" },
  depthStencil: depth,
});

const UBYTES = 144 + 260 * 16;
const ubuf = device.createBuffer({ size: UBYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const uData = new ArrayBuffer(UBYTES), uF = new Float32Array(uData), uU = new Uint32Array(uData), uI = new Int32Array(uData);
uF.set(colors, 144 / 4);

const planeBG = device.createBindGroup({
  layout: planePipe.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: ubuf } },
    { binding: 1, resource: idxTex.createView({ dimension: "2d-array" }) },
    { binding: 2, resource: permTex.createView() },
    ...chunkBufs.map((buffer, k) => ({ binding: 3 + k, resource: { buffer } })),
  ],
});
const lineBG = device.createBindGroup({ layout: linePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ubuf } }] });

// Box outline plus the feature/target boundary at y = 0.
const { WX, WF, WT, WZ } = BOX;
const boxPts = [];
const edge = (a, b) => boxPts.push(...a, ...b);
for (const y of [-WT, WF]) for (const z of [0, WZ]) edge([0, y, z], [WX, y, z]);
for (const x of [0, WX]) for (const z of [0, WZ]) edge([x, -WT, z], [x, WF, z]);
for (const x of [0, WX]) for (const y of [-WT, WF]) edge([x, y, 0], [x, y, WZ]);
edge([0, 0, 0], [WX, 0, 0]); edge([0, 0, WZ], [WX, 0, WZ]); edge([0, 0, 0], [0, 0, WZ]); edge([WX, 0, 0], [WX, 0, WZ]);
const boxBuf = device.createBuffer({ size: boxPts.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(boxBuf, 0, new Float32Array(boxPts));

// ---------- state ----------
const state = {
  date: 0, col: 0, tick: 0,         // tick = ticker id
  show: [true, true, true],         // date, feature, ticker planes
  hideEmpty: false, quality: 5, loadedD: 0,
  hover: null,                      // {tp, col, d}
  interacting: false,
};
const cam = { target: [WX / 2, (WF - WT) / 2, WZ / 2], dist: 6.4, yaw: 0.62, pitch: 0.38, fov: (40 * Math.PI) / 180 };
const camHome = structuredClone(cam);

const sliceCell = () => [
  inv[state.tick] + 0.5,
  state.col < NF ? state.col + 0.5 : -(state.col - NF + 0.5),
  state.date + 0.5,
];

// ---------- math ----------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => scale(a, 1 / Math.hypot(...a));

function camBasis() {
  const cp = Math.cos(cam.pitch);
  const eye = add(cam.target, scale([cp * Math.sin(cam.yaw), Math.sin(cam.pitch), cp * Math.cos(cam.yaw)], cam.dist));
  const fwd = norm(sub(cam.target, eye));
  const right = norm(cross(fwd, [0, 1, 0]));
  return { eye, fwd, right, up: cross(right, fwd) };
}

function viewProj() {
  const { eye, fwd, right, up } = camBasis();
  const aspect = canvas.width / canvas.height;
  const f = 1 / Math.tan(cam.fov / 2), n = cam.dist * 0.01, fa = cam.dist * 10 + 20;
  // column-major view (rows = right, up, -fwd) then reversed-Z perspective
  const V = [right[0], up[0], -fwd[0], 0, right[1], up[1], -fwd[1], 0, right[2], up[2], -fwd[2], 0,
    -dot(right, eye), -dot(up, eye), dot(fwd, eye), 1];
  const P = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, n / (fa - n), -1, 0, 0, (n * fa) / (fa - n), 0];
  const M = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += P[k * 4 + r] * V[c * 4 + k];
    M[c * 4 + r] = s;
  }
  return M;
}

function project(M, p) {
  const x = M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12];
  const y = M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13];
  const w = M[3] * p[0] + M[7] * p[1] + M[11] * p[2] + M[15];
  if (w <= 0) return null;
  return [(x / w * 0.5 + 0.5) * canvas.clientWidth, (0.5 - y / w * 0.5) * canvas.clientHeight];
}

// JS mirror of the shader's coordinate mapping
const toCell = (w) => [w[0] / WX * T, w[1] >= 0 ? w[1] / WF * NF : w[1] / WT * NT, w[2] / WZ * D];
const cellIds = (q) => {
  const tp = Math.min(Math.max(Math.floor(q[0]), 0), T - 1);
  const col = q[1] >= 0 ? Math.min(Math.floor(q[1]), NF - 1) : NF + Math.min(Math.floor(-q[1]), NT - 1);
  const d = Math.min(Math.max(Math.floor(q[2]), 0), D - 1);
  return { tp, col, d };
};
const sliceWorld = () => {
  const s = sliceCell();
  return [s[0] / T * WX, s[1] >= 0 ? s[1] / NF * WF : s[1] / NT * WT, s[2] / D * WZ];
};

// Ray from mouse -> nearest visible slice hit
function pick(mx, my) {
  const { eye, fwd, right, up } = camBasis();
  const t = Math.tan(cam.fov / 2), aspect = canvas.clientWidth / canvas.clientHeight;
  const nx = (mx / canvas.clientWidth) * 2 - 1, ny = 1 - (my / canvas.clientHeight) * 2;
  const dir = norm(add(fwd, add(scale(right, nx * t * aspect), scale(up, ny * t))));
  const s = sliceWorld();
  let best = null;
  const tryPlane = (plane, axis) => {
    if (!state.show[plane] || Math.abs(dir[axis]) < 1e-12) return;
    const k = (s[axis] - eye[axis]) / dir[axis];
    if (k <= 0 || (best && k >= best.k)) return;
    const p = add(eye, scale(dir, k));
    if (p[0] < -1e-9 || p[0] > WX + 1e-9 || p[1] < -WT - 1e-9 || p[1] > WF + 1e-9 || p[2] < -1e-9 || p[2] > WZ + 1e-9) return;
    best = { k, p, plane };
  };
  tryPlane(0, 2); tryPlane(1, 1); tryPlane(2, 0);
  if (!best) return null;
  const q = toCell(best.p), sc = sliceCell();
  q[[2, 1, 0][best.plane]] = sc[[2, 1, 0][best.plane]];
  return { ...best, ...cellIds(q) };
}

// ---------- rendering ----------
let depthTex = null, frameQueued = false;
function requestRender() {
  if (!frameQueued) { frameQueued = true; requestAnimationFrame(render); }
}
function render() {
  frameQueued = false;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h || !depthTex) {
    canvas.width = w; canvas.height = h;
    depthTex?.destroy();
    depthTex = device.createTexture({ size: [w, h], format: "depth32float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }
  const M = viewProj();
  uF.set(M, 0);
  uF.set([WX, WF, WT, WZ], 16);
  uF.set([...sliceCell(), 0], 20);
  uU.set([T, NF, NT, D], 24);
  const flags = (state.hideEmpty ? 1 : 0) | (state.show[0] ? 2 : 0) | (state.show[1] ? 4 : 0) | (state.show[2] ? 8 : 0);
  uU.set([NC, flags, state.interacting ? Math.min(2, state.quality) : state.quality, state.loadedD], 28);
  const hv = state.hover;
  uI.set(hv ? [hv.tp, hv.col, hv.d, 1] : [0, 0, 0, 0], 32);
  device.queue.writeBuffer(ubuf, 0, uData);

  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({
    colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: [...BG, 1], loadOp: "clear", storeOp: "store" }],
    depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "discard" },
  });
  pass.setPipeline(linePipe);
  pass.setBindGroup(0, lineBG);
  pass.setVertexBuffer(0, boxBuf);
  pass.draw(boxPts.length / 3);
  pass.setPipeline(planePipe);
  pass.setBindGroup(0, planeBG);
  for (let p = 0; p < 3; p++) if (state.show[p]) pass.draw(6, 1, 0, p);
  pass.end();
  device.queue.submit([enc.finish()]);
  drawLabels(M);
}
new ResizeObserver(requestRender).observe(canvas);

// ---------- labels ----------
const labelsEl = $("labels");
const labelPool = [];
function drawLabels(M) {
  const items = [
    [[WX / 2, -WT, -0.12], "tickers (X) →"],
    [[-0.1, WF / 2, 0], "features ↑"],
    [[-0.1, -WT / 2, 0], "targets ↓"],
    [[-0.32, -WT, WZ / 2], "dates (Z) →"],
  ];
  let lastYear = "";
  meta.dates.forEach((dt, d) => {
    const y = dt.slice(0, 4);
    if (y !== lastYear && +y % 2 === 1) items.push([[-0.04, -WT - 0.04, (d / D) * WZ], y]);
    lastYear = y;
  });
  const s = sliceWorld();
  if (state.show[0]) items.push([[WX + 0.06, WF, s[2]], meta.dates[state.date], true]);
  if (state.show[1]) items.push([[WX + 0.06, s[1], WZ], columns[state.col].replace(/^feature_|^target_/, ""), true]);
  if (state.show[2]) items.push([[s[0], WF + 0.06, WZ], meta.tickers[state.tick], true]);
  items.forEach(([p, text, slice], i) => {
    let el = labelPool[i];
    if (!el) { el = labelPool[i] = document.createElement("div"); labelsEl.appendChild(el); }
    const sp = project(M, p);
    el.style.display = sp ? "" : "none";
    if (!sp) return;
    el.className = slice ? "lbl slice" : "lbl";
    el.textContent = text;
    el.style.left = `${sp[0]}px`;
    el.style.top = `${sp[1]}px`;
  });
  for (let i = items.length; i < labelPool.length; i++) labelPool[i].style.display = "none";
}

// ---------- hover readout (exact byte read back from the GPU) ----------
const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
let reading = false, pendingRead = null;
function describe(v, col) {
  if (v === 255) return "null";
  const c = meta.columns[col];
  if (c.kind === "cont") return `≈ ${(c.min + (v / 254) * (c.max - c.min)).toPrecision(4)}  (range ${c.min.toFixed(2)} … ${c.max.toFixed(2)})`;
  const k = BIN_BYTES.indexOf(v);
  return col < NF ? `bin ${k}` : `${k / 4}`;
}
function showHover(h, text) {
  const t = perm[h.tp], col = columns[h.col];
  $("hover").innerHTML =
    `<div><span class="k">ticker</span>${meta.tickers[t]} · ${meta.country[t]}</div>` +
    `<div><span class="k">date</span>${meta.dates[h.d]} · ${meta.date_type[h.d]}</div>` +
    `<div><span class="k">${h.col < NF ? "feature" : "target"}</span>${col.replace(/^feature_|^target_/, "")}</div>` +
    `<div><span class="k">value</span><span class="val">${text}</span></div>`;
}
async function readHover(h) {
  if (reading) { pendingRead = h; return; }
  reading = true;
  try {
    if (h.d >= state.loadedD) { showHover(h, "loading…"); return; }
    const packed = indexAt(perm[h.tp], h.d);
    if (packed < 0) { showHover(h, "no row (not in universe this week)"); return; }
    const b = (packed & 0xffffff) * NC + h.col;
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(chunkBufs[packed >>> 24], Math.floor(b / 4) * 4, readBuf, 0, 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const v = new Uint8Array(readBuf.getMappedRange())[b % 4];
    readBuf.unmap();
    showHover(h, describe(v, h.col));
  } finally {
    reading = false;
    if (pendingRead) { const p = pendingRead; pendingRead = null; readHover(p); }
  }
}

// ---------- UI ----------
const dateEl = $("date"), colEl = $("col"), tickEl = $("tick");
dateEl.max = D - 1;
colEl.min = -NT; colEl.max = NF - 1;
tickEl.max = T - 1;
const colToSlider = (c) => (c < NF ? c : -(c - NF) - 1);
const sliderToCol = (s) => (s >= 0 ? s : NF - s - 1);
$("colSel").innerHTML =
  `<optgroup label="Targets (−Y)">${meta.targets.map((n, i) => `<option value="${NF + i}">${n}</option>`).join("")}</optgroup>` +
  `<optgroup label="Features (+Y)">${meta.features.map((n, i) => `<option value="${i}">${n}</option>`).join("")}</optgroup>`;
$("tickList").innerHTML = meta.tickers.map((t) => `<option value="${t}">`).join("");
const tickerId = new Map(meta.tickers.map((t, i) => [t, i]));

function syncUI() {
  dateEl.value = state.date;
  $("dateLbl").textContent = `${meta.dates[state.date]} · ${meta.date_type[state.date]}`;
  colEl.value = colToSlider(state.col);
  $("colSel").value = state.col;
  $("colLbl").textContent = state.col < NF ? `feature ${state.col + 1}/${NF}` : `target ${state.col - NF + 1}/${NT}`;
  tickEl.value = inv[state.tick];
  $("tickLbl").textContent = `${meta.tickers[state.tick]} · ${meta.country[state.tick]}`;
  $("qualLbl").textContent = `${state.quality}×${state.quality} samples/pixel`;
  requestRender();
}
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const setDate = (d) => { state.date = clamp(d, 0, D - 1); syncUI(); };
const setCol = (c) => { state.col = clamp(c, 0, NC - 1); syncUI(); };
const setTickPos = (p) => { state.tick = perm[clamp(p, 0, T - 1)]; syncUI(); };

dateEl.oninput = () => setDate(+dateEl.value);
colEl.oninput = () => setCol(sliderToCol(+colEl.value));
$("colSel").onchange = (e) => setCol(+e.target.value);
tickEl.oninput = () => setTickPos(+tickEl.value);
$("tickSearch").onchange = (e) => {
  const id = tickerId.get(e.target.value.trim().toUpperCase());
  if (id !== undefined) { state.tick = id; syncUI(); }
};
[["showDate", 0], ["showCol", 1], ["showTick", 2]].forEach(([id, p]) => {
  $(id).onchange = (e) => { state.show[p] = e.target.checked; syncUI(); };
});
$("order").onchange = (e) => { setOrder(e.target.value); syncUI(); };
$("hideEmpty").onchange = (e) => { state.hideEmpty = e.target.checked; requestRender(); };
$("quality").oninput = (e) => { state.quality = +e.target.value; syncUI(); };

$("legend").innerHTML =
  `<div class="bins">${RAMP.map((c, i) => `<div><span class="sw" style="background:${c}"></span>${i}</div>`).join("")}</div>` +
  `<div class="ends"><span>low</span><span>features: bins 0–4 · targets: 0, .25, .5, .75, 1</span><span>high</span></div>` +
  `<div class="grad" style="background:linear-gradient(90deg,${RAMP.join(",")})"></div>` +
  `<div class="ends"><span>min</span><span>22 continuous factors</span><span>max</span></div>` +
  `<div class="extra"><span><span class="sw" style="background:repeating-linear-gradient(45deg,${NULL_FG} 0 2px,${NULL_BG} 2px 6px)"></span> null</span>` +
  `<span><span class="sw" style="background:${EMPTY};outline:1px solid #383835"></span> no row</span></div>`;

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type === "text") return;
  const step = e.shiftKey ? 10 : 1;
  const keys = {
    "[": () => setDate(state.date - step), "]": () => setDate(state.date + step),
    "{": () => setDate(state.date - step), "}": () => setDate(state.date + step),
    ",": () => setTickPos(inv[state.tick] - step), ".": () => setTickPos(inv[state.tick] + step),
    "<": () => setTickPos(inv[state.tick] - step), ">": () => setTickPos(inv[state.tick] + step),
    ";": () => setCol(sliderToCol(colToSlider(state.col) - step)), "'": () => setCol(sliderToCol(colToSlider(state.col) + step)),
    ":": () => setCol(sliderToCol(colToSlider(state.col) - step)), '"': () => setCol(sliderToCol(colToSlider(state.col) + step)),
    r: () => { Object.assign(cam, structuredClone(camHome)); requestRender(); },
  };
  if (keys[e.key]) { keys[e.key](); e.preventDefault(); }
});

// ---------- mouse ----------
let drag = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  const mode = e.ctrlKey ? "slice" : e.button === 2 || e.shiftKey ? "pan" : "orbit";
  drag = { mode, x: e.offsetX, y: e.offsetY };
  if (mode === "slice") {
    const hit = pick(e.offsetX, e.offsetY);
    if (!hit) { drag = null; return; }
    // Screen-space direction of the plane normal, to convert mouse motion into motion along the axis.
    const axis = [2, 1, 0][hit.plane], n = [0, 0, 0];
    const M = viewProj();
    const p0 = project(M, hit.p);
    n[axis] = 1e-3 * cam.dist;
    const p1 = project(M, add(hit.p, n));
    if (!p0 || !p1) { drag = null; return; }
    Object.assign(drag, { plane: hit.plane, axis, start: hit.p, sdir: sub(p1, p0).concat(0), unit: 1e-3 * cam.dist });
  }
  state.interacting = true;
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) {
    const h = pick(e.offsetX, e.offsetY);
    const key = h && `${h.tp},${h.col},${h.d}`;
    if (key !== state.hoverKey) {
      state.hoverKey = key;
      state.hover = h;
      if (h) readHover(h);
      requestRender();
    }
    return;
  }
  const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
  if (drag.mode === "orbit") {
    cam.yaw -= dx * 0.005;
    cam.pitch = clamp(cam.pitch + dy * 0.005, -1.55, 1.55);
    drag.x = e.offsetX; drag.y = e.offsetY;
  } else if (drag.mode === "pan") {
    const { right, up } = camBasis();
    const k = (2 * cam.dist * Math.tan(cam.fov / 2)) / canvas.clientHeight;
    cam.target = add(cam.target, add(scale(right, -dx * k), scale(up, dy * k)));
    drag.x = e.offsetX; drag.y = e.offsetY;
  } else {
    const sd = drag.sdir, len2 = sd[0] * sd[0] + sd[1] * sd[1];
    if (len2 < 1e-6) return;
    const along = ((dx * sd[0] + dy * sd[1]) / len2) * drag.unit;
    const w = drag.start.slice();
    w[drag.axis] = clamp(w[drag.axis] + along, [0, -WT, 0][drag.axis], [WX, WF, WZ][drag.axis] - 1e-9);
    const id = cellIds(toCell(w));
    if (drag.plane === 0) state.date = id.d;
    else if (drag.plane === 1) state.col = id.col;
    else state.tick = perm[id.tp];
    syncUI();
  }
  requestRender();
});
const endDrag = () => { drag = null; state.interacting = false; requestRender(); };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const f = Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
  const hit = pick(e.offsetX, e.offsetY);
  const anchor = hit ? hit.p : cam.target;
  cam.target = add(anchor, scale(sub(cam.target, anchor), f));
  cam.dist = clamp(cam.dist * f, 1e-5, 50);
  requestRender();
}, { passive: false });

// ---------- initial state, then stream all values onto the GPU ----------
setOrder("first");
state.date = Math.floor(D / 2);
state.col = Math.floor(NF / 2);
state.tick = tickerId.get("AAPL US") ?? perm[0];
syncUI();

const totalBytes = chunks.reduce((s, c) => s + c.bytes, 0);
setStatus(`${gpuName} · loading ${(totalBytes / 2 ** 30).toFixed(2)} GiB into ${K} GPU buffers…`);
const t0 = performance.now();
let maxDateBytes = 0;
for (let d = 0; d < D; d++) maxDateBytes = Math.max(maxDateBytes, (offs[d + 1] - offs[d]) * NC);
const stage = new Uint8Array(STAGE + maxDateBytes + 4);
let fill = 0, chunkI = 0, chunkOff = 0, loadedBytes = 0;
function advanceLoaded() {
  while (state.loadedD < D && (dateChunk[state.loadedD] < chunkI || (dateChunk[state.loadedD] === chunkI && dateEnd[state.loadedD] <= chunkOff))) state.loadedD++;
}
async function flush(final) {
  const n = final ? Math.ceil(fill / 4) * 4 : STAGE;
  if (n === 0) return;
  if (final) stage.fill(0, fill, n);
  device.queue.writeBuffer(chunkBufs[chunkI], chunkOff, stage, 0, n);
  chunkOff += n;
  if (final) fill = 0;
  else { stage.copyWithin(0, STAGE, fill); fill -= STAGE; }
  await device.queue.onSubmittedWorkDone();  // keep the upload queue from ballooning
  advanceLoaded();
  requestRender();
}
const fetchDate = (d) => fetch(`${DATA}values/${String(d).padStart(4, "0")}.u8`).then((r) => {
  if (!r.ok) throw new Error(`values ${d}: HTTP ${r.status}`);
  return r.arrayBuffer();
});
const inflight = new Map();
let next = 0;
const CONCURRENCY = 6;
for (; next < Math.min(CONCURRENCY, D); next++) inflight.set(next, fetchDate(next));
for (let d = 0; d < D; d++) {
  const buf = new Uint8Array(await inflight.get(d));
  inflight.delete(d);
  if (next < D) { inflight.set(next, fetchDate(next)); next++; }
  if (dateChunk[d] !== chunkI) { await flush(true); chunkI = dateChunk[d]; chunkOff = 0; }
  stage.set(buf, fill);
  fill += buf.length;
  loadedBytes += buf.length;
  if (fill >= STAGE) await flush(false);
  if (d % 16 === 0) setProgress(loadedBytes / totalBytes);
}
await flush(true);
state.loadedD = D;
setProgress(1);
const secs = (performance.now() - t0) / 1000;
setStatus(`${gpuName} · ${(totalBytes / 2 ** 30).toFixed(2)} GiB in ${K} buffers · ${offs[D].toLocaleString()} rows × ${NC} columns · loaded in ${secs.toFixed(1)}s`);
requestRender();
