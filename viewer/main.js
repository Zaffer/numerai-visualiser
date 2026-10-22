// Numerai Cube: the whole Numerai v6.0 dataset as a GPU-resident uint8 volume, viewed as three slices.
//   X = ticker, +Y = feature, -Y = target, Z = date. Every cell is one byte on a shared colour ramp.
//   Data comes from the viewer's own parquet files (decoded in workers, nothing is uploaded) or from
//   the copy cached in this browser's origin-private file system after the first load.
import { parquetMetadataAsync } from "./vendor/hyparquet/index.js";
import { KIND, byteOfNibble, columnLayout, packColumn, taskBytes } from "./layout.js";

const $ = (id) => document.getElementById(id);
const TEX_W = 8192;                      // texture row width (the guaranteed WebGPU 2D limit)
const STAGE = 64 << 20;                  // upload slice size when reading the cache
const MAX_TASKS = 256, MAX_TASK_ROWS = 1 << 20;   // index packs task << 20 | row
const CACHE_DIR = "numerai-cube", CACHE_VERSION = 2;  // 2: nibble-packed cells (layout.js)
const BOX = { WX: 4, WF: 1.2, WT: 0.18, WZ: 2 };  // world extents: tickers, features, targets, dates
const BIN_BYTES = [0, 64, 127, 191, 254];
const PALETTES = {  // low -> high
  blueRed: ["#7fb0ee", "#3a73bd", "#4a4a47", "#bd4744", "#f0948d"],
  redGreen: ["#f7746d", "#b3423d", "#4a4a47", "#2a8a52", "#3fc27b"],
  viridis: ["#472d7b", "#3b528b", "#21918c", "#5ec962", "#fde725"],  // monotone lightness, colour-blind safe
};
let palette = "blueRed";
try { if (PALETTES[localStorage.getItem("palette")]) palette = localStorage.getItem("palette"); } catch {}
const NULL_FG = "#898781", NULL_BG = "#2a2a28", EMPTY = "#222220";
const BG = [0x0d / 255, 0x0d / 255, 0x0d / 255];

const setStatus = (s) => { $("status").textContent = s; $("popStatus").textContent = s; };
const setProgress = (f) => {
  for (const id of ["bar", "popBar"]) $(id).firstElementChild.style.width = `${(f * 100).toFixed(1)}%`;
};
const fmt = (n) => n.toLocaleString();
const gib = (b) => `${(b / 2 ** 30).toFixed(2)} GiB`;
function fail(html) {
  $("start").style.display = "none";
  $("err").style.display = "block";
  $("err").innerHTML = html;
  setStatus("error");
  throw new Error(html);
}

// ---------- WebGPU adapter ----------
if (!navigator.gpu) fail("WebGPU is not available in this browser. Use a recent Chrome or Edge; on Linux, start Chrome with <code>--enable-unsafe-webgpu --enable-features=Vulkan</code>.");
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) fail("No WebGPU adapter found.");
const gpuName = adapter.info?.description || adapter.info?.architecture || "GPU";
// Values are split into storage buffers ("chunks") no larger than the device can bind.
const bindLimit = Math.floor(Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize, 2 ** 31) / 4) * 4;
const maxChunks = adapter.limits.maxStorageBuffersPerShaderStage - 1;  // one more storage buffer holds the column layout

let device = null, stopViewer = null;
// A new device per dataset: the previous viewer is stopped and its device (and GPU memory) released first.
async function createDevice(chunkBytes) {
  if (chunkBytes.length > maxChunks) {
    throw new Error(`Needs ${chunkBytes.length} GPU buffers of ≤${gib(bindLimit)}, but this GPU allows ${maxChunks}. Try fewer files.`);
  }
  stopViewer?.();
  stopViewer = null;
  // An adapter makes one device, so later datasets need a fresh adapter.
  const source = device ? await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }) : adapter;
  device?.destroy();
  device = null;
  if (!source) throw new Error("No WebGPU adapter found.");
  device = await source.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: source.limits.maxStorageBufferBindingSize,
      maxBufferSize: source.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: source.limits.maxStorageBuffersPerShaderStage,
    },
  });
  device.lost.then((info) => { if (info.reason !== "destroyed") fail(`GPU device lost: ${info.message}. Reload to try again.`); });
  device.addEventListener("uncapturederror", (e) => console.error("WebGPU:", e.error.message));
  device.pushErrorScope("out-of-memory");
  const bufs = chunkBytes.map((size) => device.createBuffer({
    size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  }));
  const oom = await device.popErrorScope();
  if (oom) throw new Error(`Not enough GPU memory for ${gib(chunkBytes.reduce((a, b) => a + b, 0))} of cells (${oom.message}). Try fewer files, e.g. validation only.`);
  return bufs;
}

// Greedily pack task byte sizes into chunks.
function packChunks(sizes) {
  const chunk = [], base = [], bytes = [];
  for (const s of sizes) {
    if (!bytes.length || bytes.at(-1) + s > bindLimit) bytes.push(0);
    chunk.push(bytes.length - 1);
    base.push(bytes.at(-1));
    bytes[bytes.length - 1] += s;
  }
  return { chunk, base, bytes };
}

// ---------- browser cache (origin-private file system) ----------
async function cacheDir(create) {
  try { return await (await navigator.storage.getDirectory()).getDirectoryHandle(CACHE_DIR, { create }); }
  catch { return null; }
}
async function clearCache() {
  try { await (await navigator.storage.getDirectory()).removeEntry(CACHE_DIR, { recursive: true }); } catch { /* none */ }
}
async function readCacheMeta() {
  const dir = await cacheDir(false);
  if (!dir) return null;
  try {
    const meta = JSON.parse(await (await (await dir.getFileHandle("meta.json")).getFile()).text());
    return meta.version === CACHE_VERSION ? meta : null;
  } catch { return null; }
}
// meta.json is written last, so an interrupted save is never mistaken for a cache.
async function openCacheWriter(chunkBytes) {
  if (!navigator.storage?.getDirectory) return null;
  await clearCache();
  const need = chunkBytes.reduce((a, b) => a + b, 0) + (256 << 20);
  const { quota = 0, usage = 0 } = await navigator.storage.estimate();
  if (quota - usage < need) return null;
  navigator.storage.persist?.();
  const dir = await cacheDir(true);
  const writers = await Promise.all(chunkBytes.map(async (_, k) => (await dir.getFileHandle(`chunk_${k}.bin`, { create: true })).createWritable()));
  const writeFile = async (name, data) => {
    const w = await (await dir.getFileHandle(name, { create: true })).createWritable();
    await w.write(data);
    await w.close();
  };
  return {
    write: (k, position, data) => writers[k].write({ type: "write", position, data }),
    async finish(meta, index) {
      await Promise.all(writers.map((w) => w.close()));
      await writeFile("index.bin", index);
      await writeFile("meta.json", JSON.stringify(meta));
    },
    async abort() {
      await Promise.all(writers.map((w) => w.abort().catch(() => {})));
      await clearCache();
    },
  };
}

// ---------- source 1: the user's parquet files ----------
const fileBuffer = (file) => ({ byteLength: file.size, slice: (s, e) => file.slice(s, e).arrayBuffer() });

// Run tasks on a pool of module workers; onResult is awaited before that worker gets more work.
function runPool(count, job, onResult) {
  const n = Math.min(count, Math.max(1, Math.min(8, Math.floor((navigator.hardwareConcurrency || 4) * 0.75))));
  const workers = Array.from({ length: n }, () => new Worker(new URL("./ingest-worker.js", import.meta.url), { type: "module" }));
  let next = 0;
  return Promise.all(workers.map((w) => new Promise((resolve, reject) => {
    const feed = () => {
      if (next >= count) { w.terminate(); resolve(); return; }
      w.postMessage(job(next++));
    };
    w.onmessage = async (e) => {
      if (e.data.error) { reject(new Error(e.data.error)); return; }
      try { await onResult(e.data); feed(); } catch (err) { reject(err); }
    };
    w.onerror = (e) => reject(new Error(e.message || "worker failed"));
    feed();
  }))).finally(() => workers.forEach((w) => w.terminate()));
}

async function ingestParquet(files) {
  files = [...files].sort((a, b) => a.name.localeCompare(b.name));
  setStatus("reading file footers…");
  const infos = [];
  for (const file of files) infos.push({ file, md: await parquetMetadataAsync(fileBuffer(file)) });

  // Columns come from the first file: features (minus the text country) then targets (minus the `target` alias).
  const leaves = infos[0].md.schema.slice(1);
  for (const { file, md } of infos) {
    const names = new Set(md.schema.map((e) => e.name));
    if (!names.has("numerai_ticker") || !names.has("date")) {
      throw new Error(`${file.name} isn't a Numerai v6 file (it has no numerai_ticker/date columns).`);
    }
  }
  const features = leaves.filter((e) => e.name.startsWith("feature_") && e.name !== "feature_country").map((e) => e.name);
  const targets = leaves.filter((e) => e.name.startsWith("target_")).map((e) => e.name);
  for (const { file, md } of infos.slice(1)) {
    const names = new Set(md.schema.map((e) => e.name));
    const missing = [...features, ...targets].find((c) => !names.has(c));
    if (missing) throw new Error(`${file.name} has different columns from ${infos[0].file.name} (no ${missing}).`);
  }
  const typeOf = new Map(leaves.map((e) => [e.name, e.type]));
  const columns = [...features, ...targets].map((name) =>
    name.startsWith("target_") ? { name, kind: "target" }
      : typeOf.get(name) === "INT32" ? { name, kind: "bin" }
      : { name, kind: "cont", min: Infinity, max: -Infinity });

  // Continuous min/max over every row group, straight from the footer statistics.
  const cont = new Map(columns.filter((c) => c.kind === "cont").map((c) => [c.name, c]));
  for (const { md } of infos) for (const rg of md.row_groups) for (const cc of rg.columns) {
    const col = cont.get(cc.meta_data?.path_in_schema?.[0]);
    const st = cc.meta_data?.statistics;
    if (!col || !st) continue;
    if (Number.isFinite(Number(st.min_value))) col.min = Math.min(col.min, Number(st.min_value));
    if (Number.isFinite(Number(st.max_value))) col.max = Math.max(col.max, Number(st.max_value));
  }
  for (const c of cont.values()) {
    if (!(c.max > c.min)) { console.warn(`no statistics for ${c.name}; assuming -4..4`); c.min = -4; c.max = 4; }
  }

  // One task per row group, split if a row group would not fit in one GPU buffer.
  const kinds = Uint8Array.from(columns, (c) => KIND[c.kind]);
  const { perRow } = columnLayout(kinds);
  const maxRows = Math.min(MAX_TASK_ROWS, Math.floor((bindLimit * 2) / perRow));
  const tasks = [];
  for (const { file, md } of infos) {
    let start = 0;
    for (const rg of md.row_groups) {
      const n = Number(rg.num_rows);
      for (let s = 0; s < n; s += maxRows) tasks.push({ file, rowStart: start + s, rows: Math.min(maxRows, n - s) });
      start += n;
    }
  }
  if (tasks.length > MAX_TASKS) throw new Error(`Too many row groups (${tasks.length}); at most ${MAX_TASKS} are supported.`);
  const totalRows = tasks.reduce((a, t) => a + t.rows, 0);
  const plan = packChunks(tasks.map((t) => taskBytes(perRow, t.rows)));
  const chunkBufs = await createDevice(plan.bytes);
  const cache = $("useCache").checked ? await openCacheWriter(plan.bytes) : null;

  const t0 = performance.now();
  const lo = Float64Array.from(columns, (c) => c.min ?? 0), hi = Float64Array.from(columns, (c) => c.max ?? 1);
  const names = columns.map((c) => c.name);
  const results = new Array(tasks.length);
  let doneRows = 0;
  try {
    await runPool(tasks.length,
      (i) => ({ i, file: tasks[i].file, rowStart: tasks[i].rowStart, rowEnd: tasks[i].rowStart + tasks[i].rows, columns: names, kinds, lo, hi }),
      async (res) => {
        const i = res.i, bytes = new Uint8Array(res.bytes);
        device.queue.writeBuffer(chunkBufs[plan.chunk[i]], plan.base[i], bytes);
        if (cache) await cache.write(plan.chunk[i], plan.base[i], bytes);
        await device.queue.onSubmittedWorkDone();  // keep queued uploads bounded
        delete res.bytes;
        results[i] = res;
        doneRows += tasks[i].rows;
        setProgress(doneRows / totalRows);
        const rate = doneRows / ((performance.now() - t0) / 1000);
        setStatus(`decoding ${fmt(doneRows)} / ${fmt(totalRows)} rows · ${fmt(Math.round(rate))} rows/s`);
      });
  } catch (e) {
    await cache?.abort();
    throw e;
  }

  // Global tickers and dates, then the (ticker, date) -> task/row index.
  setStatus("building index…");
  const dateTypeOf = new Map();
  for (const r of results) r.dateNames.forEach((d, k) => { if (!dateTypeOf.has(d)) dateTypeOf.set(d, r.dateTypes[k]); });
  const dates = [...dateTypeOf.keys()].sort();
  const dateIdx = new Map(dates.map((d, i) => [d, i]));
  const tickerId = new Map(), tickers = [], country = [], countryAt = [], firstSeen = [];
  for (const r of results) {
    r.gTick = new Uint32Array(r.tickNames.length);
    r.tickNames.forEach((name, k) => {
      let id = tickerId.get(name);
      if (id === undefined) {
        id = tickers.length;
        tickerId.set(name, id);
        tickers.push(name); country.push(""); countryAt.push(""); firstSeen.push("~");
      }
      r.gTick[k] = id;
      const first = r.dateNames[r.tickFirst[k]], last = r.dateNames[r.tickLast[k]];
      if (first < firstSeen[id]) firstSeen[id] = first;
      if (last >= countryAt[id]) { countryAt[id] = last; country[id] = r.tickCountry[k]; }
    });
  }
  const D = dates.length, L = Math.ceil(tickers.length / TEX_W);
  const index = new Int32Array(TEX_W * D * L).fill(-1);
  results.forEach((r, i) => {
    const gDate = Uint32Array.from(r.dateNames, (d) => dateIdx.get(d));
    for (let row = 0; row < r.tickCodes.length; row++) {
      const t = r.gTick[r.tickCodes[row]], d = gDate[r.dateCodes[row]];
      index[(((t / TEX_W) | 0) * D + d) * TEX_W + (t % TEX_W)] = (i << 20) | row;
    }
  });

  const meta = {
    version: CACHE_VERSION, created: new Date().toISOString(), source: files.map((f) => f.name).join(", "), rows: totalRows,
    tickers, country, first_seen: firstSeen, dates, date_type: dates.map((d) => dateTypeOf.get(d)),
    features, targets, columns, tasks: tasks.map((t, i) => [plan.chunk[i], plan.base[i], t.rows]), chunkBytes: plan.bytes,
  };
  if (cache) {
    setStatus("saving to browser cache…");
    try { await cache.finish(meta, index); } catch (e) { console.warn("cache save failed", e); await clearCache(); }
  }
  return { meta, index, chunkBufs, seconds: (performance.now() - t0) / 1000 };
}

// ---------- source 2: the browser cache ----------
async function loadCache(meta) {
  if (meta.chunkBytes.some((b) => b > bindLimit) || meta.chunkBytes.length > maxChunks) {
    throw new Error("The cached cube was built for a different GPU. Choose the parquet files again.");
  }
  const t0 = performance.now();
  const chunkBufs = await createDevice(meta.chunkBytes);
  const dir = await cacheDir(false);
  const total = meta.chunkBytes.reduce((a, b) => a + b, 0);
  let done = 0;
  for (let k = 0; k < meta.chunkBytes.length; k++) {
    const file = await (await dir.getFileHandle(`chunk_${k}.bin`)).getFile();
    if (file.size !== meta.chunkBytes[k]) throw new Error("The browser cache is incomplete. Choose the parquet files again.");
    for (let pos = 0; pos < file.size; pos += STAGE) {
      const part = await file.slice(pos, Math.min(file.size, pos + STAGE)).arrayBuffer();
      device.queue.writeBuffer(chunkBufs[k], pos, part);
      done += part.byteLength;
      setProgress(done / total);
      setStatus(`loading from browser cache · ${gib(done)} / ${gib(total)}`);
      await device.queue.onSubmittedWorkDone();
    }
  }
  const index = new Int32Array(await (await (await dir.getFileHandle("index.bin")).getFile()).arrayBuffer());
  return { meta, index, chunkBufs, seconds: (performance.now() - t0) / 1000 };
}

// ---------- source 3: the built-in demo, one slice per axis (see scripts/build_demo.py) ----------
// Unzipped here unless the server already did it (Content-Encoding), so check for the gzip magic bytes.
async function gunzip(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const blob = await r.blob();
  const head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
  if (head[0] !== 0x1f || head[1] !== 0x8b) return blob.arrayBuffer();
  return new Response(blob.stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}
async function loadDemo() {
  const t0 = performance.now();
  setStatus("loading demo…");
  const [meta, rowsBuf, values] = await Promise.all([
    fetch("demo/meta.json").then((r) => r.json()), gunzip("demo/rows.bin.gz"), gunzip("demo/values.bin.gz"),
  ]);
  const { demo } = meta, NC = meta.columns.length, D = meta.dates.length, L = Math.ceil(meta.tickers.length / TEX_W);
  // The file holds bytes: A = every column on the demo date, B = every column of the demo ticker (both
  // column-major), then C = the demo column alone for every other row. Each is packed as in layout.js;
  // C goes in tasks of at most MAX_TASK_ROWS rows whose 4th field names the one column they hold.
  const src = new Uint8Array(values), layout = columnLayout(meta.columns.map((c) => KIND[c.kind]));
  const parts = [{ rows: demo.rowsA, from: 0 }, { rows: demo.rowsB, from: demo.rowsA * NC }];
  const rowsC = meta.rows - demo.rowsA - demo.rowsB, fromC = (demo.rowsA + demo.rowsB) * NC;
  for (let s = 0; s < rowsC; s += MAX_TASK_ROWS) parts.push({ rows: Math.min(MAX_TASK_ROWS, rowsC - s), from: fromC + s, only: demo.col + 1 });
  const wideC = layout.wide[demo.col];
  const sizes = parts.map((p) => taskBytes(p.only ? (wideC ? 2 : 1) : layout.perRow, p.rows));
  const bytes = new Uint8Array(sizes.reduce((a, b) => a + b, 0));
  const tasks = [];
  for (let i = 0, base = 0; i < parts.length; base += sizes[i++]) {
    const { rows, from, only } = parts[i];
    if (only) packColumn(bytes, base, 0, wideC, rows, src.subarray(from, from + rows));
    else for (let c = 0; c < NC; c++) packColumn(bytes, base, layout.off[c], layout.wide[c], rows, src.subarray(from + c * rows));
    tasks.push([0, base, rows, only ?? 0]);
  }
  const chunkBufs = await createDevice([bytes.length]);
  device.queue.writeBuffer(chunkBufs[0], 0, bytes);

  // rows.bin: row count per date, then each date's ticker ids as deltas. Rows fall in A, B, C in that walk order.
  const counts = new Uint32Array(rowsBuf, 0, D), deltas = new Uint16Array(rowsBuf, D * 4);
  const index = new Int32Array(TEX_W * D * L).fill(-1);
  let k = 0, nA = 0, nB = 0, nC = 0;
  for (let d = 0; d < D; d++) {
    for (let i = 0, t = 0; i < counts[d]; i++, k++) {
      t = i === 0 ? deltas[k] : t + deltas[k];
      index[(((t / TEX_W) | 0) * D + d) * TEX_W + (t % TEX_W)] =
        d === demo.date ? nA++
          : t === demo.tick ? (1 << 20) | nB++
          : ((2 + Math.floor(nC / MAX_TASK_ROWS)) << 20) | (nC++ % MAX_TASK_ROWS);
    }
  }
  Object.assign(meta, { tasks, chunkBytes: [bytes.length] });
  return { meta, index, chunkBufs, seconds: (performance.now() - t0) / 1000 };
}

// ---------- data popup: the user's files or the browser cache ----------
const popup = $("start");
let loading = false;
const openPopup = () => { refreshCacheInfo(); popup.hidden = false; };
const closePopup = () => { if (!loading) popup.hidden = true; };
$("loadData").onclick = openPopup;
$("closePopup").onclick = closePopup;
popup.onclick = (e) => { if (e.target === popup) closePopup(); };
window.addEventListener("keydown", (e) => { if (e.key === "Escape" && !popup.hidden) closePopup(); });

// Errors before a new GPU device exists leave the current view running and show in the popup.
async function load(source) {
  if (loading) return;
  loading = true;
  popup.classList.add("busy");
  $("popErr").textContent = "";
  const before = device;
  try {
    const cube = await source();
    loading = false;
    closePopup();
    startViewer(cube);
  } catch (e) {
    if (device !== before) fail(`${e.message}<br><br><button onclick="location.reload()">Start again</button>`);
    $("popErr").textContent = e.message;
  } finally {
    loading = false;
    popup.classList.remove("busy");
  }
}
const pick = (files) => {
  files = [...files].filter((f) => f.name.endsWith(".parquet"));
  if (files.length) load(() => ingestParquet(files));
};
$("files").onchange = (e) => { pick(e.target.files); e.target.value = ""; };
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files); };
async function refreshCacheInfo() {
  const meta = await readCacheMeta();
  if (!meta) { $("cacheInfo").innerHTML = ""; return; }
  const size = meta.chunkBytes.reduce((a, b) => a + b, 0);
  $("cacheInfo").innerHTML =
    `<div class="cache"><div><b>Saved in this browser</b><br>${meta.source}<br>${fmt(meta.rows)} rows · ${meta.dates[0]} → ${meta.dates.at(-1)} · ${gib(size)}</div>` +
    `<div class="btns"><button id="openCache" class="primary">Open</button><button id="dropCache">Delete</button></div></div>`;
  $("openCache").onclick = () => load(() => loadCache(meta));
  $("dropCache").onclick = async () => { await clearCache(); $("cacheInfo").innerHTML = ""; };
}

// ---------- side panel: open beside the view on wide screens, a drawer behind the cog on narrow ones ----------
// While it is closed the hover readout floats over the view.
const hoverBox = $("hoverBox"), hoverSlot = $("hoverSlot");
function setPanel(open) {
  document.body.classList.toggle("collapsed", !open);
  $("cog").setAttribute("aria-expanded", String(open));
  if (open) hoverSlot.after(hoverBox); else $("stage").append(hoverBox);
}
$("cog").onclick = () => setPanel(document.body.classList.contains("collapsed"));
setPanel(!matchMedia("(max-width: 720px)").matches);

// ---------- the viewer, rebuilt for each dataset ----------
function startViewer(cube) {
  const life = new AbortController(), { signal } = life;
  stopViewer = () => life.abort();
  const meta = cube.meta, demo = meta.demo;
  const T = meta.tickers.length, D = meta.dates.length, NF = meta.features.length, NT = meta.targets.length;
  const NC = NF + NT, L = Math.ceil(T / TEX_W), index = cube.index, chunkBufs = cube.chunkBufs, K = chunkBufs.length;
  const columns = meta.columns.map((c) => c.name);
  const tasks = new Uint32Array(MAX_TASKS * 4);  // per task: chunk, byte base, rows, column + 1 if it holds only that column
  meta.tasks.forEach(([chunk, base, rows, only = 0], i) => tasks.set([chunk, base, rows, only], i * 4));
  // Per column: its nibble offset within a row of a task, << 1, | 1 if it is a whole-byte (continuous) column.
  const layout = columnLayout(meta.columns.map((c) => KIND[c.kind]));
  const colInfo = new Uint32Array(Math.ceil(NC / 4) * 4);
  for (let c = 0; c < NC; c++) colInfo[c] = (layout.off[c] << 1) | layout.wide[c];

  // Feature families are contiguous runs (intelligence_1.., charisma_.., …) followed by the named signals features.
  const featureGroups = [];
  meta.features.forEach((f, i) => {
    const name = f.match(/^feature_([a-z]+)_\d+$/)?.[1] ?? "signals";
    if (featureGroups.at(-1)?.name === name) featureGroups.at(-1).end = i + 1;
    else featureGroups.push({ name, start: i, end: i + 1 });
  });

  // ---------- row index texture: (ticker, date) -> task << 20 | row within task, -1 = no row ----------
  const indexAt = (t, d) => index[(((t / TEX_W) | 0) * D + d) * TEX_W + (t % TEX_W)];
  const idxTex = device.createTexture({ size: [TEX_W, D, L], format: "r32sint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: idxTex }, index, { bytesPerRow: TEX_W * 4, rowsPerImage: D }, [TEX_W, D, L]);
  const taskBuf = device.createBuffer({ size: tasks.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const colBuf = device.createBuffer({ size: colInfo.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(colBuf, 0, colInfo);
  device.queue.writeBuffer(taskBuf, 0, tasks);

  // ---------- ticker orderings (position on X -> ticker id) ----------
  const permTex = device.createTexture({ size: [TEX_W, L], format: "r32uint", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  let perm = new Uint32Array(T), inv = new Uint32Array(T), tickerGroups = [];  // country runs, when ordered by country
  function setOrder(kind) {
    const ids = Array.from({ length: T }, (_, i) => i);
    const fs = meta.first_seen, co = meta.country, nm = meta.tickers;
    const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
    if (kind === "alpha") ids.sort((a, b) => cmp(nm[a], nm[b]));
    if (kind === "first") ids.sort((a, b) => cmp(fs[a], fs[b]) || cmp(nm[a], nm[b]));
    if (kind === "country") ids.sort((a, b) => cmp(co[a], co[b]) || cmp(fs[a], fs[b]) || cmp(nm[a], nm[b]));
    const texData = new Uint32Array(TEX_W * L);
    tickerGroups = [];
    ids.forEach((id, pos) => {
      perm[pos] = texData[pos] = id;
      inv[id] = pos;
      if (kind !== "country") return;
      if (tickerGroups.at(-1)?.name === co[id]) tickerGroups.at(-1).end = pos + 1;
      else tickerGroups.push({ name: co[id], start: pos, end: pos + 1 });
    });
    device.queue.writeTexture({ texture: permTex }, texData, { bytesPerRow: TEX_W * 4 }, [TEX_W, L]);
  }

  // ---------- colours ----------
  const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
  const colors = new Float32Array(260 * 4);
  function fillRamp() {
    const ramp = PALETTES[palette];
    for (let v = 0; v <= 254; v++) {
      let i = 0;
      while (i < 3 && v > BIN_BYTES[i + 1]) i++;
      const f = (v - BIN_BYTES[i]) / (BIN_BYTES[i + 1] - BIN_BYTES[i]);
      const a = hex(ramp[i]), b = hex(ramp[i + 1]);
      colors.set([0, 1, 2].map((k) => a[k] + (b[k] - a[k]) * f).concat(1), v * 4);
    }
  }
  fillRamp();
  [[255, NULL_FG], [256, EMPTY], [257, NULL_BG]].forEach(([i, h]) => colors.set([...hex(h), 1], i * 4));

  // ---------- shaders ----------
  const UNIFORM_WGSL = /* wgsl */`
  struct U {
    viewProj: mat4x4f,
    box: vec4f,     // WX, WF, WT, WZ
    slice: vec4f,   // ticker pos centre, signed Y cell centre (targets < 0), date centre
    dims: vec4u,    // T, NF, NT, D
    misc: vec4u,    // NC, flags (2/4/8 date/feature/ticker plane visible), max samples, unused
    hover: vec4i,   // ticker pos, column, date, active
    mask: vec4u,    // hidden by the legend (bits 0-4 bins, 5 null, 6 no row)
    colors: array<vec4f, 260>,  // 0..254 ramp, 255 null, 256 empty, 257 null background
  };
  @group(0) @binding(0) var<uniform> u: U;
  `;
  const chunkDecls = chunkBufs.map((_, k) => `@group(0) @binding(${4 + k}) var<storage, read> c${k}: array<u32>;`).join("\n");
  const chunkSwitch = chunkBufs.map((_, k) => `if (c == ${k}u) { return c${k}[w]; }`).join(" else ") + " return 0u;";

  const planeWGSL = UNIFORM_WGSL + /* wgsl */`
  @group(0) @binding(1) var idxTex: texture_2d_array<i32>;
  @group(0) @binding(2) var permTex: texture_2d<u32>;
  @group(0) @binding(3) var<uniform> tasks: array<vec4u, ${MAX_TASKS}>;  // chunk, byte base, rows, only column + 1
  @group(0) @binding(${4 + chunkBufs.length}) var<storage, read> colInfo: array<u32>;  // nibble offset << 1 | whole byte
  ${chunkDecls}

  fn word(c: u32, w: u32) -> u32 { ${chunkSwitch} }

  struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) world: vec3f,
    @location(1) @interpolate(flat) plane: u32,
  };

  fn sliceWorld() -> vec3f {
    let y = select(u.slice.y / f32(u.dims.z) * u.box.z, u.slice.y / f32(u.dims.y) * u.box.y, u.slice.y >= 0.0);
    return vec3f(u.slice.x / f32(u.dims.x) * u.box.x, y, (1.0 - u.slice.z / f32(u.dims.w)) * u.box.w);
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

  // world -> continuous cell coords (ticker pos, signed Y cell, date); dates run from z = WZ (oldest) to z = 0
  fn toCell(w: vec3f) -> vec3f {
    let y = select(w.y / u.box.z * f32(u.dims.z), w.y / u.box.y * f32(u.dims.y), w.y >= 0.0);
    return vec3f(w.x / u.box.x * f32(u.dims.x), y, (1.0 - w.z / u.box.w) * f32(u.dims.w));
  }

  fn cellIds(q: vec3f) -> vec3u {
    let tp = u32(clamp(floor(q.x), 0.0, f32(u.dims.x - 1u)));
    var col: u32;
    if (q.y >= 0.0) { col = u32(min(floor(q.y), f32(u.dims.y - 1u))); }
    else { col = u.dims.y + u32(min(floor(-q.y), f32(u.dims.z - 1u))); }
    let d = u32(clamp(floor(q.z), 0.0, f32(u.dims.w - 1u)));
    return vec3u(tp, col, d);
  }

  // 0..255 = byte on the colour ramp, 256 = no row (or column) for this cell. Cells are packed as in layout.js.
  fn cellValue(id: vec3u) -> u32 {
    let t = textureLoad(permTex, vec2i(i32(id.x % 8192u), i32(id.x / 8192u)), 0).r;
    let packed = textureLoad(idxTex, vec2i(i32(t % 8192u), i32(id.z)), i32(t / 8192u), 0).r;
    if (packed < 0) { return 256u; }
    let task = tasks[u32(packed) >> 20u];
    let info = colInfo[id.y];
    let wide = (info & 1u) != 0u;
    var off = info >> 1u;
    if (task.w != 0u) {
      if (id.y + 1u != task.w) { return 256u; }
      off = 0u;
    }
    let a = task.y * 2u + off * task.z + (u32(packed) & 0xFFFFFu) * select(1u, 2u, wide);  // nibble address
    let bits = word(task.x, a >> 3u) >> ((a & 7u) * 4u);
    if (wide) { return bits & 0xFFu; }
    let n = bits & 0xFu;
    return select((n * 127u + 1u) >> 1u, 255u, n == 15u);  // level -> 0, 64, 127, 191, 254
  }

  fn colorOf(v: u32, q: vec3f) -> vec3f {
    if (v == 255u) {  // null value: diagonal hatch
      return select(u.colors[257].rgb, u.colors[255].rgb, fract((q.x + q.y + q.z) * 2.0) < 0.35);
    }
    return u.colors[v].rgb;
  }

  // Hidden by the legend toggles? Continuous values count as the bin colour they are nearest to.
  fn hidden(v: u32) -> bool {
    var bit = u32(round(f32(v) / 63.5));
    if (v == 255u) { bit = 5u; }
    if (v == 256u) { bit = 6u; }
    return (u.mask.x & (1u << bit)) != 0u;
  }

  @fragment fn fs(i: VOut) -> @location(0) vec4f {
    var q = toCell(i.world);
    if (i.plane == 0u) { q.z = u.slice.z; } else if (i.plane == 1u) { q.y = u.slice.y; } else { q.x = u.slice.x; }
    let dx = dpdx(q);
    let dy = dpdy(q);
    // Average n x n samples over the pixel footprint when a pixel covers many cells.
    let n = u32(clamp(ceil(max(length(dx), length(dy))), 1.0, f32(u.misc.z)));
    var acc = vec3f(0.0);
    var shown = 0.0;
    for (var a = 0u; a < n; a++) {
      for (var b = 0u; b < n; b++) {
        let o = (vec2f(f32(a), f32(b)) + 0.5) / f32(n) - 0.5;
        let qs = q + dx * o.x + dy * o.y;
        let v = cellValue(cellIds(qs));
        if (!hidden(v)) {
          acc += colorOf(v, qs);
          shown += 1.0;
        }
      }
    }
    if (shown < f32(n * n) * 0.5) { discard; }
    var col = acc / shown;
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

  const UBYTES = 160 + 260 * 16;
  const ubuf = device.createBuffer({ size: UBYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const uData = new ArrayBuffer(UBYTES), uF = new Float32Array(uData), uU = new Uint32Array(uData), uI = new Int32Array(uData);
  uF.set(colors, 160 / 4);

  const planeBG = device.createBindGroup({
    layout: planePipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ubuf } },
      { binding: 1, resource: idxTex.createView({ dimension: "2d-array" }) },
      { binding: 2, resource: permTex.createView() },
      { binding: 3, resource: { buffer: taskBuf } },
      ...chunkBufs.map((buffer, k) => ({ binding: 4 + k, resource: { buffer } })),
      { binding: 4 + chunkBufs.length, resource: { buffer: colBuf } },
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
    quality: 5,
    hiddenMask: 0,                    // legend toggles: bits 0-4 bins, 5 null, 6 no row
    hover: null,                      // {tp, col, d}
    pinned: null,                     // clicked cell; freezes the readout while set
    interacting: false,
  };
  // The demo starts higher, looking down onto its feature slice (the named factors sit near the top of Y).
  // On tall screens (phones) the start distance grows until the whole box fits the narrower horizontal view.
  function homeCam() {
    const fov = (40 * Math.PI) / 180, aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    const radius = Math.hypot(WX, WF + WT, WZ) / 2, halfH = Math.atan(Math.tan(fov / 2) * aspect);
    const dist = Math.max(6.4, (radius / Math.sin(halfH)) * 1.1);
    return { target: [WX / 2, (WF - WT) / 2, WZ / 2], dist, yaw: 0.62, pitch: demo ? 0.85 : 0.38, fov };
  }
  const cam = homeCam();

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
  const toCell = (w) => [w[0] / WX * T, w[1] >= 0 ? w[1] / WF * NF : w[1] / WT * NT, (1 - w[2] / WZ) * D];
  const dateZ = (d) => (1 - d / D) * WZ;
  const cellIds = (q) => {
    const tp = Math.min(Math.max(Math.floor(q[0]), 0), T - 1);
    const col = q[1] >= 0 ? Math.min(Math.floor(q[1]), NF - 1) : NF + Math.min(Math.floor(-q[1]), NT - 1);
    const d = Math.min(Math.max(Math.floor(q[2]), 0), D - 1);
    return { tp, col, d };
  };
  const sliceWorld = () => {
    const s = sliceCell();
    return [s[0] / T * WX, s[1] >= 0 ? s[1] / NF * WF : s[1] / NT * WT, dateZ(s[2])];
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
    if (!frameQueued && !signal.aborted) { frameQueued = true; requestAnimationFrame(render); }
  }
  function render() {
    frameQueued = false;
    if (signal.aborted) return;
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
    const flags = (state.show[0] ? 2 : 0) | (state.show[1] ? 4 : 0) | (state.show[2] ? 8 : 0);
    uU.set([NC, flags, state.interacting ? Math.min(2, state.quality) : state.quality, 0], 28);
    const hv = state.hover;
    uI.set(hv ? [hv.tp, hv.col, hv.d, 1] : [0, 0, 0, 0], 32);
    uU.set([state.hiddenMask, 0, 0, 0], 36);
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
  const resize = new ResizeObserver(requestRender);
  resize.observe(canvas);
  signal.addEventListener("abort", () => resize.disconnect());

  // ---------- labels ----------
  const labelsEl = $("labels");
  const labelPool = [];
  labelsEl.replaceChildren();
  function drawLabels(M) {
    const items = [
      [[WX / 2, -WT, -0.12], "tickers (X)"],
      [[-0.1, WF / 2, 0], "features ↑"],
      [[-0.1, -WT / 2, 0], "targets ↓"],
      [[-0.32, -WT, WZ / 2], "dates (Z)"],
    ];
    let lastYear = "";
    meta.dates.forEach((dt, d) => {
      const y = dt.slice(0, 4);
      if (y !== lastYear && +y % 2 === 1) items.push([[-0.04, -WT - 0.04, dateZ(d)], y]);
      lastYear = y;
    });
    // Group labels along the front edges, only where the group is wide enough on screen to read.
    const span = (a, b) => {
      const pa = project(M, a), pb = project(M, b);
      return pa && pb ? Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) : 0;
    };
    for (const g of tickerGroups) {
      const x0 = (g.start / T) * WX, x1 = (g.end / T) * WX;
      if (span([x0, -WT, WZ], [x1, -WT, WZ]) >= 18) items.push([[(x0 + x1) / 2, -WT - 0.03, WZ + 0.03], g.name, "grp"]);
    }
    for (const g of featureGroups) {
      const y0 = (g.start / NF) * WF, y1 = (g.end / NF) * WF;
      if (span([0, y0, WZ], [0, y1, WZ]) >= 12) items.push([[-0.03, (y0 + y1) / 2, WZ + 0.03], g.name, "grp right"]);
    }
    const s = sliceWorld();
    if (state.show[0]) items.push([[WX + 0.06, WF, s[2]], meta.dates[state.date], "slice"]);
    if (state.show[1]) items.push([[WX + 0.06, s[1], WZ], columns[state.col].replace(/^feature_|^target_/, ""), "slice"]);
    if (state.show[2]) items.push([[s[0], WF + 0.06, WZ], meta.tickers[state.tick], "slice"]);
    items.forEach(([p, text, cls], i) => {
      let el = labelPool[i];
      if (!el) { el = labelPool[i] = document.createElement("div"); labelsEl.appendChild(el); }
      const sp = project(M, p);
      el.style.display = sp ? "" : "none";
      if (!sp) return;
      el.className = cls ? `lbl ${cls}` : "lbl";
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
    if (c.kind === "cont") return `≈ ${(c.min + (v / 254) * (c.max - c.min)).toPrecision(4)}`;
    const k = BIN_BYTES.indexOf(v);
    return col < NF ? `bin ${k}` : `${k / 4}`;
  }
  // Bloomberg-style "7203 JP" -> Yahoo-style "7203.T", the symbol FMP's keyless logo CDN is keyed by.
  const YAHOO_SUFFIX = { US: "", JP: ".T", HK: ".HK", KR: ".KS", TW: ".TW", GB: ".L", AU: ".AX", CA: ".TO",
    DE: ".DE", BR: ".SA", SE: ".ST", FR: ".PA", IT: ".MI", SG: ".SI", IN: ".NS", CH: ".SW",
    NO: ".OL", ZA: ".JO", NL: ".AS", ES: ".MC" };
  function logoUrl(ticker) {
    let [code, cc] = ticker.split(" ");
    if (cc === "HK") code = code.padStart(4, "0");
    const suf = cc === "CN" ? (/^[69]/.test(code) ? ".SS" : ".SZ") : (YAHOO_SUFFIX[cc] ?? "");
    return `https://financialmodelingprep.com/image-stock/${encodeURIComponent(code.replace("/", "-") + suf)}.png`;
  }
  function showHover(h, text) {
    if (state.pinned && h !== state.pinned) return;  // a stale hover read finishing after a pin
    const t = perm[h.tp], col = columns[h.col];
    $("hover").classList.toggle("pinned", !!state.pinned);
    $("hover").innerHTML =
      `<div class="rows"><div><span class="k">ticker</span>${meta.tickers[t]} · ${meta.country[t]}</div>` +
      `<div><span class="k">date</span>${meta.dates[h.d]} · ${meta.date_type[h.d]}</div>` +
      `<div><span class="k">${h.col < NF ? "feature" : "target"}</span>${col.replace(/^feature_|^target_/, "")}</div>` +
      `<div><span class="k">value</span><span class="val">${text}</span></div></div>` +
      (state.pinned ? `<img class="logo" src="${logoUrl(meta.tickers[t])}" alt="" onerror="this.remove()">` : "");
  }
  function setHover(h) {
    const key = h && `${h.tp},${h.col},${h.d}`;
    if (key === state.hoverKey) return;
    state.hoverKey = key;
    state.hover = h;
    if (h) readHover(h);
    requestRender();
  }
  // Clicking a cell pins it: the readout freezes (so it can be copied) and the company logo loads.
  // Clicking the pinned cell or empty space, or pressing Esc, releases it; clicking another cell moves the pin.
  function togglePin(h) {
    const was = state.pinned, same = h && was && `${h.tp},${h.col},${h.d}` === state.hoverKey;
    state.pinned = same ? null : h;
    state.hoverKey = undefined;  // force setHover to refresh, even to null
    if (!state.pinned && was) {
      $("hover").classList.remove("pinned");
      $("hover").innerHTML = "";
    }
    setHover(state.pinned ?? h);
  }
  // One cell's ramp byte, read back from the GPU: 0..255, or a string when there is nothing stored.
  // Reads queue up, since readBuf is mapped while one is in flight.
  let readQueue = Promise.resolve();
  function readCell(t, d, col) {
    const next = readQueue.then(() => readCellNow(t, d, col));
    readQueue = next.catch(() => {});
    return next;
  }
  async function readCellNow(t, d, col) {
    const packed = indexAt(t, d);
    if (packed < 0) return "no row (not in universe this week)";
    const ti = (packed >>> 20) * 4, only = tasks[ti + 3];
    if (only && only !== col + 1) return "not in the demo";
    const wide = layout.wide[col], off = only ? 0 : layout.off[col];
    const a = tasks[ti + 1] * 2 + off * tasks[ti + 2] + (packed & 0xfffff) * (wide ? 2 : 1);  // nibble address
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(chunkBufs[tasks[ti]], Math.floor(a / 8) * 4, readBuf, 0, 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const bits = new Uint32Array(readBuf.getMappedRange())[0] >>> ((a % 8) * 4);
    readBuf.unmap();
    return wide ? bits & 0xff : byteOfNibble(bits & 0xf);
  }
  async function readHover(h) {
    if (reading) { pendingRead = h; return; }
    reading = true;
    try {
      const v = await readCell(perm[h.tp], h.d, h.col);
      showHover(h, typeof v === "string" ? v : describe(v, h.col));
    } catch (e) {
      if (!signal.aborted) throw e;  // reads in flight when the dataset is replaced
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
  // The demo holds one slice per axis, so its slices stay put.
  const setDate = (d) => { if (!demo) { state.date = clamp(d, 0, D - 1); syncUI(); } };
  const setCol = (c) => { if (!demo) { state.col = clamp(c, 0, NC - 1); syncUI(); } };
  const setTickPos = (p) => { if (!demo) { state.tick = perm[clamp(p, 0, T - 1)]; syncUI(); } };

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
  $("quality").oninput = (e) => { state.quality = +e.target.value; syncUI(); };

  // Legend: every colour is a toggle that hides those cells (bits 0-4 bins, 5 null, 6 no row).
  const nullSwatch = `repeating-linear-gradient(45deg,${NULL_FG} 0 2px,${NULL_BG} 2px 6px)`;
  $("legend").innerHTML =
    `<div class="bins">${BIN_BYTES.map((_, i) => `<button class="tog" data-bit="${i}" title="show/hide ${i}"><span class="sw"></span><span class="lab">${i}</span></button>`).join("")}</div>` +
    `<div class="grad"></div>` +
    `<div class="extra"><button class="tog" data-bit="5"><span class="sw" style="background:${nullSwatch}"></span> <span class="lab">null</span></button>` +
    `<button class="tog" data-bit="6"><span class="sw" style="background:${EMPTY};outline:1px solid #383835"></span> <span class="lab">no row</span></button></div>`;
  $("legend").querySelectorAll(".tog").forEach((b) => {
    b.onclick = () => {
      state.hiddenMask ^= 1 << +b.dataset.bit;
      b.classList.toggle("off", (state.hiddenMask >> +b.dataset.bit) & 1);
      b.setAttribute("aria-pressed", String(!b.classList.contains("off")));
      requestRender();
    };
  });
  function applyPalette() {
    const ramp = PALETTES[palette];
    $("legend").querySelectorAll(".bins .sw").forEach((s, i) => { s.style.background = ramp[i]; });
    $("legend").querySelector(".grad").style.background = `linear-gradient(90deg,${ramp.join(",")})`;
    fillRamp();
    uF.set(colors, 160 / 4);
    requestRender();
  }
  $("palette").value = palette;
  $("palette").onchange = (e) => {
    palette = e.target.value;
    try { localStorage.setItem("palette", palette); } catch {}
    applyPalette();
  };
  applyPalette();

  // Hotkeys: [ ] date, ( ) ticker, { } feature/target; hold ctrl or alt to step by 10.
  function resetView() { Object.assign(cam, homeCam()); requestRender(); }
  $("home").onclick = resetView;
  window.addEventListener("keydown", (e) => {
    if (!popup.hidden || (e.target.tagName === "INPUT" && e.target.type === "text")) return;
    const step = e.ctrlKey || e.altKey ? 10 : 1;
    const keys = {
      "[": () => setDate(state.date - step), "]": () => setDate(state.date + step),
      "(": () => setTickPos(inv[state.tick] - step), ")": () => setTickPos(inv[state.tick] + step),
      "{": () => setCol(sliderToCol(colToSlider(state.col) - step)), "}": () => setCol(sliderToCol(colToSlider(state.col) + step)),
      r: resetView,
      Escape: () => state.pinned && togglePin(null),
    };
    if (keys[e.key]) { keys[e.key](); e.preventDefault(); }
  }, { signal });

  // ---------- mouse and touch ----------
  // One pointer: drag orbits (right button or shift pans, ctrl moves a slice) and a click or tap pins a cell.
  // Two fingers: pinch zooms towards their midpoint and moving them together pans.
  let drag = null;
  const touches = new Map();  // pointerId -> [x, y]
  function zoomAt(x, y, f) {
    const hit = pick(x, y);
    const anchor = hit ? hit.p : cam.target;
    cam.target = add(anchor, scale(sub(cam.target, anchor), f));
    cam.dist = clamp(cam.dist * f, 1e-5, 50);
  }
  function panBy(dx, dy) {
    const { right, up } = camBasis();
    const k = (2 * cam.dist * Math.tan(cam.fov / 2)) / canvas.clientHeight;
    cam.target = add(cam.target, add(scale(right, -dx * k), scale(up, dy * k)));
  }
  const pinchOf = () => {
    const [a, b] = [...touches.values()];
    return { mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], span: Math.max(1, Math.hypot(a[0] - b[0], a[1] - b[1])) };
  };
  const on = (type, fn, opts) => canvas.addEventListener(type, fn, { signal, ...opts });
  on("contextmenu", (e) => e.preventDefault());
  on("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    if (e.pointerType === "touch") touches.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (touches.size >= 2) {
      if (touches.size === 2) drag = { mode: "pinch", ...pinchOf() };
      return;
    }
    const mode = e.ctrlKey && !demo ? "slice" : e.button === 2 || e.shiftKey ? "pan" : "orbit";
    drag = { mode, x: e.offsetX, y: e.offsetY, x0: e.offsetX, y0: e.offsetY, click: mode === "orbit" && e.button === 0 };
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
  on("pointermove", (e) => {
    if (touches.has(e.pointerId)) touches.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (drag?.mode === "pinch") {
      if (touches.size < 2) return;
      const { mid, span } = pinchOf();
      state.interacting = true;
      zoomAt(mid[0], mid[1], drag.span / span);
      panBy(mid[0] - drag.mid[0], mid[1] - drag.mid[1]);
      Object.assign(drag, { mid, span });
      requestRender();
      return;
    }
    if (!drag) {
      if (!state.pinned && e.pointerType !== "touch") setHover(pick(e.offsetX, e.offsetY));
      return;
    }
    if (Math.hypot(e.offsetX - drag.x0, e.offsetY - drag.y0) > (e.pointerType === "touch" ? 10 : 4)) drag.click = false;
    const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
    if (drag.mode === "orbit") {
      cam.yaw -= dx * 0.005;
      cam.pitch = clamp(cam.pitch + dy * 0.005, -1.55, 1.55);
      drag.x = e.offsetX; drag.y = e.offsetY;
    } else if (drag.mode === "pan") {
      panBy(dx, dy);
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
  const release = (e) => {
    touches.delete(e.pointerId);
    if (drag?.mode !== "pinch") return false;
    // Lifting one finger of a pinch carries on as a one-finger orbit from the other.
    const rest = [...touches.values()][0];
    if (rest) drag = { mode: "orbit", x: rest[0], y: rest[1], x0: rest[0], y0: rest[1], click: false };
    else endDrag();
    return true;
  };
  on("pointerup", (e) => {
    if (release(e)) return;
    if (drag?.click) togglePin(pick(e.offsetX, e.offsetY));
    endDrag();
  });
  on("pointercancel", (e) => { release(e); endDrag(); });
  on("wheel", (e) => {
    e.preventDefault();
    zoomAt(e.offsetX, e.offsetY, Math.exp(e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)));
    requestRender();
  }, { passive: false });

  // ---------- initial state ----------
  setOrder("first");
  state.date = demo ? demo.date : Math.floor(D / 2);
  state.col = demo ? demo.col : Math.floor(NF / 2);
  state.tick = demo ? demo.tick : tickerId.get("AAPL US") ?? perm[0];
  for (const id of ["date", "col", "colSel", "tick", "tickSearch"]) $(id).disabled = !!demo;
  $("demoNote").hidden = !demo;
  $("hover").replaceChildren();
  syncUI();
  setProgress(1);
  setStatus(demo
    ? `${gpuName} · demo: one slice per axis · ready in ${cube.seconds.toFixed(1)}s`
    : `${gpuName} · ${fmt(meta.rows)} rows × ${NC} columns · ${gib(meta.chunkBytes.reduce((a, b) => a + b, 0))} in ${K} buffers · ready in ${cube.seconds.toFixed(1)}s`);
  $("source").textContent = meta.source;
}

// ---------- start with the demo ----------
try {
  startViewer(await loadDemo());
} catch (e) {
  console.error(e);
  $("popErr").textContent = `The demo didn't load (${e.message}). Load your own data files instead.`;
  openPopup();
}
