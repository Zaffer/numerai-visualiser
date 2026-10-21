// Decodes one range of rows from a Numerai parquet file and encodes it for the GPU.
// Output is column-major: byte [c * rows + r] = column c of row r, on the shared 0..254 ramp (255 = null).
import { parquetMetadataAsync, parquetRead } from "./vendor/hyparquet/index.js";

const BIN = [0, 64, 127, 191, 254];
const KIND_BIN = 0, KIND_TARGET = 1;          // anything else is continuous
const STRING_COLUMNS = ["numerai_ticker", "date", "feature_country", "data_type"];
const COLUMN_BATCH = 64;
const metaCache = new Map();                  // parsed footers, per file

const fileBuffer = (file) => ({ byteLength: file.size, slice: (s, e) => file.slice(s, e).arrayBuffer() });

// Byte range of the row groups overlapping [rowStart, rowEnd).
function groupBytes(metadata, rowStart, rowEnd) {
  let start = 0, lo = Infinity, hi = 0;
  for (const rg of metadata.row_groups) {
    const n = Number(rg.num_rows);
    if (start < rowEnd && start + n > rowStart) {
      for (const { meta_data: m } of rg.columns) {
        const a = Number(m.dictionary_page_offset || m.data_page_offset);
        lo = Math.min(lo, a);
        hi = Math.max(hi, a + Number(m.total_compressed_size));
      }
    }
    start += n;
  }
  return [lo, hi];
}

// Read the task's row groups in one go and serve hyparquet's many small reads from memory;
// hundreds of concurrent small File reads across workers stall Chrome.
async function prefetched(file, metadata, rowStart, rowEnd) {
  const [a, b] = groupBytes(metadata, rowStart, rowEnd);
  const block = await file.slice(a, b).arrayBuffer();
  return {
    byteLength: file.size,
    slice: (s, e = file.size) => (s >= a && e <= b ? block.slice(s - a, e - a) : file.slice(s, e).arrayBuffer()),
  };
}

function encode(src, out, base, kind, lo, hi) {
  const len = src.length;
  if (kind === KIND_BIN) {
    for (let i = 0; i < len; i++) {
      const v = src[i];
      out[base + i] = v == null || v < 0 || v > 4 ? 255 : BIN[v];
    }
  } else if (kind === KIND_TARGET) {
    for (let i = 0; i < len; i++) {
      const v = src[i];
      out[base + i] = v == null || v !== v ? 255 : BIN[Math.min(4, Math.max(0, Math.round(v * 4)))];
    }
  } else {
    const k = 254 / (hi - lo);
    for (let i = 0; i < len; i++) {
      const v = src[i];
      out[base + i] = v == null || v !== v ? 255 : Math.min(254, Math.max(0, Math.round((v - lo) * k)));
    }
  }
}

// Unique values plus a per-row code into them.
function codeStrings(values) {
  const map = new Map(), names = [], codes = new Uint32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i] ?? "";
    let c = map.get(v);
    if (c === undefined) { c = names.length; map.set(v, c); names.push(v); }
    codes[i] = c;
  }
  return { names, codes };
}

onmessage = async ({ data: job }) => {
  try {
    const { file, rowStart, rowEnd, columns, kinds, lo, hi } = job;
    const n = rowEnd - rowStart;
    const buf = fileBuffer(file);
    const key = `${file.name}|${file.size}|${file.lastModified}`;
    let metadata = metaCache.get(key);
    if (!metadata) { metadata = await parquetMetadataAsync(buf); metaCache.set(key, metadata); }

    const present = new Set(metadata.schema.map((e) => e.name));
    const strings = Object.fromEntries(STRING_COLUMNS.filter((c) => present.has(c)).map((c) => [c, new Array(n)]));
    const colIndex = new Map(columns.map((c, i) => [c, i]));
    const out = new Uint8Array(Math.ceil((n * columns.length) / 4) * 4);
    const onChunk = ({ columnName, columnData, rowStart: s }) => {
      const off = s - rowStart;
      if (Object.hasOwn(strings, columnName)) {
        const dst = strings[columnName];
        for (let i = 0; i < columnData.length; i++) dst[off + i] = columnData[i];
        return;
      }
      const c = colIndex.get(columnName);
      encode(columnData, out, c * n + off, kinds[c], lo[c], hi[c]);
    };
    // Decode a few columns at a time: hyparquet returns plain JS arrays, and all workers share one
    // V8 heap cage, so decoding whole row groups at once in several workers exhausts it.
    const source = await prefetched(file, metadata, rowStart, rowEnd);
    const wanted = [...Object.keys(strings), ...columns];
    for (let i = 0; i < wanted.length; i += COLUMN_BATCH) {
      await parquetRead({ file: source, metadata, rowStart, rowEnd, columns: wanted.slice(i, i + COLUMN_BATCH), onChunk });
    }

    // Rows are date-sorted within a file, so first/last occurrence give each ticker's first/last date here.
    const tick = codeStrings(strings.numerai_ticker), date = codeStrings(strings.date);
    const nt = tick.names.length;
    const tickFirst = new Uint32Array(nt).fill(0xffffffff), tickLast = new Uint32Array(nt);
    const tickCountry = new Array(nt).fill(""), dateTypes = new Array(date.names.length).fill("");
    for (let r = 0; r < n; r++) {
      const t = tick.codes[r], d = date.codes[r];
      if (tickFirst[t] === 0xffffffff) tickFirst[t] = d;
      tickLast[t] = d;
      tickCountry[t] = strings.feature_country?.[r] ?? "";
      if (!dateTypes[d]) dateTypes[d] = strings.data_type?.[r] ?? "";
    }
    postMessage(
      { i: job.i, bytes: out.buffer, tickNames: tick.names, tickCodes: tick.codes, tickFirst, tickLast, tickCountry,
        dateNames: date.names, dateCodes: date.codes, dateTypes },
      [out.buffer, tick.codes.buffer, date.codes.buffer, tickFirst.buffer, tickLast.buffer],
    );
  } catch (e) {
    postMessage({ i: job.i, error: String(e?.message || e) });
  }
};
