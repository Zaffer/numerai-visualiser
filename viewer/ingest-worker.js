// Decodes one range of rows (within one row group) from a Numerai parquet file and encodes it for the GPU.
// Output is column-major: byte [c * rows + r] = column c of row r, on the shared 0..254 ramp (255 = null).
//
// Fast path: Numerai files are flat, Snappy-compressed, dictionary-encoded v1 data pages, so pages are
// decoded straight into output bytes through a dictionary -> byte lookup table, with no JS arrays.
// Anything else (other codecs, page versions, encodings) falls back to hyparquet's generic reader.
import { parquetMetadataAsync, parquetRead } from "./vendor/hyparquet/index.js";
import { deserializeTCompactProtocol } from "./vendor/hyparquet/thrift.js";
import { snappyUncompress } from "./vendor/hyparquet/snappy.js";
import { readRleBitPackedHybrid } from "./vendor/hyparquet/encoding.js";

const BIN = [0, 64, 127, 191, 254];
const KIND_BIN = 0, KIND_TARGET = 1;          // anything else is continuous
const STRING_COLUMNS = ["numerai_ticker", "date", "feature_country", "data_type"];
const PAGE_DATA = 0, PAGE_DICTIONARY = 2;
const ENC_PLAIN = 0, ENC_PLAIN_DICTIONARY = 2, ENC_RLE_DICTIONARY = 8;
const COLUMN_BATCH = 64;                      // generic fallback: columns decoded per parquetRead call
const metaCache = new Map();                  // parsed footers, per file
const utf8 = new TextDecoder();

class Unsupported extends Error {}

const fileBuffer = (file) => ({ byteLength: file.size, slice: (s, e) => file.slice(s, e).arrayBuffer() });

// value -> byte on the shared ramp
function byteEncoder(kind, lo, hi) {
  if (kind === KIND_BIN) return (v) => (v >= 0 && v <= 4 ? BIN[v] : 255);
  if (kind === KIND_TARGET) return (v) => (v !== v ? 255 : BIN[Math.min(4, Math.max(0, Math.round(v * 4)))]);
  const k = 254 / (hi - lo);
  return (v) => (v !== v ? 255 : Math.min(254, Math.max(0, Math.round((v - lo) * k))));
}

function decompress(body, size, codec) {
  if (codec === "UNCOMPRESSED") return body;
  if (codec !== "SNAPPY") throw new Unsupported(`codec ${codec}`);
  const page = new Uint8Array(size);
  snappyUncompress(body, page);
  return page;
}

// Walk a column chunk's pages, decompressing the dictionary page and the data pages overlapping [selStart, selEnd).
function readPages(u8, view, blockStart, m, selStart, selEnd, onDictionary, onData) {
  let off = Number(m.dictionary_page_offset || m.data_page_offset) - blockStart;
  const end = off + Number(m.total_compressed_size);
  let row = 0;
  while (off < end && row < selEnd) {
    const reader = { view, offset: off };
    const h = deserializeTCompactProtocol(reader);
    const body = u8.subarray(reader.offset, reader.offset + h.field_3);
    off = reader.offset + h.field_3;
    if (h.field_1 === PAGE_DICTIONARY) {
      onDictionary(decompress(body, h.field_2, m.codec), h.field_7.field_1);
    } else if (h.field_1 === PAGE_DATA) {
      const count = h.field_5.field_1;
      if (row + count > selStart) onData(decompress(body, h.field_2, m.codec), count, h.field_5.field_2, row);
      row += count;
    } else {
      throw new Unsupported(`page type ${h.field_1}`);
    }
  }
}

// Definition levels (if optional) and dictionary indices or a reader positioned at PLAIN values.
function pageValues(page, count, encoding, optional) {
  const reader = { view: new DataView(page.buffer, page.byteOffset, page.byteLength), offset: 0 };
  let defs = null, nonNull = count;
  if (optional) {
    defs = new Uint8Array(count);
    readRleBitPackedHybrid(reader, 1, defs);
    nonNull = 0;
    for (let i = 0; i < count; i++) nonNull += defs[i];
  }
  if (encoding === ENC_RLE_DICTIONARY || encoding === ENC_PLAIN_DICTIONARY) {
    const width = page[reader.offset++];
    const idx = new Uint32Array(nonNull);
    if (width) readRleBitPackedHybrid(reader, width, idx, page.length - reader.offset);
    return { defs, idx, plain: null };
  }
  if (encoding === ENC_PLAIN) return { defs, idx: null, plain: reader };
  throw new Unsupported(`encoding ${encoding}`);
}

// A numeric column chunk -> out[base + row - selStart].
function decodeNumeric(u8, view, blockStart, m, element, out, base, toByte, selStart, selEnd) {
  const size = { INT32: 4, FLOAT: 4, DOUBLE: 8 }[element.type];
  if (!size) throw new Unsupported(`type ${element.type}`);
  const get = element.type === "INT32" ? (dv, o) => dv.getInt32(o, true)
    : element.type === "FLOAT" ? (dv, o) => dv.getFloat32(o, true) : (dv, o) => dv.getFloat64(o, true);
  const optional = element.repetition_type === "OPTIONAL";
  let lut = null;
  readPages(u8, view, blockStart, m, selStart, selEnd,
    (dict, n) => {
      const dv = new DataView(dict.buffer, dict.byteOffset, dict.byteLength);
      lut = new Uint8Array(n);
      for (let i = 0; i < n; i++) lut[i] = toByte(get(dv, i * size));
    },
    (page, count, encoding, row0) => {
      const { defs, idx, plain } = pageValues(page, count, encoding, optional);
      if (idx && !lut) throw new Unsupported("dictionary page missing");
      const from = Math.max(selStart, row0), to = Math.min(selEnd, row0 + count);
      if (idx && (!defs || idx.length === count)) {  // no nulls: straight lookup
        for (let r = from; r < to; r++) out[base + r - selStart] = lut[idx[r - row0]];
        return;
      }
      let k = 0;
      for (let i = 0; i < count; i++) {
        const r = row0 + i;
        let b = 255;
        if (!defs || defs[i]) {
          b = idx ? lut[idx[k]] : toByte(get(plain.view, plain.offset + k * size));
          k++;
        }
        if (r >= from && r < to) out[base + r - selStart] = b;
      }
    });
}

// A string column chunk -> unique names plus a code per selected row.
function decodeStrings(u8, view, blockStart, m, element, selStart, selEnd) {
  if (element.type !== "BYTE_ARRAY") throw new Unsupported(`type ${element.type}`);
  const optional = element.repetition_type === "OPTIONAL";
  const names = [], map = new Map(), codes = new Uint32Array(selEnd - selStart);
  const code = (s) => {
    let c = map.get(s);
    if (c === undefined) { c = names.length; map.set(s, c); names.push(s); }
    return c;
  };
  const readString = (dv, bytes, reader) => {
    const len = dv.getUint32(reader.offset, true);
    const s = utf8.decode(bytes.subarray(reader.offset + 4, reader.offset + 4 + len));
    reader.offset += 4 + len;
    return s;
  };
  // Dictionary entries get a code only when a selected row uses them, so names lists only present values.
  let dict = null, dictCodes = null;
  readPages(u8, view, blockStart, m, selStart, selEnd,
    (page, n) => {
      const dv = new DataView(page.buffer, page.byteOffset, page.byteLength), reader = { offset: 0 };
      dict = Array.from({ length: n }, () => readString(dv, page, reader));
      dictCodes = new Int32Array(n).fill(-1);
    },
    (page, count, encoding, row0) => {
      const { defs, idx, plain } = pageValues(page, count, encoding, optional);
      if (idx && !dict) throw new Unsupported("dictionary page missing");
      let k = 0;
      for (let i = 0; i < count; i++) {
        const r = row0 + i, selected = r >= selStart && r < selEnd;
        if (defs && !defs[i]) { if (selected) codes[r - selStart] = code(""); continue; }
        if (idx) {
          const j = idx[k++];
          if (selected) codes[r - selStart] = dictCodes[j] >= 0 ? dictCodes[j] : (dictCodes[j] = code(dict[j]));
        } else {
          const s = readString(plain.view, page, plain);
          k++;
          if (selected) codes[r - selStart] = code(s);
        }
      }
    });
  return { names, codes };
}

// Unique values plus a per-row code into them (for the generic fallback).
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
    const key = `${file.name}|${file.size}|${file.lastModified}`;
    let metadata = metaCache.get(key);
    if (!metadata) { metadata = await parquetMetadataAsync(fileBuffer(file)); metaCache.set(key, metadata); }

    // The row group holding this task, read in one go (many small concurrent File reads stall Chrome).
    let g = 0, groupStart = 0;
    while (groupStart + Number(metadata.row_groups[g].num_rows) <= rowStart) groupStart += Number(metadata.row_groups[g++].num_rows);
    const chunks = new Map(metadata.row_groups[g].columns.map((cc) => [cc.meta_data.path_in_schema[0], cc.meta_data]));
    let a = Infinity, b = 0;
    for (const m of chunks.values()) {
      const s = Number(m.dictionary_page_offset || m.data_page_offset);
      a = Math.min(a, s);
      b = Math.max(b, s + Number(m.total_compressed_size));
    }
    const block = await file.slice(a, b).arrayBuffer();
    const u8 = new Uint8Array(block), view = new DataView(block);
    const selStart = rowStart - groupStart, selEnd = rowEnd - groupStart;
    const elements = new Map(metadata.schema.map((e) => [e.name, e]));

    const out = new Uint8Array(Math.ceil((n * columns.length) / 4) * 4);
    const strings = {}, fallback = [];
    for (const name of STRING_COLUMNS) {
      if (!chunks.has(name)) continue;
      try { strings[name] = decodeStrings(u8, view, a, chunks.get(name), elements.get(name), selStart, selEnd); }
      catch (e) { if (!(e instanceof Unsupported)) throw e; fallback.push(name); }
    }
    columns.forEach((name, c) => {
      try { decodeNumeric(u8, view, a, chunks.get(name), elements.get(name), out, c * n, byteEncoder(kinds[c], lo[c], hi[c]), selStart, selEnd); }
      catch (e) { if (!(e instanceof Unsupported)) throw e; fallback.push(name); }
    });

    // Generic path for anything the fast path doesn't handle, a few columns at a time to bound JS heap use.
    // It reads whole row groups and selects rows here: hyparquet misplaces rows for reads starting mid-group.
    if (fallback.length) {
      const colIndex = new Map(columns.map((name, i) => [name, i]));
      const raw = {};
      const source = { byteLength: file.size, slice: (s, e = file.size) => (s >= a && e <= b ? block.slice(s - a, e - a) : file.slice(s, e).arrayBuffer()) };
      const groupEnd = groupStart + Number(metadata.row_groups[g].num_rows);
      for (let i = 0; i < fallback.length; i += COLUMN_BATCH) {
        await parquetRead({
          file: source, metadata, rowStart: groupStart, rowEnd: groupEnd, columns: fallback.slice(i, i + COLUMN_BATCH),
          onChunk: ({ columnName, columnData, rowStart: s }) => {
            const c = colIndex.get(columnName);
            const toByte = c === undefined ? null : byteEncoder(kinds[c], lo[c], hi[c]);
            const dst = c === undefined ? (raw[columnName] ??= new Array(n)) : null;
            for (let j = 0; j < columnData.length; j++) {
              const r = s - groupStart + j;
              if (r < selStart || r >= selEnd) continue;
              const v = columnData[j];
              if (dst) dst[r - selStart] = v;
              else out[c * n + r - selStart] = v == null ? 255 : toByte(v);
            }
          },
        });
      }
      for (const [name, values] of Object.entries(raw)) strings[name] = codeStrings(values);
    }

    // Rows are date-sorted within a file, so first/last occurrence give each ticker's first/last date here.
    const tick = strings.numerai_ticker, date = strings.date;
    const country = strings.feature_country, dtype = strings.data_type;
    const nt = tick.names.length;
    const tickFirst = new Uint32Array(nt).fill(0xffffffff), tickLast = new Uint32Array(nt);
    const tickCountry = new Array(nt).fill(""), dateTypes = new Array(date.names.length).fill("");
    for (let r = 0; r < n; r++) {
      const t = tick.codes[r], d = date.codes[r];
      if (tickFirst[t] === 0xffffffff) tickFirst[t] = d;
      tickLast[t] = d;
      if (country) tickCountry[t] = country.names[country.codes[r]];
      if (dtype && !dateTypes[d]) dateTypes[d] = dtype.names[dtype.codes[r]];
    }
    postMessage(
      { i: job.i, bytes: out.buffer, tickNames: tick.names, tickCodes: tick.codes, tickFirst, tickLast, tickCountry,
        dateNames: date.names, dateCodes: date.codes, dateTypes, fallback: fallback.length },
      [out.buffer, tick.codes.buffer, date.codes.buffer, tickFirst.buffer, tickLast.buffer],
    );
  } catch (e) {
    postMessage({ i: job.i, error: String(e?.message || e) });
  }
};
