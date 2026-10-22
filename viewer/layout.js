// How cells are packed on the GPU, shared by the decoder workers and the viewer.
// Binned columns (features 0-4, targets 0-1 in quarters) take a 4-bit nibble per cell: the level 0-4, or 15 for null.
// Continuous columns keep a byte per cell: 0-254 on the colour ramp, 255 for null.
// A task is column-major: column c starts at nibble (2 * task byte base + off[c] * rows), and a byte column's
// off is even so its bytes stay whole.

export const KIND = { bin: 0, target: 1, cont: 2 };
const NULL_NIBBLE = 15;

// ramp byte -> nibble, and back (the five bin bytes are round(level * 63.5))
export const NIBBLE_OF_BYTE = new Uint8Array(256).fill(NULL_NIBBLE);
[0, 64, 127, 191, 254].forEach((b, level) => { NIBBLE_OF_BYTE[b] = level; });
export const byteOfNibble = (n) => (n === NULL_NIBBLE ? 255 : (n * 127 + 1) >> 1);

export function columnLayout(kinds) {
  const off = new Uint32Array(kinds.length), wide = new Uint8Array(kinds.length);
  let at = 0;
  kinds.forEach((k, c) => {
    wide[c] = k === KIND.cont ? 1 : 0;
    if (wide[c] && at % 2) at++;
    off[c] = at;
    at += wide[c] ? 2 : 1;
  });
  return { off, wide, perRow: at };  // perRow = nibbles per row
}

// Bytes for a task of `rows` rows, padded to whole 32-bit words.
export const taskBytes = (perRow, rows) => Math.ceil((perRow * rows) / 8) * 4;

// Write one column (ramp bytes, one per row) into a zero-filled task at byte offset `base`.
export function packColumn(out, base, off, wide, rows, src) {
  if (wide) { out.set(src.subarray(0, rows), base + (off * rows) / 2); return; }
  const N = NIBBLE_OF_BYTE, a = base * 2 + off * rows;
  let r = 0, b = a >> 1;
  if (a & 1 && rows) out[b++] |= N[src[r++]] << 4;  // finish the byte the previous column started
  for (; r + 1 < rows; r += 2) out[b++] = N[src[r]] | (N[src[r + 1]] << 4);
  if (r < rows) out[b] = N[src[r]];
}
