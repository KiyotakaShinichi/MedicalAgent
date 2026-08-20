/**
 * Binary-to-base64 encoding for patient file uploads.
 *
 * The `/me/uploads` endpoint expects a bare base64 payload, not a data URL, so
 * the leading `data:<mime>;base64,` prefix never goes on the wire.
 */

/**
 * Chunk size for the `String.fromCharCode` spread.
 *
 * `fromCharCode(...bytes)` on a whole multi-megabyte file overflows the
 * argument-count limit and throws a RangeError, so the array is walked in
 * fixed-size windows. 0x8000 is comfortably under every engine's limit.
 */
const CHUNK_SIZE = 0x8000;

/** Encode raw bytes as base64. Exported separately so it is testable without a File. */
export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += CHUNK_SIZE) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK_SIZE));
  }
  return btoa(binary);
}

/** Read a `File`/`Blob` and return its base64 payload without a data-URL prefix. */
export async function readFileAsBase64(file: Blob): Promise<string> {
  const buffer = await file.arrayBuffer();
  return bytesToBase64(new Uint8Array(buffer));
}
