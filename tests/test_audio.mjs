// Ses son-isleme diferansiyel testi: audio.js vs Python referansi.
// Calistirma: node --test tests/test_audio.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const require = createRequire(import.meta.url);
const A = require(path.join(root, "audio.js"));

// post.py katsayilariyla birebir (16 ondalik karsilastirma)
test("TONE katsayilari post.py ile esit", () => {
  assert.deepEqual(A.TONE_LOW, {
    b0: 1.0067698504364737, b1: -1.9670261142112211, b2: 0.9613064506769197,
    a1: -1.9672880375281616, a2: 0.9678143777964527,
  });
  assert.deepEqual(A.TONE_HIGH, {
    b0: 0.428238064722779, b1: -0.4673907653709056, b2: 0.1672663299532398,
    a1: -1.4337659737473694, a2: 0.5618796030524825,
  });
});

const vec = JSON.parse(readFileSync(path.join(here, "post_vectors.json"), "utf-8"));
for (const [i, c] of vec.cases.entries()) {
  test(`biquad zinciri vektor[${i}] (n=${c.input.length})`, () => {
    const low = new A.Biquad(A.TONE_LOW);
    const high = new A.Biquad(A.TONE_HIGH);
    const got = A.postProcessAudio(new Float32Array(c.input), low, high);
    assert.equal(got.length, c.expected.length);
    let maxDiff = 0;
    for (let k = 0; k < got.length; k++) {
      maxDiff = Math.max(maxDiff, Math.abs(got[k] - c.expected[k]));
    }
    // float32 depolama + float64 aritmetik: 1e-6 bollukla yeterli
    assert.ok(maxDiff < 1e-6, `maxDiff=${maxDiff}`);
  });
}

test("parca durumu tasiyici: tek sefer = parca parca", () => {
  const input = new Float32Array(vec.cases[0].input);
  const whole = A.postProcessAudio(input, new A.Biquad(A.TONE_LOW), new A.Biquad(A.TONE_HIGH));
  const low = new A.Biquad(A.TONE_LOW), high = new A.Biquad(A.TONE_HIGH);
  const half = input.length >> 1;
  const p1 = A.postProcessAudio(input.slice(0, half), low, high);
  const p2 = A.postProcessAudio(input.slice(half), low, high);
  assert.equal(p1.length + p2.length, whole.length);
  for (let k = 0; k < whole.length; k++) {
    const got = k < half ? p1[k] : p2[k - half];
    assert.ok(Math.abs(got - whole[k]) < 1e-6, `k=${k}`);
  }
});

// api.py _pause kurali: cumle sonu 0.12, diger 0.06
for (const [part, sec] of [
  ["merhaba.", 0.12], ["tamam!", 0.12], ["nasılsın?", 0.12],
  ["geldim, gittim", 0.06], ["saat on dört otuz", 0.06], ["", 0.06],
  ["boşluklu.   ", 0.12],
]) {
  test(`pauseSec(${JSON.stringify(part)}) = ${sec}`, () => {
    assert.equal(A.pauseSec(part), sec);
    assert.equal(Math.round(A.pauseSec(part) * 48000), Math.round(sec * 48000));
  });
}
