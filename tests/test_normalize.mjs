// Diferansiyel test: normalize.js vs Python to_model_text (tests/fixtures.json).
// Calistirma: node --test tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const require = createRequire(import.meta.url);
const AntaliaText = require(path.join(root, "normalize.js"));

function loadJSON(p) {
  return JSON.parse(readFileSync(p, "utf-8"));
}

const foreign = loadJSON(path.join(root, "models/foreign_dict.json"));
const parts = loadJSON(path.join(root, "models/foreign_parts.json"));
const known = loadJSON(path.join(root, "models/known_words.json"));
const englishI = loadJSON(path.join(root, "models/english_i.json"));
AntaliaText.setData({ foreign, parts, known, englishI });

// Gomulu LEXICON, commited lexicon.json ile birebir olmali.
const committed = loadJSON(path.join(root, "antalia_mini/text/data/lexicon.json"));
const встроенные = Object.fromEntries(
  Object.entries(committed).filter(([k]) => !k.startsWith("_")),
);

test("LEXICON gomulu sozlukle esit", () => {
  assert.deepEqual({ ...AntaliaText.LEXICON }, встроенные);
});

const fixtures = loadJSON(path.join(here, "fixtures.json"));

for (const [i, f] of fixtures.entries()) {
  test(`fixture[${i}] ${JSON.stringify(f.input.slice(0, 48))}`, () => {
    const got = AntaliaText.toModelText(f.input);
    assert.equal(got, f.expected);
  });
}
