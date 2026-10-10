/* Antalia-2 Mini Web — tarayıcı TTS arayüzü (@geajs/core) + çıkarım hattı.
 * Model: cloud0day3/antalia-mini (PatientDesk AI, Apache-2.0). Ağırlıklar: fr0stb1rd/antalia-mini-web-onnx.
 */
"use strict";

const ORT_VERSION = "1.30.0"; // index.html'deki CDN script ile AYNI olmalı
const HF_REPO = "fr0stb1rd/antalia-mini-web-onnx";
const MODEL_BASE = `https://huggingface.co/${HF_REPO}/resolve/main`;
const MODEL_FILES = ["text_stage.onnx", "sound_stage.onnx", "decoder.onnx"];
const CACHE_NAME = "antalia-mini-web-v1";
const HIST_KEY = "antalia-mini-web-hist";
const RATE = 48000;
const HOP = 512;
const FPS = RATE / HOP; // 93.75 Hz
const N_MELS = 128;
const STACK = 2;
const NOISE_DIM = N_MELS * STACK; // 256 (mel_dim * stack)
const MAX_DUR_FRAMES = 60;

// Model çıkarım parametreleri (config.json)
const TEMPERATURE = 0.6; // optimum flow-matching gürültü sıcaklığı
const GEM_STRETCH = 1.3; // Türkçe çift ünsüz uzatma çarpanı
const LOG_GEM = Math.log(GEM_STRETCH);
const PREFIX = "ee, ";
const SUFFIX = " ee.";
const PREFIX_INITIALS = new Set(["ö", "ı"]);

// Stüdyo Ses Son İşleme (antalia_mini post.py RBJ Biquad Filtreleri)
// 1. 90 Hz High-Pass: DC kayması ve sub-bass uğultusunu keser
// 2. 210 Hz Low-Shelf (-5 dB): boğukluğu giderir, konuşmayı berraklaştırır
// 3. Make-up Gain (+0.54 dB): ses şiddetini dengeler
// 4. Soft-Knee Peak Limiter: dijital clipping (cızırtı/çıtırtı) riskini önler
const HPF_COEFFS = {
  b0: 0.9917029581768267, b1: -1.9834059163536535, b2: 0.9917029581768267,
  a1: -1.9833370948844034, a2: 0.9834747378229038
};
const SHELF_COEFFS = {
  b0: 0.9944017033785219, b1: -1.9553249800432573, b2: 0.9614774186259423,
  a1: -1.9551093414039344, a2: 0.9560947606437868
};
const LIMIT_THRESH = Math.pow(10, -1.0 / 20); // ~0.89125 (-1 dBFS)
const LIMIT_CEIL = Math.pow(10, -0.1 / 20);   // ~0.98855 (-0.1 dBFS)
const LIMIT_ROOM = LIMIT_CEIL - LIMIT_THRESH;
const MAKEUP_GAIN = Math.pow(10, 0.54 / 20);  // ~1.064161 (+0.54 dB)

class Biquad {
  constructor(c) {
    this.b0 = c.b0; this.b1 = c.b1; this.b2 = c.b2;
    this.a1 = c.a1; this.a2 = c.a2;
    this.x1 = 0; this.x2 = 0;
    this.y1 = 0; this.y2 = 0;
  }
  process(x) {
    const y = new Float32Array(x.length);
    let { b0, b1, b2, a1, a2, x1, x2, y1, y2 } = this;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      const out = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = v;
      y2 = y1; y1 = out;
      y[i] = out;
    }
    this.x1 = x1; this.x2 = x2;
    this.y1 = y1; this.y2 = y2;
    return y;
  }
}

function postProcessAudio(samples, hpf, shelf) {
  let s = hpf.process(samples);
  s = shelf.process(s);
  for (let i = 0; i < s.length; i++) {
    let v = s[i] * MAKEUP_GAIN;
    const a = Math.abs(v);
    if (a > LIMIT_THRESH) {
      v = Math.sign(v) * (LIMIT_THRESH + LIMIT_ROOM * Math.tanh((a - LIMIT_THRESH) / LIMIT_ROOM));
    }
    s[i] = v;
  }
  return s;
}

// Antalia 47 sembol (normalize.js ile tek kaynaktan; drift onlemek icin referans)
const SYMBOLS = AntaliaText.SYMBOLS;
const STOI = AntaliaText.SYMBOL_TO_ID;

/* ---------- i18n & Tercihler ---------- */
const BROWSER_TR = (navigator.language || "tr").toLowerCase().startsWith("tr");
const PREFS_KEY = "antalia-mini-web-prefs";
let prefs = { theme: "system", lang: "auto", speed: 0.95, seed: 0, backend: "auto" };
try { Object.assign(prefs, JSON.parse(localStorage.getItem(PREFS_KEY) || "{}")); } catch { }
if (!["system", "light", "dark"].includes(prefs.theme)) prefs.theme = "system";
if (!["auto", "tr", "en"].includes(prefs.lang)) prefs.lang = "auto";
if (!(prefs.speed >= 0.25 && prefs.speed <= 4)) prefs.speed = 0.95;
if (!(Number.isInteger(prefs.seed) && prefs.seed >= 0)) prefs.seed = 0;
if (!["auto", "wasm", "webgpu"].includes(prefs.backend)) prefs.backend = "auto";
const savePrefs = () => { try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch { } };
const effLang = () => prefs.lang === "auto" ? (BROWSER_TR ? "tr" : "en") : prefs.lang;

const T = {
  tr: {
    title: "⚡ Antalia-2 Mini ONNX (Tarayıcıda)",
    sub: "Tarayıcıda çevrimdışı Türkçe TTS. Dahili normalizer ile sayıları, tarihleri ve kısaltmaları otomatik okur. Sunucu yok — her şey cihazınızda olur.",
    ph: "Okunacak Türkçe metni yazın… (Örn: '15 Ekim saat 14:30'da 2.500 TL ödemeniz var.')",
    ex: "Örnekler:", exT: ["Kısa", "Orta", "Uzun", "Sayı & Tarih", "Paragraf"],
    speed: "Hız:", say: "Sesi Üret", busy: "Üretiliyor…", stop: "Durdur",
    seed: "Seed:", first: "ilk ses",
    backend: "Motor:", beAuto: "Otomatik",
    mText: "metin", mSound: "ses", mDec: "çöz",
    playing: "Çalınıyor (kalan üretiliyor…)",
    loading: "Modeller indiriliyor (ilk sefer, ~30 MB)…",
    ready: "Hazır.",
    done: (d) => `Tamamlandı (${d} sn ses).`,
    err: (m) => `Hata: ${m}`,
    empty: "önce metin yazın",
    dl: "İndir", bald: ".wav indir", wait: "Bekleyin…",
    cached: "önbellekten",
    remain: (s) => `~${s} sn kaldı`, elapsed: (s) => `${s} sn geçti`,
    hist: "Geçmiş", emptyHist: "Henüz üretim yok.",
    replay: "Oynat", del: "Sil",
    faq: "Sık sorulanlar",
    faqs: [
      ["Antalia-2 Mini modelinin avantajı nedir?",
       "PatientDesk AI tarafından geliştirilen 7.62M parametrelik bu model, dahili Türkçe metin normalizer'a sahiptir. Sayıları ('1250'), tarihleri ('15.10.2026'), saatleri ('14:30'), yüzdeleri ('%25') ve kısaltmaları ('dr.', 'km', 'SGK') otomatik olarak doğru okur."],
      ["İnternet bağlantısı gerekli mi?",
       "Modeller ilk açılışta bir kez iner (~30 MB); sonrası tamamen çevrimdışı çalışır."],
      ["Sesim veya yazdıklarım bir yere gönderiliyor mu?",
       "Hayır. Üretim dahil her şey tarayıcınızda olur; sunucu yok."],
      ["Hangi model kullanılıyor?",
       "Antalia-2 Mini (7.62M parametre, Apache-2.0): ONNX formatına dönüştürülüp WebGPU/WASM ile çalıştırılıyor."],
      ["Hangi tarayıcılarda çalışıyor?",
       "WebGPU olan Chromium tabanlı tarayıcılarda GPU ile, diğerlerinde (Safari, Firefox dahil) WASM ile çalışır."],
      ["Sesi indirebilir miyim?",
       "Evet. Üretim bitince tarih + metin adlı 48 kHz .wav dosyası olarak iner."],
      ["Ticari kullanım serbest mi?",
       "Evet. Model de bu site de Apache-2.0 lisanslı."],
      ["İlk açılış neden yavaş?",
       "Modeller ilk seferde iner (~30 MB). Sonraki gün içi ziyaretlerde indirme olmaz; bir gün sonra tazelenir."],
      ["Uzun metinlerde ne oluyor?",
       "Metin parçalara bölünür; ilk parça hemen çalar, kalanı arkada üretilir. Arayüz donmaz."]
    ],
    clearCache: "Önbelleği temizle", redownload: "Modeli yeniden indir",
    cacheCleared: "Önbellek temizlendi.", clearHist: "Geçmişi temizle",
    histCleared: "Geçmiş temizlendi.",
    theme: "Tema:", thSystem: "Sistem", thLight: "Açık", thDark: "Koyu",
    lang: "Dil:", langAuto: "Otomatik",
    disc: `Bu yazılım bilgisayarınıza <b>~30 MB</b> model indirir ve cihazınızda çalıştırır. Bu yazılımın hiçbir garantisi yoktur. Bu yazılımı kullanarak <a href="https://github.com/fr0stb1rd/antalia-mini-web/blob/main/LICENSE">LICENSE</a>'ı okumuş ve onaylamış sayılırsınız.`
  },
  en: {
    title: "⚡ Antalia-2 Mini ONNX (in-browser)",
    sub: "Offline Turkish TTS in your browser. Automatically reads numbers, dates, and abbreviations with built-in normalizer. No server — runs on your device.",
    ph: "Type Turkish text to speak… (e.g. '15 Ekim saat 14:30'da 2.500 TL ödemeniz var.')",
    ex: "Examples:", exT: ["Short", "Medium", "Long", "Number & Date", "Paragraph"],
    speed: "Speed:", say: "Speak", busy: "Working…", stop: "Stop",
    seed: "Seed:", first: "first audio",
    backend: "Engine:", beAuto: "Auto",
    mText: "text", mSound: "sound", mDec: "decode",
    playing: "Playing (generating rest…)",
    loading: "Downloading models (first run, ~30 MB)…",
    ready: "Ready.",
    done: (d) => `Done (${d} s of audio).`,
    err: (m) => `Error: ${m}`,
    empty: "type some text first",
    dl: "Download", bald: ".wav download", wait: "Wait…",
    cached: "from cache",
    remain: (s) => `~${s} s left`, elapsed: (s) => `${s} s elapsed`,
    hist: "History", emptyHist: "Nothing yet.",
    replay: "Play", del: "Delete",
    faq: "FAQ",
    faqs: [
      ["What is Antalia-2 Mini?",
       "A lightweight 7.62M parameter Turkish TTS model by PatientDesk AI. It features a built-in text normalizer that spells out numbers, dates, times, and abbreviations automatically."],
      ["Do I need an internet connection?",
       "Models download once on first launch (~30 MB); afterwards it works offline."],
      ["Is my voice or text sent anywhere?",
       "No. Everything runs client-side in your browser."],
      ["Which browsers work?",
       "WebGPU browsers run on GPU; others (Safari, Firefox) run on WASM."],
      ["Can I download the audio?",
       "Yes. Completed speech downloads as 48 kHz studio quality .wav."]
    ],
    clearCache: "Clear cache", redownload: "Re-download model",
    cacheCleared: "Cache cleared.", clearHist: "Clear history",
    histCleared: "History cleared.",
    theme: "Theme:", thSystem: "System", thLight: "Light", thDark: "Dark",
    lang: "Language:", langAuto: "Auto",
    disc: `This software downloads <b>~30 MB</b> of models to your computer and runs them on your device. This software comes with no warranty. By using it, you agree that you have read and accepted the <a href="https://github.com/fr0stb1rd/antalia-mini-web/blob/main/LICENSE">LICENSE</a>.`
  }
};
const t = () => T[effLang()];

const EXAMPLES = [
  [ // Kisa
    "Merhaba, size nasıl yardımcı olabilirim?",
    "Günaydın, hayırlı sabahlar dilerim.",
    "Siparişiniz yola çıktı bile.",
    "Hava bugün gerçekten çok güzel.",
    "Randevunuz onaylandı, teşekkürler."
  ],
  [ // Orta
    "Siparişiniz yola çıktı, yarın sabah kapınızda olacak.",
    "Sabah erken kalktım, kahvaltı edip sahilde yürüyüş yaptım.",
    "Doktor randevum öğleden sonraydı, biraz beklemek zorunda kaldım.",
    "Yeni komşularımız dün taşındı, akşam çaya davet ettiler."
  ],
  [ // Uzun
    "Dün akşam annemlerle konuştuk, herkesin keyfi yerindeydi, bayramda memlekete gitmeyi planlıyoruz hep birlikte.",
    "Sabah işe giderken otobüsü kaçırdım, sonraki durağa kadar yürüdüm, neyse ki toplantıya son anda yetiştim.",
    "Hafta sonu evde büyük temizlik yaptık, camları sildik, akşam yorgunluktan erkenden uyuduk."
  ],
  [ // Sayi & Tarih (Antalia normalizer gucu!)
    "15 Ekim Çarşamba günü saat 14:30'da randevunuz bulunmaktadır.",
    "Toplam borcunuz 2.450 TL olup, %18 KDV dahildir.",
    "Sayın Dr. Ahmet Bey, SGK işlemleriniz için saat 10'da bekleniyorsunuz.",
    "Uçuşunuz THY ile saat 09:15'te kalkacaktır, iyi yolculuklar.",
    "Marketten 5 kg un, 2 lt süt ve 1 düzine yumurta aldım."
  ],
  [ // Paragraf
    "Sabahın erken saatlerinde liman henüz uyanmamıştı. Balıkçılar ağlarını sessizce topluyor, martılar teknelerin etrafında dönüyordu. Sahilde yürürken temiz havayı içime çektim, dalgaların kıyıya vuruşunu dinledim. Fırından taze sıcak ekmek alıp eve döndüm."
  ]
];

/* ---------- TÜRKÇE NORMALİZER (normalize.js: orijinal Python ile birebir) ----------
 * normalize.js -> antalia_mini/text (turkish-tts @ 3ed2e60) portu.
 * Sozlukler: foreign_dict + foreign_parts + known_words + english_i
 * (HF: MODEL_BASE, geri-donus: yerel models/). Hepsi Cache Storage ile onbellekli. */
const TEXT_DICT_FILES = ["foreign_dict.json", "foreign_parts.json", "known_words.json", "english_i.json"];
let textDataReady = null;

async function loadJsonDict(name) {
  try {
    const { res } = await cachedResponse(`${MODEL_BASE}/${name}`);
    return await res.json();
  } catch (e) {
    const local = await fetch(`models/${name}`);
    if (!local.ok) throw new Error(`${name} indirilemedi (${e.message || e})`);
    return await local.json();
  }
}

async function loadTextData() {
  if (!textDataReady) {
    textDataReady = (async () => {
      const [foreign, parts, known, englishI] = await Promise.all(TEXT_DICT_FILES.map(loadJsonDict));
      AntaliaText.setData({ foreign, parts, known, englishI });
      return true;
    })().catch((e) => { textDataReady = null; throw e; });
  }
  return textDataReady;
}

// Sayfa ilk yüklendiğinde arka planda sessizce indir ve önbelleğe al
if (typeof window !== "undefined") {
  setTimeout(() => loadTextData().catch(() => {}), 100);
}

function normalizeTurkishText(raw) {
  return AntaliaText.toModelText(raw);
}

function encodeText(cleanText) {
  return AntaliaText.encode(cleanText);
}

function splitModelText(text, maxChars = 250, firstChars = 80) {
  const sentences = text.split(/(?<=[.?!])\s+/).map(s => s.trim()).filter(Boolean);
  if (!sentences.length) return [];
  const chunks = [];
  let cur = "";
  for (let i = 0; i < sentences.length; i++) {
    const s = sentences[i];
    const limit = chunks.length === 0 ? firstChars : maxChars;
    if (cur && (cur.length + 1 + s.length > limit)) {
      chunks.push(cur);
      cur = s;
    } else {
      cur = cur ? `${cur} ${s}` : s;
    }
  }
  if (cur) chunks.push(cur);
  return chunks;
}

/* ---------- Gea Store ---------- */
const { Store, Component, GEA_OBSERVER_REMOVERS } = gea;
class UI extends Store {
  phase = "idle"; // idle|loading|ready|busy|playing|done|error
  text = EXAMPLES[0][0];
  speed = prefs.speed;
  seed = prefs.seed;
  pct = 0; stats = ""; status = ""; dlSeq = 0;
  audioURL = ""; audioSize = ""; histSeq = 0;
}
const ui = new UI();

/* ---------- Yardımcılar ---------- */
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtMB = (b) => b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`;
const fmtS = (s) => s < 60 ? `${Math.round(s)}` : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

function dlName(text) {
  const tr = { "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u" };
  const slug = (text || "antalia").replaceAll("İ", "i").replaceAll("I", "ı").toLowerCase()
    .split("").map((c) => tr[c] || c).join("")
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "antalia";
  const d = new Date(), p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_${p(d.getHours())}-${p(d.getMinutes())}_${slug}.wav`;
}

function applyTheme() {
  if (prefs.theme === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", prefs.theme);
}

function fillSelects(root) {
  const th = root.querySelector(".theme"), ls = root.querySelector(".langsel");
  if (th) th.innerHTML = [["system", t().thSystem], ["light", t().thLight], ["dark", t().thDark]]
    .map(([v, l]) => `<option value="${v}"${prefs.theme === v ? " selected" : ""}>${l}</option>`).join("");
  if (ls) ls.innerHTML = [["auto", t().langAuto], ["tr", "Türkçe"], ["en", "English"]]
    .map(([v, l]) => `<option value="${v}"${prefs.lang === v ? " selected" : ""}>${l}</option>`).join("");
  paintEp(root);
}

function paintEp(root) {
  const be = root.querySelector(".be");
  if (be) be.innerHTML = [["auto", t().beAuto], ["webgpu", "WebGPU"], ["wasm", "WASM"]]
    .map(([v, l]) => `<option value="${v}"${prefs.backend === v ? " selected" : ""}>${l}</option>`).join("");
}

const IS_IOS = /iPhone|iPad|iPod/.test(navigator.userAgent) ||
  (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
let effEP = "wasm";
function effEPS() {
  const list = prefs.backend === "auto"
    ? (IS_IOS ? ["wasm"] : ["webgpu", "wasm"])
    : [prefs.backend];
  effEP = list[0];
  return list;
}

async function setBackend(v) {
  if (ui.phase === "busy" || ui.phase === "playing" || ui.phase === "loading") return;
  prefs.backend = v; savePrefs();
  if (!sessText) return;
  try {
    for (const s of [sessText, sessSound, sessDec]) await s?.release().catch(() => {});
    sessText = sessSound = sessDec = null;
    await loadModels();
  } catch (e) { ui.phase = "error"; ui.status = t().err(e.message); }
}

/* ---------- Cache-First İndirme ---------- */
const CACHE_TTL = 24 * 3600 * 1000; // 1 gün
const TS_KEY = "antalia-mini-web-cache-ts";
const cacheTs = () => { try { return JSON.parse(localStorage.getItem(TS_KEY) || "{}"); } catch { return {}; } };
const stampCache = (url) => {
  try {
    const m = cacheTs(); m[url] = Date.now();
    localStorage.setItem(TS_KEY, JSON.stringify(m));
  } catch { }
};

async function cachedResponse(url) {
  const jar = ("caches" in self) ? await caches.open(CACHE_NAME).catch(() => null) : null;
  if (jar) {
    const hit = await jar.match(url).catch(() => null);
    if (hit) {
      if (Date.now() - (cacheTs()[url] || 0) < CACHE_TTL) return { res: hit, fromCache: true, jar };
      jar.delete(url).catch(() => { });
    }
  }
  const net = await fetch(url);
  if (!net.ok) throw new Error(`${url.split("/").pop()} (HTTP ${net.status})`);
  if (jar && (net.type === "basic" || net.type === "cors")) {
    jar.put(url, net.clone()).catch(() => { });
    stampCache(url);
  }
  return { res: net, fromCache: false, jar };
}

async function fetchBuffer(url, onTick) {
  const { res, fromCache } = await cachedResponse(url);
  const total = Number(res.headers.get("content-length")) || 0;
  const chunks = []; let loaded = 0;
  const reader = res.body.getReader();
  for (; ;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.byteLength;
    onTick(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let o = 0; for (const c of chunks) { buf.set(c, o); o += c.byteLength; }
  return { buf, fromCache };
}

/* ---------- Model Oturumu ---------- */
let sessText = null, sessSound = null, sessDec = null;

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(len, rand) {
  const o = new Float32Array(len);
  for (let i = 0; i < len; i += 2) {
    const u1 = Math.max(rand(), 1e-9), u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2;
    o[i] = r * Math.cos(th);
    if (i + 1 < len) o[i + 1] = r * Math.sin(th);
  }
  return o;
}

async function loadModels() {
  ui.phase = "loading"; ui.status = t().loading; ui.pct = 0; ui.stats = "";
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  ort.env.wasm.proxy = true;

  const files = MODEL_FILES;
  const t0 = performance.now();
  const state = Object.fromEntries(files.map((f) => [f, { loaded: 0, total: 0 }]));
  let last = 0, anyCached = false;

  const tick = () => {
    const now = performance.now();
    if (now - last < 120) return;
    last = now;
    const loaded = files.reduce((a, f) => a + state[f].loaded, 0);
    const total = files.reduce((a, f) => a + state[f].total, 0);
    const el = (now - t0) / 1000, spd = loaded / Math.max(el, 0.01);
    ui.pct = total ? Math.min(100, (100 * loaded) / total) : 0;
    const left = spd > 0 && total ? Math.max(0, (total - loaded) / spd) : 0;
    ui.stats = (total
      ? `${ui.pct.toFixed(0)}% • ${fmtMB(loaded)} / ${fmtMB(total)} • ${fmtMB(spd)}/sn • ${t().elapsed(fmtS(el))} • ${t().remain(fmtS(left))}`
      : `${fmtMB(loaded)} • ${fmtMB(spd)}/sn • ${t().elapsed(fmtS(el))}`)
      + (anyCached ? ` • ${t().cached}` : "");
    ui.dlSeq++;
  };

  const opt = () => ({ executionProviders: effEPS(), graphOptimizationLevel: "all" });

  const makeSessions = (o) => files.map(async (f) => {
    const { buf, fromCache } = await fetchBuffer(`${MODEL_BASE}/${f}`,
      (loaded, total) => { state[f] = { loaded, total }; tick(); });
    if (fromCache) anyCached = true;
    state[f].loaded = state[f].total || state[f].loaded; tick();
    return ort.InferenceSession.create(buf, o);
  });

  const createAll = async (o) => {
    try {
      return await Promise.all(makeSessions(o));
    } catch (e) {
      ort.env.wasm.proxy = false;
      return await Promise.all(makeSessions(o));
    }
  };

  const warm = async () => { for await (const _ of synthPieces("Merhaba.", 1, 0)) {} };

  try {
    const [sessions] = await Promise.all([createAll(opt()), loadTextData()]);
    [sessText, sessSound, sessDec] = sessions;
    await warm();
  } catch (e) {
    if (!effEPS().includes("webgpu")) throw e;
    for (const s of [sessText, sessSound, sessDec]) await s?.release().catch(() => {});
    effEP = "wasm";
    [sessText, sessSound, sessDec] = await createAll({ ...opt(), executionProviders: ["wasm"] });
    await warm();
  }

  ui.pct = 100;
  const totalAll = files.reduce((a, f) => a + (state[f].total || state[f].loaded), 0);
  ui.stats = `100% • ${fmtMB(totalAll)} / ${fmtMB(totalAll)}` + (anyCached ? ` • ${t().cached}` : "");
  ui.dlSeq++;
  ui.phase = "ready"; ui.status = t().ready;
}

/* ---------- Sentezleme & Parça Akışı ---------- */
async function* synthPieces(text, speed, seed, stats = null) {
  const clean = normalizeTurkishText(text);
  if (!clean) return;
  const chunks = splitModelText(clean);
  let seedI = 0;
  const tick = (key, t0) => { if (stats) stats[key] = (stats[key] || 0) + performance.now() - t0; };

  // Akış parçaları boyunca biquad filtre geçmişi korunur (antalia_mini StreamFIR ile birebir aynı)
  const hpf = new Biquad(HPF_COEFFS);
  const shelf = new Biquad(SHELF_COEFFS);

  for (const part of chunks) {
    // 0. Ön Ek & Son Ek (ö ve ı başlangıcındaki parazit "l" artefaktını gidermek ve bitiş kırpılmasını önlemek için)
    const trimmed = part.trim();
    const firstChar = trimmed[0];
    const hasPrefix = Boolean(firstChar && PREFIX_INITIALS.has(firstChar));
    const prefixStr = hasPrefix ? PREFIX : "";
    const suffixStr = SUFFIX;
    const fullChunkText = prefixStr + part + suffixStr;

    const ids = encodeText(fullChunkText);
    const L = ids.length;
    const nPreTokens = hasPrefix ? (encodeText(prefixStr).length - 2) : 0;
    const nSufTokens = encodeText(suffixStr).length - 2;

    let t0 = performance.now();

    // 1. Text Stage
    const tOut = await sessText.run({
      ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, L]),
      mask: new ort.Tensor("bool", new Uint8Array(L).fill(1), [1, L])
    });
    const dText = tOut.h.dims[2];
    const h = await tOut.h.getData();
    const logd = new Float32Array(await tOut.logd.getData());
    tick("text", t0);

    // Çift ünsüz uzatma (gem_stretch = 1.3): Türkçe çift ünsüzlerin süresi log(1.3) kadar uzatılır
    const spaceId = STOI[" "];
    for (let i = 2; i < L - 1; i++) {
      if (ids[i] === ids[i - 1] && ids[i] !== spaceId) {
        logd[i] += LOG_GEM;
      } else if (ids[i - 1] === spaceId && ids[i] === ids[i - 2]) {
        logd[i - 1] += LOG_GEM;
      }
    }

    // Frame durations (clamp 1..MAX_DUR_FRAMES)
    const dur = new Int32Array(L);
    let totalFrames = 0;
    for (let i = 0; i < L; i++) {
      dur[i] = Math.max(1, Math.min(MAX_DUR_FRAMES, Math.round(Math.exp(logd[i]) / speed)));
      totalFrames += dur[i];
    }
    const melLen = Math.ceil(totalFrames / STACK) * STACK;

    // Kelime kelime vurgulama zamanları (yalnızca kullanıcı metnine ait kelimeler)
    const rawWords = part.split(/\s+/).filter(Boolean);
    const words = rawWords.length ? rawWords : [part];
    const starts = [];
    let cumF = 0, curWord = 0;
    const textStartToken = 1 + nPreTokens;
    const textEndToken = L - 1 - nSufTokens;
    for (let i = textStartToken; i < textEndToken; i++) {
      if (ids[i] === spaceId) {
        curWord++;
      } else {
        if (starts.length === curWord) {
          starts.push(+(cumF / FPS).toFixed(3));
        }
      }
      cumF += dur[i];
    }
    while (starts.length < words.length) starts.push(+(cumF / FPS).toFixed(3));

    // Expand h -> cond [1, melLen, dText]
    const cond = new Float32Array(melLen * dText);
    let fIdx = 0;
    for (let i = 0; i < L; i++) {
      const d = dur[i];
      for (let k = 0; k < d && fIdx < melLen; k++) {
        for (let j = 0; j < dText; j++) cond[fIdx * dText + j] = h[i * dText + j];
        fIdx++;
      }
    }
    while (fIdx < melLen) {
      for (let j = 0; j < dText; j++) cond[fIdx * dText + j] = h[(L - 1) * dText + j];
      fIdx++;
    }

    // Gürültü üretimi (optimum temperature: 0.6 ile ölçeklenir)
    const rand = mulberry32((seed * 1000003 + seedI++) >>> 0);
    const noise = randn((melLen / STACK) * NOISE_DIM, rand);
    for (let i = 0; i < noise.length; i++) noise[i] *= TEMPERATURE;

    t0 = performance.now();
    // 2. Sound Stage (8-step DiT flow-matching with CFG 2.0)
    const sOut = await sessSound.run({
      cond: new ort.Tensor("float32", cond, [1, melLen, dText]),
      fmask: new ort.Tensor("bool", new Uint8Array(melLen).fill(1), [1, melLen]),
      noise: new ort.Tensor("float32", noise, [1, melLen / STACK, NOISE_DIM])
    });
    const mel = await sOut.mel.getData();
    tick("sound", t0);

    t0 = performance.now();
    // 3. Decoder (Vocos ConvNeXt + iSTFT)
    const dOut = await sessDec.run({
      mel: new ort.Tensor("float32", mel, [1, N_MELS, melLen])
    });
    const wavRaw = new Float32Array(await dOut.audio.getData());
    tick("decode", t0);

    // Kırpma (Prefix ve Suffix seslerini hassas şekilde kesip çıkarma):
    let cumTokens = 0;
    let spanA = 0, spanB = wavRaw.length;
    for (let i = 0; i < L; i++) {
      cumTokens += dur[i];
      if (i === nPreTokens && hasPrefix) {
        spanA = cumTokens * HOP;
      }
      if (i === L - 2 - nSufTokens) {
        spanB = cumTokens * HOP;
      }
    }
    const chunkWav = wavRaw.slice(Math.max(0, spanA), Math.min(wavRaw.length, spanB));

    // 5 ms yumuşak fade
    const fadeSamples = Math.floor(0.005 * RATE);
    if (hasPrefix && chunkWav.length > 2 * fadeSamples) {
      for (let i = 0; i < fadeSamples; i++) chunkWav[i] *= (i / fadeSamples);
    }
    if (chunkWav.length > 2 * fadeSamples) {
      for (let i = 0; i < fadeSamples; i++) chunkWav[chunkWav.length - 1 - i] *= (i / fadeSamples);
    }

    // 4. Stüdyo Ses Son İşleme (90 Hz HPF + 210 Hz Low-Shelf EQ + Make-up Gain + Soft-Knee Limiter)
    const audio = postProcessAudio(chunkWav, hpf, shelf);

    yield { audio, words, starts, seconds: audio.length / RATE };
  }
}

/* ---------- Ses Çalma & WAV ---------- */
let AC = null, PS = null, runId = 0, lastGenText = "";
function ensureCtx() {
  if (!AC) {
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RATE });
    const master = ctx.createGain(); master.connect(ctx.destination);
    AC = { ctx, master };
    hookMediaSession();
  }
  return AC;
}

function hookMediaSession() {
  if (!("mediaSession" in navigator) || !AC) return;
  try {
    navigator.mediaSession.setActionHandler("play", () => AC.ctx.resume().catch(() => {}));
    navigator.mediaSession.setActionHandler("pause", () => AC.ctx.suspend().catch(() => {}));
  } catch {}
}

function setMediaText(text) {
  if (!("mediaSession" in navigator)) return;
  try {
    navigator.mediaSession.metadata = new MediaMetadata({
      title: text.slice(0, 80) || "Antalia-2 Mini",
      artist: "Antalia-2 Mini",
      album: effLang() === "tr" ? "Tarayıcıda Türkçe TTS" : "In-browser Turkish TTS",
    });
  } catch {}
}

function stopAll() {
  runId++;
  if (PS) { for (const s of PS.sources) try { s.stop(); } catch {} }
  PS = null;
  if (ui.phase === "busy" || ui.phase === "playing") { ui.phase = "ready"; ui.status = t().ready; }
}

function toWav(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const s = (o, str) => { for (let i = 0; i < str.length; i++) v.setUint8(o + i, str.charCodeAt(i)); };
  s(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); s(8, "WAVE"); s(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, RATE, true); v.setUint32(28, RATE * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  s(36, "data"); v.setUint32(40, samples.length * 2, true);
  let p = 44;
  for (let i = 0; i < samples.length; i++) {
    const x = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(p, x < 0 ? x * 0x8000 : x * 0x7FFF, true); p += 2;
  }
  return new Blob([v], { type: "audio/wav" });
}

const concat = (parts) => {
  const out = new Float32Array(parts.reduce((a, x) => a + x.length, 0));
  let o = 0; for (const c of parts) { out.set(c, o); o += c.length; }
  return out;
};

/* ---------- Geçmiş (History) ---------- */
let hist = [];
try { hist = JSON.parse(localStorage.getItem(HIST_KEY) || "[]"); } catch {}
function saveHist() {
  try {
    const meta = hist.map(({ text, speed, seed, be, dur, size, words, starts }) =>
      ({ text, speed, seed, be, dur, size, words, starts }));
    localStorage.setItem(HIST_KEY, JSON.stringify(meta));
  } catch {}
}

function dropAudio() {
  stopAll();
  if (ui.audioURL) { URL.revokeObjectURL(ui.audioURL); ui.audioURL = ""; }
}

async function clearCache() {
  try {
    if ("caches" in self) await caches.delete(CACHE_NAME).catch(() => {});
    try { localStorage.removeItem(TS_KEY); } catch {}
    hist.forEach((h) => { h.audio = null; });
    dropAudio();
    saveHist(); ui.histSeq++;
    ui.status = t().cacheCleared;
  } catch (e) { ui.status = t().err(e.message); }
}

function clearHist() {
  dropAudio();
  hist = [];
  saveHist(); ui.histSeq++;
  ui.status = t().histCleared;
}

async function redownload() {
  if (ui.phase === "busy" || ui.phase === "playing" || ui.phase === "loading") return;
  try {
    if ("caches" in self) {
      const jar = await caches.open(CACHE_NAME).catch(() => null);
      if (jar) for (const f of MODEL_FILES) await jar.delete(`${MODEL_BASE}/${f}`).catch(() => {});
    }
    for (const s of [sessText, sessSound, sessDec]) await s?.release().catch(() => {});
    sessText = sessSound = sessDec = null;
    await loadModels();
  } catch (e) { ui.phase = "error"; ui.status = t().err(e.message); }
}

/* ---------- Gea Bileşeni (App Component) ---------- */
class App extends Component {
  template() {
    return `
      <div id="${this.id}">
        <div class="top">
          <h1>${t().title}</h1>
          <div class="set">
            <label>${t().theme} <select class="theme"></select></label>
            <label>${t().lang} <select class="langsel"></select></label>
            <label><span class="belab">${t().backend}</span> <select class="be"></select> <span class="beeff"></span></label>
          </div>
        </div>
        <p class="sub">${t().sub}</p>
        <textarea class="txt" placeholder="${t().ph}">${esc(ui.text)}</textarea>
        <div class="ex"><span class="exlab">${t().ex}</span>${EXAMPLES.map((x, i) =>
      `<button class="ghost exb" data-i="${i}">${t().exT[i]}</button>`).join("")}</div>
        <div class="row">
          <label class="speed"><span class="spdlab">${t().speed}</span> <input class="spd" type="number" value="${ui.speed}" step="0.05" min="0.25" max="4"></label>
          <label class="speed"><span class="seedlab">${t().seed}</span> <input class="seed" type="number" value="${ui.seed}" step="1" min="0"></label>
          <button class="ghost dice" title="🎲">🎲</button>
          <button class="ghost dl" disabled>${t().dl}</button>
          <button class="say">${t().say}</button>
        </div>
        <div class="out" hidden>
          <div class="bar" hidden><i></i></div>
          <div class="stats"></div>
          <p class="status"></p>
          <p class="now"></p>
        </div>
        <div class="hh" hidden><h2>${t().hist}</h2><div class="hl"></div></div>
        <div class="row store">
          <button class="ghost sclr">${t().clearCache}</button>
          <button class="ghost sredl">${t().redownload}</button>
          <button class="ghost shist">${t().clearHist}</button>
        </div>
        <div class="faq"><h2></h2><div class="fl"></div></div>
        <p class="foot"><a href="https://github.com/fr0stb1rd/antalia-mini-web">antalia-mini-web</a> · model: <a href="https://huggingface.co/cloud0day3/antalia-mini">cloud0day3/antalia-mini</a> (Apache-2.0) · onnx: <a href="https://huggingface.co/fr0stb1rd/antalia-mini-web-onnx">antalia-mini-web-onnx</a></p>
        <p class="foot disc"></p>
      </div>`;
  }

  createdHooks() {
    const R = this[GEA_OBSERVER_REMOVERS];
    R.push(ui.observe("phase", () => this.paint()));
    R.push(ui.observe("dlSeq", () => this.paintDl()));
    R.push(ui.observe("audioURL", () => this.paintAudio()));
    R.push(ui.observe("histSeq", () => this.paintHist()));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && (ui.phase === "busy" || ui.phase === "playing")) stopAll();
    });
    this.$(".spd").addEventListener("dblclick", () => {
      ui.speed = 0.95; prefs.speed = 0.95; savePrefs(); this.$(".spd").value = 0.95;
    });
    this.$(".seed").addEventListener("dblclick", () => {
      ui.seed = 0; prefs.seed = 0; savePrefs(); this.$(".seed").value = 0;
    });
    this.autosize();
    effEPS();
    applyTheme(); fillSelects(this.el); this.applyLang();
    this.paint(); this.paintHist();
  }

  autosize() {
    const el = this.$(".txt");
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, innerHeight * 0.5) + "px";
  }

  applyLang() {
    const q = (s) => this.$(s);
    q("h1").textContent = t().title;
    q(".sub").textContent = t().sub;
    q(".txt").placeholder = t().ph;
    q(".exlab").textContent = t().ex;
    this.$$(".exb").forEach((b, i) => { b.textContent = t().exT[i]; });
    q(".spdlab").textContent = t().speed;
    q(".seedlab").textContent = t().seed;
    fillSelects(this.el);
    q(".hh h2").textContent = t().hist;
    q(".disc").innerHTML = t().disc;
    q(".faq h2").textContent = t().faq;
    q(".faq .fl").innerHTML = t().faqs.map(([q_, a]) =>
      `<details><summary>${esc(q_)}</summary><p>${esc(a)}</p></details>`).join("");
    q(".sclr").textContent = t().clearCache;
    q(".sredl").textContent = t().redownload;
    q(".shist").textContent = t().clearHist;
    q(".belab").textContent = t().backend;
    this.$(".beeff").textContent = `(${effEP})`;
    if (ui.phase === "ready") ui.status = t().ready;
    this.paint(); this.paintHist();
  }

  paint() {
    const loading = ui.phase === "loading", active = ui.phase === "busy" || ui.phase === "playing";
    this.$(".say").disabled = loading;
    this.$(".say").textContent = active ? t().stop : t().say;
    this.$(".say").classList.toggle("pulse", !loading && !active);
    this.$(".bar").hidden = !(loading || ui.phase === "ready");
    this.$(".status").textContent = ui.status;
    this.$(".beeff").textContent = `(${effEP})`;
    this.$(".out").hidden = !(ui.status || ui.stats || this.$(".now").innerHTML);
    this.paintDl(); this.paintAudio();
  }

  paintDl() {
    this.$(".bar i").style.width = `${ui.pct}%`;
    this.$(".stats").textContent = ui.stats;
  }

  paintAudio() {
    const dl = this.$(".dl");
    dl.hidden = false;
    const busy = ui.phase === "busy" || ui.phase === "playing" || ui.phase === "loading";
    if (ui.audioURL) { dl.disabled = false; dl.textContent = `${t().dl} (${ui.audioSize})`; }
    else if (busy) { dl.disabled = true; dl.textContent = t().wait; }
    else { dl.disabled = true; dl.textContent = t().dl; }
  }

  paintHist() {
    const box = this.$(".hh"), list = this.$(".hl");
    box.hidden = hist.length === 0;
    if (!hist.length) { list.innerHTML = ""; return; }
    list.innerHTML = hist.map((h, i) => `
      <div class="hrow" data-i="${i}">
        <span class="ht">${esc(h.text.slice(0, 60))}${h.text.length > 60 ? "…" : ""}</span>
        <span class="hm">${h.dur} sn • ${h.size}</span>
        <button class="ghost hplay">${t().replay}</button>
        ${h.audio ? `<button class="ghost hdl">${t().dl}</button>` : ""}
        <button class="ghost hdel" title="${t().del}">×</button>
      </div>`).join("");
  }

  get events() {
    return {
      click: {
        ".say": () => this.onSay(),
        "details summary": (e) => {
          const d = e.target.closest("details");
          this.$$("details").forEach((x) => { if (x !== d && x.open) x.open = false; });
        },
        ".exb": (e) => {
          const tier = EXAMPLES[Number(e.target.dataset.i)] || EXAMPLES[0];
          let pick = tier[Math.floor(Math.random() * tier.length)];
          ui.text = pick;
          this.$(".txt").value = pick;
          this.autosize();
        },
        ".dl": () => {
          const a = document.createElement("a");
          a.href = ui.audioURL; a.download = dlName(lastGenText || ui.text); a.click();
        },
        ".hplay": (e) => this.onHistPlay(Number(e.target.closest(".hrow").dataset.i)),
        ".hdl": (e) => this.onHistDl(Number(e.target.closest(".hrow").dataset.i)),
        ".hdel": (e) => {
          const i = Number(e.target.closest(".hrow").dataset.i);
          if (hist[i]) { hist.splice(i, 1); saveHist(); ui.histSeq++; }
        },
        ".dice": () => {
          ui.seed = Math.floor(Math.random() * 100000);
          prefs.seed = ui.seed; savePrefs();
          this.$(".seed").value = ui.seed;
        },
        ".sclr": () => clearCache(),
        ".sredl": () => redownload(),
        ".shist": () => clearHist(),
      },
      input: {
        ".txt": (e) => { ui.text = e.target.value; this.autosize(); },
      },
      keydown: {
        ".txt": (e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); this.onSay(); }
        },
      },
      change: {
        ".spd": (e) => { ui.speed = parseFloat(e.target.value) || 0.95; prefs.speed = ui.speed; savePrefs(); },
        ".seed": (e) => {
          ui.seed = Math.max(0, parseInt(e.target.value) || 0);
          prefs.seed = ui.seed; savePrefs();
        },
        ".be": (e) => setBackend(e.target.value),
        ".theme": (e) => { prefs.theme = e.target.value; savePrefs(); applyTheme(); },
        ".langsel": (e) => { prefs.lang = e.target.value; savePrefs(); this.applyLang(); },
      },
    };
  }

  paintNow(words, idx) {
    this.$(".now").innerHTML = words.map((w, i) =>
      `<span class="${i < idx ? "said" : i === idx ? "on" : ""}">${esc(w)}</span>`).join(" ");
    this.$(".out").hidden = false;
  }

  follow(id) {
    if (id !== runId || !PS) return;
    const t = AC.ctx.currentTime;
    const piece = PS.pieces.filter((x) => x.start <= t + 0.03).at(-1);
    if (piece && piece !== PS.shown) {
      PS.shown = piece; PS.word = -1;
      this.paintNow(piece.words, -1);
    }
    if (piece) {
      const rel = t - piece.start;
      let w = -1;
      for (let i = 0; i < piece.words.length; i++) if ((piece.starts[i] ?? Infinity) <= rel) w = i;
      if (rel > piece.seconds) w = piece.words.length;
      if (w !== PS.word) { PS.word = w; this.paintNow(piece.words, w); }
    }
    if (PS.done && t > PS.endAt + 0.15) {
      if (id !== runId || !PS) return;
      PS = null;
      ui.phase = "done";
      return;
    }
    requestAnimationFrame(() => this.follow(id));
  }

  async onSay() {
    if (ui.phase === "busy" || ui.phase === "playing") { stopAll(); return; }
    if (ui.phase === "loading") return;
    if (!ui.text.trim()) { ui.phase = "error"; ui.status = t().err(t().empty); return; }
    const text = ui.text, speed = ui.speed, seed = ui.seed;
    try {
      effEPS();
      const hi = hist.findIndex((h) => h.audio && h.text === text && h.speed === speed &&
        h.seed === seed && h.be === effEP);
      if (hi >= 0) {
        const [h] = hist.splice(hi, 1);
        hist.unshift(h); saveHist(); ui.histSeq++;
        lastGenText = h.text;
        await this.playStored(h);
        ui.status = t().done(h.dur);
        return;
      }
      const { ctx, master } = ensureCtx();
      await ctx.resume().catch(() => {});
      if (!sessText) await loadModels();
      stopAll();
      if (ui.audioURL) { URL.revokeObjectURL(ui.audioURL); ui.audioURL = ""; }
      this.$(".now").innerHTML = "";
      ui.phase = "busy"; ui.status = t().busy;
      const t0 = performance.now();
      const stats = {};
      const id = ++runId;
      const ps = PS = { sources: [], pieces: [], at: ctx.currentTime + 0.08, done: false, endAt: 0, shown: null, word: -1 };
      const sched = (samples) => {
        ctx.resume().catch(() => {});
        const buf = ctx.createBuffer(1, samples.length, RATE);
        buf.copyToChannel(samples, 0);
        const src = ctx.createBufferSource();
        src.buffer = buf; src.connect(master);
        const start = Math.max(ps.at, ctx.currentTime + 0.02);
        src.start(start);
        ps.at = start + buf.duration; ps.endAt = ps.at;
        ps.sources.push(src);
        return start;
      };
      const chunks = [];
      let firstMs = null;
      for await (const part of synthPieces(text, speed, seed, stats)) {
        if (id !== runId) return;
        const start = sched(part.audio);
        ps.pieces.push({ words: part.words, starts: part.starts, seconds: part.seconds, start });
        chunks.push(part.audio);
        if (firstMs === null) {
          firstMs = performance.now() - t0;
          ui.phase = "playing"; ui.status = t().playing;
          setMediaText(text);
          requestAnimationFrame(() => this.follow(id));
        }
      }
      if (id !== runId) return;
      ps.done = true;
      const audio = concat(chunks);
      const blob = toWav(audio);
      ui.audioURL = URL.createObjectURL(blob);
      ui.audioSize = fmtMB(blob.size);
      const dur = (audio.length / RATE).toFixed(1);
      const el = (performance.now() - t0) / 1000;
      const rtf = Math.round(audio.length / RATE / Math.max(el, 0.01));
      lastGenText = text;
      const allWords = [], allStarts = [];
      for (const pc of ps.pieces)
        pc.words.forEach((w, i) => { allWords.push(w); allStarts.push(+(pc.start + (pc.starts[i] || 0)).toFixed(3)); });
      const t00 = allStarts.length ? allStarts[0] : 0;
      hist.unshift({ text, speed, seed, be: effEP, dur, size: ui.audioSize, audio,
        words: allWords, starts: allStarts.map((s) => +(s - t00).toFixed(3)) });
      hist = hist.slice(0, 20);
      hist.forEach((h, i) => { if (i > 4) h.audio = null; });
      saveHist(); ui.histSeq++;
      ui.status = `${t().done(dur)} • ${t().first} ${Math.round(firstMs)} ms • ${rtf}×`;
      ui.stats = `${t().mText} ${Math.round(stats.text || 0)}ms • ${t().mSound} ${Math.round(stats.sound || 0)}ms • ${t().mDec} ${Math.round(stats.decode || 0)}ms`;
      ui.dlSeq++;
      if (ctx.currentTime > ps.endAt) { PS = null; ui.phase = "done"; }
    } catch (e) {
      ui.phase = "error"; ui.status = t().err(e.message);
    }
  }

  async playStored(h) {
    stopAll();
    if (ui.audioURL) URL.revokeObjectURL(ui.audioURL);
    ui.audioURL = URL.createObjectURL(toWav(h.audio));
    ui.audioSize = h.size;
    const { ctx, master } = ensureCtx();
    await ctx.resume().catch(() => {});
    const id = ++runId;
    const ps = PS = { sources: [], pieces: [], at: 0, done: true, endAt: 0, shown: null, word: -1 };
    const buf = ctx.createBuffer(1, h.audio.length, RATE);
    buf.copyToChannel(h.audio, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(master);
    ps.at = ctx.currentTime + 0.02;
    src.start(ps.at); ps.endAt = ps.at + buf.duration; ps.sources.push(src);
    ps.pieces.push({ words: h.words?.length ? h.words : h.text.split(" "),
      starts: h.starts?.length ? h.starts : [], seconds: h.audio.length / RATE, start: ps.at });
    setMediaText(h.text);
    ui.phase = "playing"; ui.status = t().playing;
    requestAnimationFrame(() => this.follow(id));
  }

  async onHistPlay(i) {
    if (ui.phase === "loading") return;
    const h = hist[i];
    if (!h) return;
    if (h.audio) {
      stopAll();
      hist.splice(i, 1); hist.unshift(h); saveHist(); ui.histSeq++;
      lastGenText = h.text;
      await this.playStored(h);
    } else {
      if (ui.phase === "busy" || ui.phase === "playing") stopAll();
      ui.text = h.text; this.$(".txt").value = h.text;
      ui.speed = h.speed; this.$(".spd").value = h.speed;
      this.onSay();
    }
  }

  onHistDl(i) {
    const h = hist[i];
    if (!h || !h.audio) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(toWav(h.audio));
    a.download = dlName(h.text); a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }
}

new App().render(document.getElementById("app"));
