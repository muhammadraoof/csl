// ============================================================
// Sign to Voice: webcam -> hand landmarks (up to 2 hands) -> sign -> text -> speech
// Everything runs in the browser. No server, no video upload.
//
// Recognition has two layers:
//   1. Signs the user taught (k-NN on one-hand OR two-hand features)
//   2. Built-in basic ISL vocabulary (rules in signs.js, incl. motion + two hands)
// ============================================================

const STORAGE_KEY = "sign_to_voice_samples_v1";   // same key: old saved signs still load
const RECORD_FRAMES = 50;      // how many frames one recording captures
const K = 5;                   // neighbours used by the classifier
const ONE_HAND_LEN = 63;       // 21 points x 3
const TWO_HAND_LEN = 128;      // 2 x 63 + 2 (relative position of the hands)

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);
const videoEl = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const statusEl = $("status");
const liveLabel = $("liveLabel").querySelector("b");
const handsLabel = $("handsLabel").querySelector("b");
const sentenceEl = $("sentence");
const signListEl = $("signList");
const progressWrap = $("recordProgress");
const progressBar = progressWrap.firstElementChild;
const practiceEl = $("practiceBox");

// ---------- State ----------
let samples = loadSamples();   // [{label, f:[63 or 128 numbers], n:1|2}]
let words = [];                // translated sentence (stored as English labels)
let recording = null;          // {label, count, n} while recording
let candidate = null;          // label currently being held
let candidateCount = 0;
let lastAdded = null;          // stops the same sign repeating while held
let holdFrames = 15;
let maxDistance = 1.2;
let target = null;             // sign chosen in practice mode
const recognizer = ISL.createRecognizer();

// ---------- Storage ----------
function loadSamples() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    return raw.filter((s) => s && s.label && Array.isArray(s.f))
              .map((s) => ({ label: s.label, f: s.f, n: s.f.length === ONE_HAND_LEN ? 1 : 2 }));
  } catch (e) {
    return [];
  }
}
function saveSamples() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  } catch (e) {
    setStatus("Could not save signs (browser storage full or blocked).");
  }
}

// ---------- Helpers ----------
function setStatus(msg) { statusEl.textContent = msg; }
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// One hand -> 63 numbers, independent of position and size:
// wrist moved to (0,0,0), scaled so the farthest point is at distance 1.
function toFeatures(lm) {
  const base = lm[0];
  const pts = lm.map((p) => [p.x - base.x, p.y - base.y, p.z - base.z]);
  let maxD = 0;
  for (const p of pts) maxD = Math.max(maxD, Math.hypot(p[0], p[1], p[2]));
  const scale = maxD || 1;
  const out = [];
  for (const p of pts) for (const v of p) out.push(Math.round((v / scale) * 1000) / 1000);
  return out;
}

// All visible hands -> one feature vector.
//   1 hand : 63 numbers
//   2 hands: left-on-screen hand (63) + right-on-screen hand (63) + where the hands are
//            relative to each other (2). Sorting by screen position keeps the order stable.
function buildFeatures(handsLm, W, H) {
  const sorted = [...handsLm].sort((a, b) => a[0].x - b[0].x);
  if (sorted.length === 1) return toFeatures(sorted[0]);
  const [a, b] = sorted;
  const size = (lm) => Math.hypot((lm[0].x - lm[9].x) * W, (lm[0].y - lm[9].y) * H) || 1;
  const hs = (size(a) + size(b)) / 2;
  const dx = ((b[0].x - a[0].x) * W) / hs;
  const dy = ((b[0].y - a[0].y) * H) / hs;
  return [
    ...toFeatures(a),
    ...toFeatures(b),
    Math.round(clamp(dx, -4, 4) * 500) / 1000,
    Math.round(clamp(dy, -4, 4) * 500) / 1000,
  ];
}

// Mirror the hands left<->right, so a sign taught with one hand also works with the other.
const mirrorHands = (handsLm) => handsLm.map((lm) => lm.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z })));

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

// k-Nearest-Neighbours over samples that have the same number of hands.
function predict(features) {
  const n = features.length === ONE_HAND_LEN ? 1 : 2;
  const pool = samples.filter((s) => s.n === n);
  if (pool.length === 0) return null;
  const norm = Math.sqrt(n);   // so one- and two-hand distances use the same slider
  const scored = pool.map((s) => ({ label: s.label, d: distance(features, s.f) / norm }));
  scored.sort((a, b) => a.d - b.d);
  const top = scored.slice(0, Math.min(K, scored.length));
  const votes = {};
  for (const t of top) votes[t.label] = (votes[t.label] || 0) + 1;
  let best = null, bestVotes = 0;
  for (const label in votes) {
    if (votes[label] > bestVotes) { best = label; bestVotes = votes[label]; }
  }
  return { label: best, confidence: bestVotes / top.length, nearest: top[0].d };
}

// ---------- Translation + speech ----------
const VOCAB_BY_LABEL = Object.fromEntries(ISL.VOCAB.map((v) => [v.label, v]));
function langCode() { return $("langSelect").value.split("-")[0]; }
function show(label) {                        // text for a word in the chosen language
  const v = VOCAB_BY_LABEL[label];
  return (v && v.tr && v.tr[langCode()]) || label;
}
function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = $("langSelect").value;
  window.speechSynthesis.speak(u);
}

// ---------- Sentence UI ----------
function renderSentence() {
  sentenceEl.textContent = words.length ? words.map(show).join(" ") : "(nothing yet)";
}
function addWord(w) {
  words.push(w);
  renderSentence();
  if ($("autoSpeak").checked) speak(show(w));
  if (target && w === target) {
    practiceEl.innerHTML = `✅ Correct! You signed <b>${w}</b>.`;
    practiceEl.className = "practice ok";
    target = null;
    renderGuide();
  }
}

// ---------- Saved signs UI ----------
function renderSigns() {
  signListEl.innerHTML = "";
  const info = {};
  for (const s of samples) {
    info[s.label] = info[s.label] || { count: 0, two: false };
    info[s.label].count++;
    if (s.n === 2) info[s.label].two = true;
  }
  const labels = Object.keys(info);
  if (!labels.length) {
    signListEl.textContent = "No custom signs saved yet (built-in ISL signs are always available).";
    return;
  }
  for (const label of labels) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${info[label].two ? "🙌 " : ""}${label} (${info[label].count})`;
    const del = document.createElement("button");
    del.textContent = "✕";
    del.title = "Delete this sign";
    del.onclick = () => {
      samples = samples.filter((s) => s.label !== label);
      saveSamples();
      renderSigns();
    };
    chip.appendChild(del);
    signListEl.appendChild(chip);
  }
}

// ---------- ISL guide + practice ----------
function renderGuide() {
  const box = $("guideList");
  box.innerHTML = "";
  for (const v of ISL.VOCAB) {
    const item = document.createElement("div");
    item.className = "guide-item" + (target === v.label ? " active" : "");
    const tags = `${v.hands === 2 ? "🙌 two hands" : "✋ one hand"}${v.motion ? " · motion" : ""}`;
    item.innerHTML =
      `<div class="g-top"><span class="g-emoji">${v.emoji}</span><b>${v.label}</b></div>` +
      `<div class="g-tags">${tags}</div>` +
      `<div class="g-how">${v.how}</div>` +
      `<div class="g-tr">हिं ${v.tr.hi} · മല ${v.tr.ml} · தமி ${v.tr.ta}</div>`;
    const btn = document.createElement("button");
    btn.textContent = target === v.label ? "Practising…" : "Practice";
    btn.onclick = () => {
      target = v.label;
      practiceEl.innerHTML = `🎯 Now sign: <b>${v.label}</b> — ${v.how}`;
      practiceEl.className = "practice";
      renderGuide();
    };
    item.appendChild(btn);
    box.appendChild(item);
  }
}
function renderSuggested() {
  const box = $("suggestList");
  for (const w of ISL.SUGGESTED) {
    const b = document.createElement("button");
    b.className = "tag";
    b.textContent = w;
    b.onclick = () => { $("labelInput").value = w; $("labelInput").focus(); };
    box.appendChild(b);
  }
}

// ---------- MediaPipe results (runs on every camera frame) ----------
const HAND_COLORS = [
  { line: "#4f46e5", fill: "#22c55e" },
  { line: "#f59e0b", fill: "#ef4444" },
];

function onResults(results) {
  const W = results.image.width, H = results.image.height;
  canvas.width = W;
  canvas.height = H;
  ctx.save();
  ctx.clearRect(0, 0, W, H);

  const hands = results.multiHandLandmarks || [];
  if (hands.length === 0) {
    ctx.restore();
    liveLabel.textContent = "–";
    handsLabel.textContent = "0";
    candidate = null;
    candidateCount = 0;
    lastAdded = null;           // hands lowered -> the same sign may be used again
    recognizer.reset();
    if (!recording) setStatus("Show one or both hands to the camera");
    return;
  }

  // Draw every detected hand, each in its own colour.
  [...hands].sort((a, b) => a[0].x - b[0].x).forEach((lm, i) => {
    const c = HAND_COLORS[i % HAND_COLORS.length];
    drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: c.line, lineWidth: 4 });
    drawLandmarks(ctx, lm, { color: "#ffffff", fillColor: c.fill, lineWidth: 1, radius: 4 });
  });
  ctx.restore();
  handsLabel.textContent = String(hands.length);

  recognizer.push(hands, W, H);
  const features = buildFeatures(hands, W, H);
  const n = hands.length;

  // --- Recording mode: save frames as training samples ---
  if (recording) {
    if (recording.n === null) recording.n = n;           // lock to the first frame's hand count
    if (n !== recording.n) {
      setStatus(`Keep ${recording.n} hand${recording.n > 1 ? "s" : ""} in view while recording`);
      return;
    }
    samples.push({ label: recording.label, f: features, n });
    if ($("mirrorAug").checked) {
      samples.push({ label: recording.label, f: buildFeatures(mirrorHands(hands), W, H), n });
    }
    recording.count++;
    progressBar.style.width = `${(recording.count / RECORD_FRAMES) * 100}%`;
    setStatus(`Recording "${recording.label}" (${n} hand${n > 1 ? "s" : ""})… ${recording.count}/${RECORD_FRAMES}`);
    if (recording.count >= RECORD_FRAMES) finishRecording();
    return;
  }

  // --- Translate mode ---
  // 1) taught signs
  let result = null;
  const p = predict(features);
  if (p && p.confidence >= 0.6 && p.nearest <= maxDistance) {
    result = { label: p.label, motion: false, source: "taught" };
  }
  // 2) built-in ISL vocabulary
  if (!result && $("useBuiltin").checked) {
    const b = recognizer.recognize();
    if (b) result = { label: b.label, motion: !!b.motion, source: "built-in" };
  }

  if (!result) {
    liveLabel.textContent = "not sure";
    candidate = null;
    candidateCount = 0;
    setStatus(`${n} hand${n > 1 ? "s" : ""} detected, sign not recognised`);
    return;
  }

  liveLabel.textContent = show(result.label);
  if (result.label === candidate) {
    candidateCount++;
  } else {
    candidate = result.label;
    candidateCount = 1;
  }

  const need = result.motion ? Math.min(holdFrames, 6) : holdFrames;   // motion signs already took time
  if (candidateCount >= need && candidate !== lastAdded) {
    addWord(candidate);
    lastAdded = candidate;
  }
  setStatus(`Recognising "${result.label}"`);
}

function startRecording() {
  const label = $("labelInput").value.trim();
  if (!label) { alert("Type the word for this sign first."); return; }
  if (!cameraRunning) { alert("Start the camera first."); return; }
  recording = { label, count: 0, n: null };
  progressBar.style.width = "0%";
  progressWrap.classList.remove("hidden");
  $("recordBtn").disabled = true;
}

function finishRecording() {
  recording = null;
  saveSamples();
  renderSigns();
  progressWrap.classList.add("hidden");
  $("recordBtn").disabled = false;
  setStatus("Saved! Record again for more accuracy, or try the sign.");
}

// ---------- Camera ----------
let cameraRunning = false;
let camera = null;
let hands = null;

async function startCamera() {
  if (cameraRunning) return;
  if (typeof Hands === "undefined") {
    setStatus("MediaPipe failed to load. Check your internet connection.");
    return;
  }
  $("startBtn").disabled = true;
  setStatus("Loading hand model…");

  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
  });
  hands.setOptions({
    maxNumHands: 2,               // <- both hands at the same time
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
  });
  hands.onResults(onResults);

  camera = new Camera(videoEl, {
    onFrame: async () => { await hands.send({ image: videoEl }); },
    width: 640,
    height: 480,
  });

  try {
    await camera.start();
    cameraRunning = true;
    $("startBtn").textContent = "Camera running";
    setStatus("Show one or both hands to the camera");
  } catch (err) {
    console.error(err);
    $("startBtn").disabled = false;
    setStatus("Could not open the camera. Allow camera access and use https or localhost.");
  }
}

// ---------- Wire up buttons ----------
$("startBtn").onclick = startCamera;
$("recordBtn").onclick = startRecording;
$("speakBtn").onclick = () => speak(words.map(show).join(" "));
$("undoBtn").onclick = () => { words.pop(); renderSentence(); };
$("clearBtn").onclick = () => { words = []; renderSentence(); };
$("langSelect").onchange = renderSentence;

$("holdRange").oninput = (e) => {
  holdFrames = Number(e.target.value);
  $("holdVal").textContent = holdFrames;
};
$("strictRange").oninput = (e) => {
  maxDistance = Number(e.target.value) / 10;
  $("strictVal").textContent = maxDistance.toFixed(1);
};
$("stopPracticeBtn").onclick = () => {
  target = null;
  practiceEl.textContent = "Pick a sign below and press Practice.";
  practiceEl.className = "practice";
  renderGuide();
};

$("exportBtn").onclick = () => {
  const blob = new Blob([JSON.stringify(samples)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "my-signs.json";
  a.click();
  URL.revokeObjectURL(a.href);
};
$("importBtn").onclick = () => $("importFile").click();
$("importFile").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    const ok = (s) => s && s.label && Array.isArray(s.f) && (s.f.length === ONE_HAND_LEN || s.f.length === TWO_HAND_LEN);
    if (!Array.isArray(data) || !data.every(ok)) throw new Error("bad format");
    samples = samples.concat(data.map((s) => ({ label: s.label, f: s.f, n: s.f.length === ONE_HAND_LEN ? 1 : 2 })));
    saveSamples();
    renderSigns();
  } catch (err) {
    alert("That file is not a valid signs export.");
  }
  e.target.value = "";
};
$("wipeBtn").onclick = () => {
  if (confirm("Delete all your saved custom signs? (Built-in ISL signs are not affected.)")) {
    samples = [];
    saveSamples();
    renderSigns();
  }
};

// ---------- Init ----------
renderSentence();
renderSigns();
renderGuide();
renderSuggested();
