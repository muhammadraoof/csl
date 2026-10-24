// ============================================================
// Built-in basic Indian Sign Language (ISL) vocabulary + recogniser.
//
// Works on MediaPipe hand landmarks (21 points per hand, up to 2 hands).
// It recognises three kinds of signs without any training:
//   1. static one-hand poses   (numbers 1-5, Good, Bad, OK, I love you ...)
//   2. motion one-hand actions (Yes = fist nod, No = finger wag, Hello = wave)
//   3. two-hand signs          (Namaste, Clap, Home, Big, Together, Ten)
//
// Anything not covered here can still be taught by the user (k-NN in app.js),
// and taught signs may also use one or two hands.
// ============================================================
(function (root) {
  "use strict";

  const HISTORY_MS = 1400;   // how much history is kept
  const MOTION_MS = 1200;    // window used to detect waving / nodding / clapping
  const STILL_MS = 500;      // window used to decide "the hand is held still"

  // ---------------------------------------------------------
  // Vocabulary shown in the guide, with spoken translations.
  // "how" is a plain-language description of the gesture used by this prototype.
  // ---------------------------------------------------------
  const VOCAB = [
    { label: "Hello", emoji: "👋", hands: 1, motion: true,
      how: "Open hand, fingers up, wave left-right.",
      tr: { hi: "हैलो", ml: "ഹലോ", ta: "வணக்கம்" } },
    { label: "Namaste", emoji: "🙏", hands: 2, motion: false,
      how: "Both palms pressed together at chest height, fingers up. Hold still.",
      tr: { hi: "नमस्ते", ml: "നമസ്കാരം", ta: "வணக்கம்" } },
    { label: "Yes", emoji: "✊", hands: 1, motion: true,
      how: "Make a fist and nod it up and down.",
      tr: { hi: "हाँ", ml: "അതെ", ta: "ஆம்" } },
    { label: "No", emoji: "☝️", hands: 1, motion: true,
      how: "Raise only the index finger and wag it left-right.",
      tr: { hi: "नहीं", ml: "ഇല്ല", ta: "இல்லை" } },
    { label: "Good", emoji: "👍", hands: 1, motion: false,
      how: "Thumbs up (fist, thumb pointing up).",
      tr: { hi: "अच्छा", ml: "നല്ലത്", ta: "நல்லது" } },
    { label: "Bad", emoji: "👎", hands: 1, motion: false,
      how: "Thumbs down (fist, thumb pointing down).",
      tr: { hi: "बुरा", ml: "മോശം", ta: "மோசம்" } },
    { label: "OK", emoji: "👌", hands: 1, motion: false,
      how: "Touch thumb tip to index tip, other three fingers up.",
      tr: { hi: "ठीक है", ml: "ശരി", ta: "சரி" } },
    { label: "I love you", emoji: "🤟", hands: 1, motion: false,
      how: "Thumb, index finger and little finger up; middle and ring folded.",
      tr: { hi: "मैं तुमसे प्यार करता हूँ", ml: "ഞാൻ നിന്നെ സ്നേഹിക്കുന്നു", ta: "நான் உன்னை நேசிக்கிறேன்" } },
    { label: "Phone", emoji: "🤙", hands: 1, motion: false,
      how: "Thumb and little finger out, other fingers folded.",
      tr: { hi: "फ़ोन", ml: "ഫോൺ", ta: "தொலைபேசி" } },
    { label: "Food", emoji: "🤌", hands: 1, motion: false,
      how: "Bring all fingertips together with the thumb (flat 'O' shape).",
      tr: { hi: "खाना", ml: "ഭക്ഷണം", ta: "உணவு" } },
    { label: "1", emoji: "1️⃣", hands: 1, motion: false, how: "Index finger up.",
      tr: { hi: "एक", ml: "ഒന്ന്", ta: "ஒன்று" } },
    { label: "2", emoji: "2️⃣", hands: 1, motion: false, how: "Index + middle finger up.",
      tr: { hi: "दो", ml: "രണ്ട്", ta: "இரண்டு" } },
    { label: "3", emoji: "3️⃣", hands: 1, motion: false, how: "Index, middle, ring finger up.",
      tr: { hi: "तीन", ml: "മൂന്ന്", ta: "மூன்று" } },
    { label: "4", emoji: "4️⃣", hands: 1, motion: false, how: "Four fingers up, thumb folded in.",
      tr: { hi: "चार", ml: "നാല്", ta: "நான்கு" } },
    { label: "5", emoji: "5️⃣", hands: 1, motion: false, how: "Open hand, all five fingers spread. Hold still.",
      tr: { hi: "पाँच", ml: "അഞ്ച്", ta: "ஐந்து" } },
    { label: "Clap", emoji: "👏", hands: 2, motion: true,
      how: "Both open hands: bring them together and apart a few times.",
      tr: { hi: "ताली", ml: "കൈയടി", ta: "கைதட்டு" } },
    { label: "Home", emoji: "🏠", hands: 2, motion: false,
      how: "Both open hands, fingertips touching at the top like a roof, wrists apart.",
      tr: { hi: "घर", ml: "വീട്", ta: "வீடு" } },
    { label: "Together", emoji: "🤝", hands: 2, motion: false,
      how: "Both hands with only the index finger up; touch the two fingertips.",
      tr: { hi: "साथ", ml: "ഒരുമിച്ച്", ta: "சேர்ந்து" } },
    { label: "Big", emoji: "↔️", hands: 2, motion: false,
      how: "Both open hands raised, held wide apart. Hold still.",
      tr: { hi: "बड़ा", ml: "വലുത്", ta: "பெரியது" } },
    { label: "Ten", emoji: "🔟", hands: 2, motion: false,
      how: "Both open hands raised side by side (5 + 5), a little apart. Hold still.",
      tr: { hi: "दस", ml: "പത്ത്", ta: "பத்து" } },
  ];

  // Words people commonly want to teach next (click to fill the label box).
  // Check the exact sign in the official ISLRTC dictionary before recording.
  const SUGGESTED = [
    "Thank you", "Please", "Sorry", "Water", "Help", "Mother", "Father",
    "Friend", "Name", "Eat", "Drink", "Go", "Come", "Stop", "Love", "Good morning",
  ];

  // ---------------------------------------------------------
  // Geometry helpers
  // ---------------------------------------------------------
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));

  // Turn normalised landmarks into pixel-scale points and work out finger states.
  function analyseHand(lm, W, H) {
    const P = lm.map((p) => ({ x: p.x * W, y: p.y * H, z: (p.z || 0) * W }));
    const w = P[0];
    const hs = Math.max(dist(w, P[9]), 1e-6);            // palm length = size unit
    const far = (tip, pip, t) => dist(w, P[tip]) / hs > t && dist(w, P[tip]) > dist(w, P[pip]) * 1.05;

    const ext = {
      I: far(8, 6, 1.5),
      M: far(12, 10, 1.55),
      R: far(16, 14, 1.5),
      P: far(20, 18, 1.35),
    };

    // Thumb: straight (not bent) and not tucked against the palm.
    const seg = dist(P[2], P[3]) + dist(P[3], P[4]);
    const straight = dist(P[2], P[4]) / (seg || 1);
    const thumbOut = straight > 0.82 && dist(P[4], P[5]) / hs > 0.4;

    // Thumb direction in the image (y grows downward).
    const tv = { x: P[4].x - P[2].x, y: P[4].y - P[2].y };
    const tl = Math.hypot(tv.x, tv.y) || 1;
    // Thumbs up/down: straight thumb pointing mostly vertically (may hug the fist).
    const thumbVert = straight > 0.8 && Math.abs(tv.y) / hs > 0.45;
    const thumbUp = thumbVert && -tv.y / tl > 0.65;
    const thumbDown = thumbVert && tv.y / tl > 0.65;

    const fingers = [ext.I, ext.M, ext.R, ext.P].filter(Boolean).length;

    const pinch = dist(P[4], P[8]) / hs < 0.3;
    const tipsNearThumb = [8, 12, 16, 20].every((i) => dist(P[4], P[i]) / hs < 0.5);
    const flatO = tipsNearThumb && dist(P[8], P[20]) / hs < 0.7 && dist(w, P[12]) / hs > 1.3;

    const palm = {
      x: (P[0].x + P[5].x + P[9].x + P[13].x + P[17].x) / 5,
      y: (P[0].y + P[5].y + P[9].y + P[13].y + P[17].y) / 5,
    };
    const pointsUp = (w.y - P[12].y) / hs > 0.9;

    return { P, hs, palm, ext, fingers, thumbOut, thumbUp, thumbDown, pinch, flatO, pointsUp };
  }

  // Count back-and-forth reversals in a series (ignores wiggles smaller than minAmp).
  function reversals(vals, minAmp) {
    if (vals.length < 3) return 0;
    let dir = 0, lo = vals[0], hi = vals[0], rev = 0;
    for (const v of vals) {
      if (dir === 0) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        if (v - lo > minAmp) { dir = 1; hi = v; }
        else if (hi - v > minAmp) { dir = -1; lo = v; }
      } else if (dir === 1) {
        if (v > hi) hi = v;
        else if (hi - v > minAmp) { rev++; dir = -1; lo = v; }
      } else {
        if (v < lo) lo = v;
        else if (v - lo > minAmp) { rev++; dir = 1; hi = v; }
      }
    }
    return rev;
  }

  // ---------------------------------------------------------
  // Recogniser: feed it every frame, ask it what it sees.
  // ---------------------------------------------------------
  function createRecognizer() {
    let hist = [];
    let cur = null;

    function push(handsLm, W, H, t) {
      t = t == null ? performance.now() : t;
      // Sort left-to-right on screen so "hand 0" is always the same physical side.
      const sorted = [...handsLm].sort((a, b) => a[0].x - b[0].x);
      const infos = sorted.map((lm) => analyseHand(lm, W, H));
      cur = { t, n: infos.length, hands: infos };
      hist.push(cur);
      while (hist.length && t - hist[0].t > HISTORY_MS) hist.shift();
      return infos;
    }

    function reset() { hist = []; cur = null; }

    // Trailing frames with the same hand count within `ms`.
    function recent(ms) {
      const out = [];
      for (let i = hist.length - 1; i >= 0; i--) {
        if (hist[i].n !== cur.n || cur.t - hist[i].t > ms) break;
        out.unshift(hist[i]);
      }
      return out;
    }

    const series = (frames, f) => frames.map(f);
    const range = (arr) => Math.max(...arr) - Math.min(...arr);

    // Is every hand held roughly still over the last STILL_MS?
    function isStill() {
      const fr = recent(STILL_MS);
      if (fr.length < 3) return true;
      for (let k = 0; k < cur.n; k++) {
        const xs = series(fr, (f) => f.hands[k].palm.x);
        const ys = series(fr, (f) => f.hands[k].palm.y);
        if (Math.hypot(range(xs), range(ys)) > 0.5 * cur.hands[k].hs) return false;
      }
      return true;
    }

    function waves(slot, tipIdx, minAmpHs, minRev) {
      const fr = recent(MOTION_MS);
      if (fr.length < 8) return false;
      const hs = cur.hands[slot].hs;
      return reversals(series(fr, (f) => f.hands[slot].P[tipIdx].x), minAmpHs * hs) >= minRev;
    }
    function nods(slot, minAmpHs, minRev) {
      const fr = recent(MOTION_MS);
      if (fr.length < 8) return false;
      const hs = cur.hands[slot].hs;
      return reversals(series(fr, (f) => f.hands[slot].palm.y), minAmpHs * hs) >= minRev;
    }

    // ---- one hand ----
    function oneHand(h) {
      const e = h.ext;
      const none = h.fingers === 0;

      // Motion signs first.
      if (e.I && e.M && e.R && e.P && waves(0, 12, 0.35, 2)) return { label: "Hello", motion: true };
      if (e.I && !e.M && !e.R && !e.P && waves(0, 8, 0.3, 2)) return { label: "No", motion: true };
      if (none && !h.thumbOut && nods(0, 0.25, 2)) return { label: "Yes", motion: true };

      // Static signs need a steady hand.
      if (!isStill()) return null;

      if (none && h.thumbUp) return { label: "Good" };
      if (none && h.thumbDown) return { label: "Bad" };
      if (h.thumbOut && e.I && !e.M && !e.R && e.P) return { label: "I love you" };
      if (h.thumbOut && !e.I && !e.M && !e.R && e.P) return { label: "Phone" };
      if (h.pinch && e.M && e.R && e.P) return { label: "OK" };
      if (h.flatO) return { label: "Food" };

      if (e.I && !e.M && !e.R && !e.P) return { label: "1" };
      if (e.I && e.M && !e.R && !e.P) return { label: "2" };
      if (e.I && e.M && e.R && !e.P) return { label: "3" };
      if (e.I && e.M && e.R && e.P) return { label: h.thumbOut ? "5" : "4" };
      return null;
    }

    // ---- two hands ----
    function twoHands(a, b) {
      const hs = (a.hs + b.hs) / 2;
      const palmD = Math.hypot(a.palm.x - b.palm.x, a.palm.y - b.palm.y) / hs;
      const tipD = dist(a.P[12], b.P[12]) / hs;
      const open = (h) => h.ext.I && h.ext.M && h.ext.R && h.ext.P;
      const bothOpen = open(a) && open(b);

      // Clap: open hands moving together and apart.
      if (bothOpen) {
        const fr = recent(MOTION_MS);
        if (fr.length >= 8) {
          const d = series(fr, (f) => Math.hypot(
            f.hands[0].palm.x - f.hands[1].palm.x, f.hands[0].palm.y - f.hands[1].palm.y));
          if (reversals(d, 0.6 * hs) >= 2 && Math.min(...d) / hs < 1.6) return { label: "Clap", motion: true };
        }
      }

      if (!isStill()) return null;

      const indexOnly = (h) => h.ext.I && !h.ext.M && !h.ext.R && !h.ext.P;
      if (indexOnly(a) && indexOnly(b) && dist(a.P[8], b.P[8]) / hs < 0.6) return { label: "Together" };

      if (bothOpen) {
        if (palmD < 1.0 && tipD < 0.6) return { label: "Namaste" };
        if (palmD >= 1.1 && tipD < 0.6) return { label: "Home" };
        if (a.pointsUp && b.pointsUp) {
          if (palmD > 3.2) return { label: "Big" };
          if (palmD >= 1.4 && tipD > 1.0) return { label: "Ten" };
        }
      }
      return null;
    }

    function recognize() {
      if (!cur || cur.n === 0) return null;
      if (cur.n === 1) return oneHand(cur.hands[0]);
      return twoHands(cur.hands[0], cur.hands[1]);
    }

    return { push, recognize, reset };
  }

  const api = { VOCAB, SUGGESTED, createRecognizer, analyseHand, reversals };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ISL = api;
})(typeof window !== "undefined" ? window : globalThis);
