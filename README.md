# Sign to Voice

A website that reads Indian Sign Language (ISL) through your laptop webcam, shows the result as text on screen, and speaks it aloud. Works with **one or both hands at the same time**.

- **Hand tracking:** [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) finds 21 landmark points on each hand, **up to 2 hands at once**.
- **Built-in ISL vocabulary:** 20 basic signs work with no training (see below), including motion signs and two-hand signs.
- **Recognition of your own signs:** a small k-Nearest-Neighbours classifier compares your hand pose(s) against signs you have taught it, for one-hand and two-hand signs.
- **Voice:** the browser's built-in Web Speech API (English, Hindi, Malayalam, Tamil, and more, depending on the voices installed on your device).
- **Privacy:** everything runs in the browser. No video leaves your computer.

## How to use

1. Click **Start camera** and allow access.
2. Under **Teach a sign**, type a word (e.g. `Hello`), make the sign, press **Record**. Repeat 2-3 times with small variations. Add as many signs as you like.
3. Make a sign and hold it steadily. The word appears in **Translation** and is spoken.
4. Use **Export signs** to back up your data, and **Import signs** to load it on another device.

Start with 5-10 signs. Once those work well, add more.

## Built-in basic ISL signs

Open the **Basic ISL sign guide & practice** card in the app: every sign shows how to perform it, and **Practice** gives a ✅ when the camera recognises it. Spoken output is available in English, Hindi, Malayalam and Tamil for these words.

| Type | Signs |
|------|-------|
| One hand, still | 1, 2, 3, 4, 5, Good (thumbs up), Bad (thumbs down), OK, I love you, Phone, Food |
| One hand, motion | Hello (wave), Yes (fist nod), No (finger wag) |
| Two hands, still | Namaste, Home, Together, Big, Ten |
| Two hands, motion | Clap |

These gestures are simplified so a webcam prototype can detect them reliably. For anything you present as "official ISL", confirm the exact sign in the ISLRTC dictionary and record it with **Teach a sign** (which supports two hands too).

## Teaching more signs (one or two hands)

1. Type the word (or click one of the suggested words).
2. Make the sign with one or both hands and press **Record**. The number of hands in view is detected automatically and must stay the same during the recording.
3. Leave **Also learn the mirrored version** ticked so the sign works with either hand (or either side for two-hand signs).
4. Taught signs take priority over built-in ones. Untick **Use built-in ISL signs** to test only your own.

## Run locally

The camera only works on `https` or `localhost`, so don't just double-click `index.html`. From this folder:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000> in Chrome or Edge.

## Put it online with GitHub Pages

1. Create a new GitHub repository and upload all files from this folder.
2. Go to **Settings -> Pages**, set the source to the `main` branch and root folder, and save.
3. After a minute your site is live at `https://<your-username>.github.io/<repo-name>/` (https, so the camera works).

## Project files

| File | Purpose |
|------|---------|
| `index.html` | Page layout |
| `style.css` | Styling |
| `app.js` | Camera, 2-hand tracking, features, k-NN classifier, speech, practice mode |
| `signs.js` | Built-in ISL vocabulary, finger/pose geometry, motion + two-hand rules |

## Tips for better accuracy

- Good, even lighting and a plain background.
- Record each sign several times, at slightly different distances and angles.
- For two-hand signs keep both hands fully inside the camera frame.
- Make signs look clearly different from each other.
- Adjust **Hold time** and **Strictness** sliders in the Translation panel.

## Limitations and next steps

- Motion signs are limited to simple repeated movements (wave, nod, wag, clap). Signs that trace paths or touch the face/body (e.g. Thank you, Sorry, Water) are not built in because the app tracks hands only. You can still teach a hand-shape version.
- Two hands pressed palm-to-palm (Namaste) can confuse the tracker; angle your hands slightly toward the camera and keep them well lit. If a sign is flaky, adjust **Hold time** / **Strictness** or re-record it.
- Thresholds were tuned on geometry, not on a large dataset, so test with your own hands and lighting.
- Ideas: motion signs using sequences of frames (LSTM), a pre-trained ISL dataset, body/face landmarks (MediaPipe Holistic), sentence building.
