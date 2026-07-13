# Sign Language Translator

BIM (Bahasa Isyarat Malaysia) sign → text/speech translator. Phase 1: recognizes
isolated signs (the 10 words in the [MyWSL](https://data.mendeley.com/datasets/zvk55p7ktd)
dataset) from a live webcam feed and speaks/displays the result in the browser.

## How it works

Browser webcam → WebSocket → backend decodes frame (OpenCV) + extracts hand
landmarks (MediaPipe) → landmark sequence buffered until a sign-boundary
heuristic detects motion-then-stillness → PyTorch GRU sequence classifier →
predicted label sent back over WebSocket → frontend shows subtitle + speaks it
via Web Speech API.

## Structure

- `frontend/` — React + Tailwind webcam UI, WebSocket client, subtitle/status
  display, speech output (Vite, Vitest)
- `backend/app/` — FastAPI WebSocket server, landmark extraction, sign
  boundary segmentation, inference
- `backend/train/` — offline training pipeline (dataset loader + training loop),
  shares the landmark extraction module with the server so train/inference
  features never drift; `prepare_mywsl.py` reorganizes the raw MyWSL dataset
  into the loader's expected folder layout
- `docs/superpowers/` — design spec and implementation plan

## Running locally

### Backend

```
cd backend
pip install -r requirements.txt
uvicorn app.main:create_app --factory --reload
```

Needs a trained checkpoint at `backend/checkpoints/model.pt` (see
`backend/train/train.py`) and the MediaPipe hand landmarker model at
`backend/models/hand_landmarker.task`.

`mediapipe` needs system OpenGL libraries just to import, even for CPU-only
inference — without them, `import mediapipe` fails with
`OSError: libGLESv2.so.2: cannot open shared object file`. On Debian/Ubuntu:

```
sudo apt-get install -y libgles2 libegl1 libgl1
```

### Dataset

Training data is [MyWSL2023](https://data.mendeley.com/datasets/zvk55p7ktd) —
Malaysian Words Sign Language Dataset (Johari et al., 2023), Mendeley Data,
CC BY 4.0. Download and extract `MyWSL2023 CROP DATA.zip` from that page, then
reorganize it into the `<root>/<class>/<sample>/*.jpg` layout `SignDataset`
expects:

```
cd backend
python -m train.prepare_mywsl "/path/to/MyWSL2023 CROP DATA" /path/to/dataset_root
python -m train.train /path/to/dataset_root backend/checkpoints/model.pt --epochs 30
```

### Frontend

```
cd frontend
npm install
npm run dev
```

Set `VITE_WS_URL` if the backend isn't at `ws://localhost:8000/ws/sign-stream`.

## Deployment

- **Backend**: `backend/Dockerfile` + `render.yaml` deploy it as a Render web
  service (free tier, no credit card) — connect the repo at
  https://dashboard.render.com/blueprints and Render builds from the
  Dockerfile. `backend/checkpoints/model.pt` and
  `backend/models/hand_landmarker.task` must exist locally before building
  the image (both are gitignored, so they don't come from the repo). The
  free plan spins the service down after 15 min idle, with a cold start on
  the next request.
- **Frontend**: any static host (Vercel, Netlify, etc.) building
  `frontend/` with Vite. Set `VITE_WS_URL` to the backend's `wss://` URL —
  browsers refuse a plain `ws://` connection from an HTTPS page.

## Tests

```
cd backend && pytest
cd frontend && npm test
```

## Scope

Phase 1 only: isolated sign recognition, local dev/demo. Continuous
sentence/grammar recognition and the reverse (text/speech → sign) direction
are out of scope — see `docs/superpowers/specs/`.
