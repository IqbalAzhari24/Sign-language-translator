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

## Tests

```
cd backend && pytest
cd frontend && npm test
```

## Scope

Phase 1 only: isolated sign recognition, local dev/demo. Continuous
sentence/grammar recognition and the reverse (text/speech → sign) direction
are out of scope — see `docs/superpowers/specs/`.
