# BIM Sign → Text/Speech Translator — Phase 1 Design

Date: 2026-07-08
Status: Approved for planning

## Overview & Scope

Sign language translator for Bahasa Isyarat Malaysia (BIM). This spec covers **Phase 1 only**:
BIM sign → text/speech. The reverse direction (Text/Speech → Sign, e.g. avatar generation) is
out of scope and will get its own spec later once Phase 1 is proven.

- **Input:** live webcam, captured in-browser
- **Recognition scope:** isolated signs (alphabet + digits) — not continuous sentence/grammar recognition
- **Motion handling:** some BIM signs involve movement (not just static hand shapes), so recognition
  uses a short sequence of frames, not a single frame
- **Output:** recognized text displayed live + spoken aloud via browser TTS
- **Platform:** web app — React/Tailwind frontend (built from scratch, using 21st.dev component
  generation), Python FastAPI backend
- **Dataset:** public BIM alphabet/digit dataset. Exact source is decided at implementation time;
  the data loader is built dataset-agnostic so swapping the source doesn't require pipeline changes.
- **Explicitly out of scope for Phase 1:** continuous sentence/grammar recognition, Text→Sign
  direction, public deployment/hosting (this phase targets local dev/demo only)

## Architecture & Components

**Data flow:** browser webcam → WebSocket → backend (OpenCV decode + MediaPipe landmark
extraction) → PyTorch sequence classifier → predicted text sent back over WebSocket → browser
displays subtitle + speaks it via Web Speech API.

### 1. Frontend (React + Tailwind, 21st.dev-generated components)
- Webcam capture view (`getUserMedia`)
- WebSocket client streaming frames to backend at a fixed reduced rate (~10–15 fps — enough to
  capture gesture motion without excess bandwidth/latency)
- Live subtitle display for recognized text
- Web Speech API call triggered on each new recognized word
- Status indicator: connected / no hand detected / recognizing / unsure

### 2. Backend (Python, FastAPI + WebSocket endpoint)
- Receives frame bytes, decodes via OpenCV
- MediaPipe Hands extracts 21 landmark (x, y, z) points per detected hand (supports up to 2 hands,
  since BIM has one- and two-handed signs)
- Maintains a rolling buffer (~20–30 frames) of landmark sequences per connection
- Feeds the buffer into a PyTorch sequence model (small GRU/LSTM or 1D-CNN over the landmark
  sequence) once a sign boundary is detected
- Returns predicted label + confidence over WebSocket

### 3. Model training pipeline (offline, separate from serving)
- Dataset loader, abstracted behind a swappable interface
- Landmark extraction code lives in a **shared module** imported by both the training script and
  the FastAPI server, so train-time and inference-time feature extraction never drift apart
- Standard train/val split, PyTorch training loop, checkpoint export (`.pt`)
- Backend loads the checkpoint at startup; fails fast with a clear error if missing

### 4. Sign-boundary / gesture segmentation
- Continuous webcam stream needs a heuristic to detect when a sign "starts/ends" — e.g.
  hand-presence + motion-stillness threshold, or sliding window gated by confidence threshold
- Prevents reacting to hand entering/leaving frame as spurious signs

## Error Handling & Edge Cases

- **No hand detected:** MediaPipe returns empty → backend skips inference; frontend shows
  "no hand detected", does not spam predictions
- **WebSocket disconnect:** frontend auto-reconnects with backoff; backend clears that
  connection's frame buffer on disconnect
- **Low-confidence prediction:** below threshold (e.g. <60%) → suppress output, show "unsure"
  rather than a wrong guess
- **Multiple hands / variable hand count:** model input handles up to 2 hands, zero-padded if
  only one hand present (matches MediaPipe Hands' native multi-hand output)
- **Browser TTS unsupported/blocked:** fall back to text-only; some browsers block autoplay
  audio without a user gesture, so first speech call may need a user-initiated "enable audio" click
- **Missing model checkpoint at backend startup:** fail fast with a clear error log; never
  silently serve untrained/random weights

## Testing Strategy

- **Landmark extraction module:** unit tests against sample images/videos of known hand poses,
  assert expected landmark count/shape
- **Data pipeline:** test dataset loader against a small fixture subset (few samples/class) to
  catch format/label mismatches early
- **Model:** train on the fixture subset, assert loss decreases and output shape is correct
  (no accuracy target at this stage — real training happens later with full data)
- **Segmentation heuristic:** unit test with synthetic landmark sequences (still hand → no
  trigger; moving-then-still → trigger)
- **End-to-end:** manual dev-server check — webcam → live subtitle appears. Full webcam-in-browser
  flow isn't easily automatable, so this stays manual/exploratory per session, not CI
