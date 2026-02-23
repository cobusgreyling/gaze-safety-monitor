# Gaze Safety Monitor

A Gradio web app for detecting gaze distraction events using the NVIDIA Maxine Eye Contact NIM.

The approach: send a video through the NVIDIA Eye Contact API, compare the original against the gaze-redirected output frame by frame. Where frames differ significantly, the subject was looking away from the camera.

## Safety Use Cases

- Driver alertness monitoring
- Machine operator attention tracking
- Workplace safety compliance

## Features

- **Full API mode** — Sends video to NVIDIA Maxine Eye Contact NIM for gaze redirection, then compares frames to detect distraction
- **Demo mode** — Generates synthetic gaze deviation scores to demonstrate the annotation and reporting pipeline without an API key
- Annotated video output with real-time HUD overlay (status banner, score bar, colour-coded border)
- Gaze deviation timeline chart
- Safety report with attentiveness percentage, distraction event table, and pass/warning/fail verdict

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Gradio Web UI

```bash
python gaze_safety_app.py
# Opens at http://localhost:7860
```

Upload a video, optionally enter your NGC API key, and click **Process (Full API)** or **Process (Demo Mode)**.

### CLI

```bash
export NGC_API_KEY="nvapi-..."
python gaze_safety_monitor.py path/to/video.mp4
```

## Requirements

- Python 3.10+
- NVIDIA NGC API key (for full pipeline mode)
- Dependencies listed in `requirements.txt`

## NVIDIA NIM Client

The `nim-clients-eye-contact/` directory contains the NVIDIA Maxine Eye Contact gRPC client interfaces and protobuf definitions from the [NVIDIA NIM Clients](https://github.com/NVIDIA/nim-clients) repository.
