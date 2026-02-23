"""
NVIDIA Eye Contact — Gaze Safety Monitor

Uses the NVIDIA Maxine Eye Contact NIM to detect distraction events
in video. The approach: send a video through the Eye Contact API,
compare the original against the gaze-redirected output frame by frame.
Where frames differ, the person was looking away from the camera.

Safety use cases:
  - Driver alertness monitoring
  - Machine operator attention tracking
  - Workplace safety compliance

Requires:
  - NVIDIA NGC API key (set NGC_API_KEY env var)
  - pip install grpcio grpcio-tools opencv-python numpy tqdm
"""

import os
import sys
import time
import pathlib
import tempfile
import json
from dataclasses import dataclass, field
from typing import Iterator

import grpc
import cv2
import numpy as np
from tqdm import tqdm

# Add paths for NVIDIA NIM client modules
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR / "nim-clients-eye-contact" / "scripts"))
sys.path.append(str(SCRIPT_DIR / "nim-clients-eye-contact" / "interfaces"))

import eyecontact_pb2
import eyecontact_pb2_grpc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_CHUNK_SIZE = 64 * 1024  # 64KB chunks for gRPC streaming
NVCF_TARGET = "grpc.nvcf.nvidia.com:443"
FUNCTION_ID = "b75dbca7-b5a4-458c-9275-6d2effeb432a"

# Safety thresholds (configurable)
DISTRACTION_THRESHOLD = 5.0    # Mean pixel difference to flag distraction
ALERT_THRESHOLD = 12.0         # Mean pixel difference for high-severity alert
MIN_EVENT_DURATION_SEC = 0.5   # Minimum duration to count as a distraction event


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class DistractionEvent:
    """A single distraction event detected in the video."""
    start_time: float       # seconds
    end_time: float         # seconds
    start_frame: int
    end_frame: int
    peak_score: float       # max pixel difference during event
    avg_score: float        # average pixel difference during event
    severity: str           # "LOW", "MODERATE", "HIGH"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SafetyReport:
    """Safety analysis report for a video."""
    video_path: str
    duration_sec: float
    total_frames: int
    fps: float
    events: list = field(default_factory=list)
    frame_scores: list = field(default_factory=list)

    @property
    def attentive_pct(self) -> float:
        distracted_time = sum(e.duration for e in self.events)
        if self.duration_sec == 0:
            return 100.0
        return max(0, (1 - distracted_time / self.duration_sec)) * 100

    @property
    def high_severity_count(self) -> int:
        return sum(1 for e in self.events if e.severity == "HIGH")

    def to_dict(self) -> dict:
        return {
            "video": self.video_path,
            "duration_sec": round(self.duration_sec, 2),
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "attentive_pct": round(self.attentive_pct, 1),
            "total_distraction_events": len(self.events),
            "high_severity_events": self.high_severity_count,
            "events": [
                {
                    "start_time": round(e.start_time, 2),
                    "end_time": round(e.end_time, 2),
                    "duration": round(e.duration, 2),
                    "severity": e.severity,
                    "peak_score": round(e.peak_score, 2),
                    "avg_score": round(e.avg_score, 2),
                }
                for e in self.events
            ],
        }


# ---------------------------------------------------------------------------
# Eye Contact API interaction
# ---------------------------------------------------------------------------
def create_safety_config() -> dict:
    """Create Eye Contact config tuned for safety monitoring.

    Uses tight thresholds so the model redirects gaze even for small
    deviations — maximising the difference signal we use for detection.
    """
    return {
        "temporal": 0xFFFFFFFF,
        "detect_closure": 1,          # detect blinks / closed eyes
        "eye_size_sensitivity": 3,
        "enable_lookaway": 0,         # no artificial look-away
        "lookaway_max_offset": 5,
        "lookaway_interval_min": 3,
        "lookaway_interval_range": 8,
        "gaze_pitch_threshold_low": 10.0,   # tight — redirect early
        "gaze_pitch_threshold_high": 35.0,  # wide — redirect up to max range
        "gaze_yaw_threshold_low": 10.0,
        "gaze_yaw_threshold_high": 35.0,
        "head_pitch_threshold_low": 10.0,
        "head_pitch_threshold_high": 35.0,
        "head_yaw_threshold_low": 10.0,
        "head_yaw_threshold_high": 35.0,
        "output_video_encoding": eyecontact_pb2.OutputVideoEncoding(
            lossy=eyecontact_pb2.LossyEncoding(
                bitrate=20_000_000,
                idr_interval=8,
            )
        ),
    }


def generate_requests(
    video_path: str, config_params: dict
) -> Iterator[eyecontact_pb2.RedirectGazeRequest]:
    """Stream config + video chunks to the Eye Contact service."""
    # Send config first
    yield eyecontact_pb2.RedirectGazeRequest(
        config=eyecontact_pb2.RedirectGazeConfig(**config_params)
    )

    # Stream video data in 64KB chunks
    file_size = os.path.getsize(video_path)
    with open(video_path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True,
                  desc="Uploading video", leave=False) as pbar:
            while True:
                chunk = f.read(DATA_CHUNK_SIZE)
                if not chunk:
                    break
                pbar.update(len(chunk))
                yield eyecontact_pb2.RedirectGazeRequest(video_file_data=chunk)


def call_eye_contact_api(video_path: str, output_path: str, api_key: str) -> None:
    """Send video to NVIDIA NIM Eye Contact API, save redirected output."""
    config = create_safety_config()
    metadata = (
        ("authorization", f"Bearer {api_key}"),
        ("function-id", FUNCTION_ID),
    )

    print(f"Connecting to NVIDIA NIM at {NVCF_TARGET} ...")
    channel = grpc.secure_channel(NVCF_TARGET, grpc.ssl_channel_credentials())

    try:
        stub = eyecontact_pb2_grpc.MaxineEyeContactServiceStub(channel)
        start = time.time()

        responses = stub.RedirectGaze(
            generate_requests(video_path, config),
            metadata=metadata,
        )

        # Skip the config echo response
        next(responses)

        # Write output video
        total_bytes = 0
        chunk_count = 0
        with open(output_path, "wb") as f:
            pbar = tqdm(desc="Receiving processed video", unit="chunks",
                        leave=False,
                        bar_format="{desc}: {n} chunks | {postfix}")
            for response in responses:
                if response.HasField("video_file_data"):
                    data = response.video_file_data
                    f.write(data)
                    chunk_count += 1
                    total_bytes += len(data)
                    pbar.set_postfix_str(f"{total_bytes / (1024*1024):.1f} MB")
                    pbar.update(1)
            pbar.close()

        elapsed = time.time() - start
        print(f"API processing complete: {total_bytes / (1024*1024):.1f} MB "
              f"in {elapsed:.1f}s")
    finally:
        channel.close()


# ---------------------------------------------------------------------------
# Frame comparison & distraction detection
# ---------------------------------------------------------------------------
def compute_frame_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute mean absolute pixel difference between two frames.

    Focuses on the upper-third of the frame (head/eye region) for more
    targeted distraction detection.
    """
    h = frame_a.shape[0]
    # Focus on upper third (head region)
    roi_a = frame_a[: h // 3, :, :]
    roi_b = frame_b[: h // 3, :, :]

    diff = cv2.absdiff(roi_a, roi_b)
    return float(np.mean(diff))


def extract_frame_scores(
    original_path: str, redirected_path: str
) -> tuple[list[float], float, int]:
    """Compare original vs redirected video frame-by-frame.

    Returns:
        (frame_scores, fps, total_frames)
    """
    cap_orig = cv2.VideoCapture(original_path)
    cap_redir = cv2.VideoCapture(redirected_path)

    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    total = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))

    scores = []
    pbar = tqdm(total=total, desc="Comparing frames", unit="frame", leave=False)

    while True:
        ret_o, frame_o = cap_orig.read()
        ret_r, frame_r = cap_redir.read()

        if not ret_o or not ret_r:
            break

        # Ensure same dimensions
        if frame_o.shape != frame_r.shape:
            frame_r = cv2.resize(frame_r, (frame_o.shape[1], frame_o.shape[0]))

        score = compute_frame_difference(frame_o, frame_r)
        scores.append(score)
        pbar.update(1)

    pbar.close()
    cap_orig.release()
    cap_redir.release()

    return scores, fps, len(scores)


def detect_events(
    scores: list[float],
    fps: float,
    threshold: float = DISTRACTION_THRESHOLD,
    alert_threshold: float = ALERT_THRESHOLD,
    min_duration: float = MIN_EVENT_DURATION_SEC,
) -> list[DistractionEvent]:
    """Detect distraction events from frame difference scores."""
    events = []
    in_event = False
    event_start = 0
    event_scores = []

    for i, score in enumerate(scores):
        if score >= threshold and not in_event:
            in_event = True
            event_start = i
            event_scores = [score]
        elif score >= threshold and in_event:
            event_scores.append(score)
        elif score < threshold and in_event:
            # End of event
            in_event = False
            duration = len(event_scores) / fps
            if duration >= min_duration:
                peak = max(event_scores)
                avg = sum(event_scores) / len(event_scores)
                severity = "HIGH" if peak >= alert_threshold else (
                    "MODERATE" if avg >= threshold * 1.5 else "LOW"
                )
                events.append(DistractionEvent(
                    start_time=event_start / fps,
                    end_time=(event_start + len(event_scores)) / fps,
                    start_frame=event_start,
                    end_frame=event_start + len(event_scores),
                    peak_score=peak,
                    avg_score=avg,
                    severity=severity,
                ))

    # Handle event that extends to end of video
    if in_event and len(event_scores) / fps >= min_duration:
        peak = max(event_scores)
        avg = sum(event_scores) / len(event_scores)
        severity = "HIGH" if peak >= alert_threshold else (
            "MODERATE" if avg >= threshold * 1.5 else "LOW"
        )
        events.append(DistractionEvent(
            start_time=event_start / fps,
            end_time=(event_start + len(event_scores)) / fps,
            start_frame=event_start,
            end_frame=event_start + len(event_scores),
            peak_score=peak,
            avg_score=avg,
            severity=severity,
        ))

    return events


# ---------------------------------------------------------------------------
# Annotated video output
# ---------------------------------------------------------------------------
# Colour scheme (BGR)
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 200, 255)
COLOR_RED = (0, 0, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_BG = (30, 30, 30)
COLOR_BAR_BG = (60, 60, 60)


def _severity_color(score: float, threshold: float, alert: float) -> tuple:
    if score >= alert:
        return COLOR_RED
    elif score >= threshold:
        return COLOR_YELLOW
    return COLOR_GREEN


def _draw_border(frame: np.ndarray, color: tuple, thickness: int = 6) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


def _draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    score: float,
    fps: float,
    events: list,
    threshold: float,
    alert: float,
    max_score: float,
) -> None:
    """Draw a heads-up display overlay on the frame."""
    h, w = frame.shape[:2]
    timestamp = frame_idx / fps if fps > 0 else 0

    # --- Status banner (top) ---
    banner_h = 48
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    color = _severity_color(score, threshold, alert)

    if score >= alert:
        status = "GAZE DIVERTED — HIGH"
    elif score >= threshold:
        status = "GAZE DIVERTED"
    else:
        status = "ATTENTIVE"

    # Status indicator dot
    cv2.circle(frame, (24, banner_h // 2), 8, color, -1)
    cv2.putText(frame, status, (42, banner_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Timestamp (right side)
    ts_text = f"{timestamp:.1f}s  |  Frame {frame_idx}"
    ts_size = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, ts_text, (w - ts_size[0] - 12, banner_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # --- Score bar (bottom) ---
    bar_y = h - 36
    bar_h = 20
    bar_margin = 12
    bar_w = w - 2 * bar_margin

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (bar_margin, bar_y), (bar_margin + bar_w, bar_y + bar_h),
                  COLOR_BAR_BG, -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

    # Fill bar proportional to score
    fill_max = max(max_score, alert * 1.5) if max_score > 0 else alert * 1.5
    fill_w = int(min(score / fill_max, 1.0) * bar_w)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_margin, bar_y),
                      (bar_margin + fill_w, bar_y + bar_h), color, -1)

    # Threshold markers
    for thr, lbl in [(threshold, "T"), (alert, "A")]:
        x = bar_margin + int(min(thr / fill_max, 1.0) * bar_w)
        cv2.line(frame, (x, bar_y), (x, bar_y + bar_h), COLOR_WHITE, 1)
        cv2.putText(frame, lbl, (x - 3, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_WHITE, 1)

    # Score value
    cv2.putText(frame, f"Score: {score:.1f}", (bar_margin, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)

    # --- Coloured border ---
    _draw_border(frame, color, thickness=4)


def generate_annotated_video(
    original_path: str,
    output_path: str,
    scores: list[float],
    events: list,
    fps: float,
    threshold: float = DISTRACTION_THRESHOLD,
    alert: float = ALERT_THRESHOLD,
) -> None:
    """Generate an annotated video with gaze safety overlays.

    Overlays include:
      - Coloured border: green (attentive), yellow (diverted), red (high)
      - Top banner with status text and timestamp
      - Bottom score bar with threshold markers
    """
    cap = cv2.VideoCapture(original_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    max_score = max(scores) if scores else 1.0
    pbar = tqdm(total=min(total, len(scores)), desc="Generating annotated video",
                unit="frame", leave=False)

    for i in range(len(scores)):
        ret, frame = cap.read()
        if not ret:
            break

        _draw_hud(frame, i, scores[i], fps, events, threshold, alert, max_score)
        writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    print(f"Annotated video saved: {output_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def print_report(report: SafetyReport) -> None:
    """Print a human-readable safety report to stdout."""
    print("\n" + "=" * 64)
    print("  GAZE SAFETY MONITORING REPORT")
    print("=" * 64)
    print(f"  Video       : {report.video_path}")
    print(f"  Duration    : {report.duration_sec:.1f}s  ({report.total_frames} frames @ {report.fps:.1f} fps)")
    print(f"  Attentive   : {report.attentive_pct:.1f}%")
    print(f"  Distractions: {len(report.events)} events "
          f"({report.high_severity_count} high severity)")
    print("-" * 64)

    if not report.events:
        print("  No distraction events detected. Subject maintained gaze.")
    else:
        print(f"  {'#':<4} {'Time':>12}  {'Duration':>8}  {'Severity':<10} {'Peak':>6} {'Avg':>6}")
        print(f"  {'—'*4} {'—'*12}  {'—'*8}  {'—'*10} {'—'*6} {'—'*6}")
        for i, e in enumerate(report.events, 1):
            time_range = f"{e.start_time:.1f}s–{e.end_time:.1f}s"
            print(f"  {i:<4} {time_range:>12}  {e.duration:>7.2f}s  {e.severity:<10} "
                  f"{e.peak_score:>5.1f}  {e.avg_score:>5.1f}")

    print("=" * 64)

    # Safety verdict
    if report.attentive_pct >= 95:
        verdict = "PASS — Subject maintained consistent gaze attention"
    elif report.attentive_pct >= 80:
        verdict = "WARNING — Intermittent gaze deviation detected"
    else:
        verdict = "FAIL — Significant gaze deviation detected"
    print(f"  Verdict: {verdict}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(video_path: str, api_key: str, output_dir: str = None) -> SafetyReport:
    """Run the full gaze safety monitoring pipeline.

    Args:
        video_path: Path to input MP4 video
        api_key: NVIDIA NGC API key
        output_dir: Directory for output files (default: same as input)

    Returns:
        SafetyReport with all analysis results
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(video_path))

    base = os.path.splitext(os.path.basename(video_path))[0]
    redirected_path = os.path.join(output_dir, f"{base}_redirected.mp4")
    annotated_path = os.path.join(output_dir, f"{base}_annotated.mp4")
    report_path = os.path.join(output_dir, f"{base}_safety_report.json")

    # Step 1: Call NVIDIA Eye Contact API
    print("\n[1/4] Sending video to NVIDIA Eye Contact API ...")
    call_eye_contact_api(video_path, redirected_path, api_key)

    # Step 2: Compare frames
    print("\n[2/4] Comparing original vs redirected frames ...")
    scores, fps, total_frames = extract_frame_scores(video_path, redirected_path)
    duration = total_frames / fps if fps > 0 else 0

    # Step 3: Detect distraction events
    print("\n[3/4] Analysing gaze deviations ...")
    events = detect_events(scores, fps)

    # Step 4: Generate annotated video
    print("\n[4/4] Generating annotated video ...")
    generate_annotated_video(
        video_path, annotated_path, scores, events, fps,
    )

    report = SafetyReport(
        video_path=video_path,
        duration_sec=duration,
        total_frames=total_frames,
        fps=fps,
        events=events,
        frame_scores=scores,
    )

    # Save JSON report
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Report saved to {report_path}")

    # Print human-readable report
    print_report(report)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gaze Safety Monitor — detect distraction events using "
                    "NVIDIA Eye Contact"
    )
    parser.add_argument("video", help="Path to input MP4 video file")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NGC_API_KEY"),
        help="NVIDIA NGC API key (or set NGC_API_KEY env var)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (default: same as input video)",
    )
    parser.add_argument(
        "--distraction-threshold",
        type=float,
        default=DISTRACTION_THRESHOLD,
        help=f"Pixel difference threshold to flag distraction (default: {DISTRACTION_THRESHOLD})",
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=ALERT_THRESHOLD,
        help=f"Pixel difference threshold for HIGH severity (default: {ALERT_THRESHOLD})",
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: NGC API key required. Set NGC_API_KEY env var or use --api-key")
        sys.exit(1)

    DISTRACTION_THRESHOLD = args.distraction_threshold
    ALERT_THRESHOLD = args.alert_threshold

    run(args.video, args.api_key, args.output_dir)
