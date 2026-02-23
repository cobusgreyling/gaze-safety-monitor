"""
Gaze Safety Monitor — Gradio Web UI

A web interface for the NVIDIA Eye Contact gaze safety monitoring pipeline.
Upload a video, process it through the NVIDIA Maxine Eye Contact NIM, and
see an annotated video with real-time gaze deviation scores plus a timeline
chart and safety report.

Usage:
    python gaze_safety_app.py
    # Opens at http://localhost:7860
"""

import os
import sys
import pathlib
import tempfile

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
import gaze_safety_monitor as gsm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SAMPLE_VIDEO = str(
    pathlib.Path(__file__).parent
    / "nim-clients-eye-contact" / "assets" / "sample_transactional.mp4"
)
TEMP_DIR = tempfile.mkdtemp(prefix="gaze_safety_")


# ---------------------------------------------------------------------------
# Score timeline chart
# ---------------------------------------------------------------------------
def create_timeline_chart(
    scores: list[float],
    events: list,
    fps: float,
    threshold: float,
    alert_threshold: float,
) -> str:
    """Generate a score timeline chart and save as PNG."""
    if not scores:
        return None

    times = [i / fps for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Shade distraction events
    for event in events:
        color = "#ff4444" if event.severity == "HIGH" else (
            "#ffaa00" if event.severity == "MODERATE" else "#ffdd57"
        )
        ax.axvspan(event.start_time, event.end_time, alpha=0.25, color=color)

    # Score line
    ax.fill_between(times, scores, alpha=0.3, color="#00d4ff")
    ax.plot(times, scores, color="#00d4ff", linewidth=1.2, label="Gaze deviation score")

    # Threshold lines
    ax.axhline(y=threshold, color="#ffaa00", linestyle="--", linewidth=1,
               label=f"Distraction threshold ({threshold})")
    ax.axhline(y=alert_threshold, color="#ff4444", linestyle="--", linewidth=1,
               label=f"Alert threshold ({alert_threshold})")

    ax.set_xlabel("Time (seconds)", color="white", fontsize=10)
    ax.set_ylabel("Score", color="white", fontsize=10)
    ax.set_title("Gaze Deviation Timeline", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(loc="upper right", fontsize=8,
                       facecolor="#2a2a4a", edgecolor="#444")
    for text in legend.get_texts():
        text.set_color("white")

    ax.set_xlim(0, times[-1] if times else 1)
    ax.set_ylim(0, max(max(scores) * 1.2, alert_threshold * 1.5))

    plt.tight_layout()
    chart_path = os.path.join(TEMP_DIR, "timeline_chart.png")
    fig.savefig(chart_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return chart_path


# ---------------------------------------------------------------------------
# Build report HTML
# ---------------------------------------------------------------------------
def build_report_html(report: gsm.SafetyReport) -> str:
    """Generate styled HTML for the safety report."""
    if report.attentive_pct >= 95:
        verdict = "PASS"
        verdict_color = "#00c853"
        verdict_text = "Subject maintained consistent gaze attention"
    elif report.attentive_pct >= 80:
        verdict = "WARNING"
        verdict_color = "#ffaa00"
        verdict_text = "Intermittent gaze deviation detected"
    else:
        verdict = "FAIL"
        verdict_color = "#ff4444"
        verdict_text = "Significant gaze deviation detected"

    # Attentiveness gauge colour
    pct = report.attentive_pct
    if pct >= 90:
        gauge_color = "#00c853"
    elif pct >= 70:
        gauge_color = "#ffaa00"
    else:
        gauge_color = "#ff4444"

    events_html = ""
    if report.events:
        rows = ""
        for i, e in enumerate(report.events, 1):
            sev_color = {"HIGH": "#ff4444", "MODERATE": "#ffaa00", "LOW": "#ffdd57"}[e.severity]
            rows += f"""
            <tr>
                <td style="text-align:center">{i}</td>
                <td>{e.start_time:.1f}s &ndash; {e.end_time:.1f}s</td>
                <td>{e.duration:.2f}s</td>
                <td><span style="color:{sev_color};font-weight:bold">{e.severity}</span></td>
                <td>{e.peak_score:.1f}</td>
                <td>{e.avg_score:.1f}</td>
            </tr>"""
        events_html = f"""
        <table style="width:100%;border-collapse:collapse;margin-top:12px;font-size:14px">
            <thead>
                <tr style="border-bottom:1px solid #555;color:#aaa">
                    <th style="padding:6px">#</th>
                    <th style="padding:6px;text-align:left">Time</th>
                    <th style="padding:6px;text-align:left">Duration</th>
                    <th style="padding:6px;text-align:left">Severity</th>
                    <th style="padding:6px;text-align:left">Peak</th>
                    <th style="padding:6px;text-align:left">Avg</th>
                </tr>
            </thead>
            <tbody style="color:#ddd">{rows}</tbody>
        </table>"""
    else:
        events_html = '<p style="color:#aaa;margin-top:12px">No distraction events detected.</p>'

    html = f"""
    <div style="font-family:system-ui;padding:16px;background:#1a1a2e;border-radius:12px;color:white">
        <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:16px">
            <div style="flex:1;min-width:180px;background:#2a2a4a;padding:16px;border-radius:8px;text-align:center">
                <div style="font-size:13px;color:#aaa;margin-bottom:4px">Attentiveness</div>
                <div style="font-size:36px;font-weight:bold;color:{gauge_color}">{pct:.1f}%</div>
            </div>
            <div style="flex:1;min-width:180px;background:#2a2a4a;padding:16px;border-radius:8px;text-align:center">
                <div style="font-size:13px;color:#aaa;margin-bottom:4px">Distraction Events</div>
                <div style="font-size:36px;font-weight:bold;color:#00d4ff">{len(report.events)}</div>
            </div>
            <div style="flex:1;min-width:180px;background:#2a2a4a;padding:16px;border-radius:8px;text-align:center">
                <div style="font-size:13px;color:#aaa;margin-bottom:4px">High Severity</div>
                <div style="font-size:36px;font-weight:bold;color:#ff4444">{report.high_severity_count}</div>
            </div>
            <div style="flex:1;min-width:180px;background:#2a2a4a;padding:16px;border-radius:8px;text-align:center">
                <div style="font-size:13px;color:#aaa;margin-bottom:4px">Verdict</div>
                <div style="font-size:24px;font-weight:bold;color:{verdict_color}">{verdict}</div>
                <div style="font-size:11px;color:#aaa;margin-top:2px">{verdict_text}</div>
            </div>
        </div>

        <div style="background:#2a2a4a;padding:12px 16px;border-radius:8px;margin-bottom:12px">
            <div style="font-size:12px;color:#aaa">
                Duration: {report.duration_sec:.1f}s &nbsp;|&nbsp;
                Frames: {report.total_frames} &nbsp;|&nbsp;
                FPS: {report.fps:.1f}
            </div>
        </div>

        <div style="background:#2a2a4a;padding:12px 16px;border-radius:8px">
            <div style="font-size:14px;font-weight:bold;margin-bottom:4px">Distraction Events</div>
            {events_html}
        </div>
    </div>"""
    return html


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------
def process_video_full(video_path: str, api_key: str, dist_thresh: float, alert_thresh: float):
    """Full pipeline: API call + analysis + annotated video."""
    if not video_path:
        raise gr.Error("Please upload a video file.")
    if not api_key:
        raise gr.Error("Please enter your NGC API key.")

    base = os.path.splitext(os.path.basename(video_path))[0]
    redirected_path = os.path.join(TEMP_DIR, f"{base}_redirected.mp4")
    annotated_path = os.path.join(TEMP_DIR, f"{base}_annotated.mp4")

    # Step 1: Call NVIDIA API
    gr.Info("Sending video to NVIDIA Eye Contact API ...")
    gsm.call_eye_contact_api(video_path, redirected_path, api_key)

    # Step 2: Compare frames
    gr.Info("Comparing original vs redirected frames ...")
    scores, fps, total_frames = gsm.extract_frame_scores(video_path, redirected_path)
    duration = total_frames / fps if fps > 0 else 0

    # Step 3: Detect events
    events = gsm.detect_events(scores, fps, threshold=dist_thresh, alert_threshold=alert_thresh)

    # Step 4: Generate annotated video
    gr.Info("Generating annotated video ...")
    gsm.generate_annotated_video(
        video_path, annotated_path, scores, events, fps,
        threshold=dist_thresh, alert=alert_thresh,
    )

    # Build report
    report = gsm.SafetyReport(
        video_path=video_path,
        duration_sec=duration,
        total_frames=total_frames,
        fps=fps,
        events=events,
        frame_scores=scores,
    )

    # Timeline chart
    chart_path = create_timeline_chart(scores, events, fps, dist_thresh, alert_thresh)

    report_html = build_report_html(report)

    gr.Info("Processing complete!")
    return annotated_path, chart_path, report_html


def process_demo(video_path: str, dist_thresh: float, alert_thresh: float):
    """Demo mode: skip API call, generate synthetic scores for any video."""
    if not video_path:
        raise gr.Error("Please upload a video file.")

    import cv2 as _cv2
    cap = _cv2.VideoCapture(video_path)
    fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total == 0:
        raise gr.Error("Could not read video frames.")

    # Generate realistic synthetic scores
    np.random.seed(42)
    base_noise = np.random.normal(1.5, 0.8, total).clip(0)

    # Inject distraction events at ~25%, ~50%, ~75% through video
    for center_pct in [0.25, 0.50, 0.75]:
        center = int(total * center_pct)
        width = int(fps * np.random.uniform(0.8, 2.5))
        peak = np.random.uniform(8, 20)
        start = max(0, center - width // 2)
        end = min(total, center + width // 2)
        for i in range(start, end):
            dist = abs(i - center) / (width / 2)
            base_noise[i] += peak * (1 - dist ** 2)

    scores = base_noise.tolist()
    events = gsm.detect_events(scores, fps, threshold=dist_thresh, alert_threshold=alert_thresh)

    base = os.path.splitext(os.path.basename(video_path))[0]
    annotated_path = os.path.join(TEMP_DIR, f"{base}_demo_annotated.mp4")

    gr.Info("Generating annotated video (demo mode) ...")
    gsm.generate_annotated_video(
        video_path, annotated_path, scores, events, fps,
        threshold=dist_thresh, alert=alert_thresh,
    )

    duration = total / fps
    report = gsm.SafetyReport(
        video_path=video_path,
        duration_sec=duration,
        total_frames=total,
        fps=fps,
        events=events,
        frame_scores=scores,
    )

    chart_path = create_timeline_chart(scores, events, fps, dist_thresh, alert_thresh)
    report_html = build_report_html(report)

    gr.Info("Demo processing complete!")
    return annotated_path, chart_path, report_html


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Gaze Safety Monitor") as app:
        gr.Markdown(
            "# Gaze Safety Monitor",
            elem_classes="main-title",
        )
        gr.Markdown(
            "Detect distraction events using NVIDIA Maxine Eye Contact NIM",
            elem_classes="subtitle",
        )

        with gr.Row():
            # --- Left column: inputs ---
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video", sources=["upload"])

                with gr.Accordion("Sample Video", open=False):
                    sample_btn = gr.Button("Load sample video", variant="secondary", size="sm")

                api_key = gr.Textbox(
                    label="NGC API Key",
                    placeholder="nvapi-...",
                    type="password",
                    info="Required for full pipeline. Leave empty for demo mode.",
                )

                with gr.Row():
                    dist_thresh = gr.Slider(
                        minimum=1.0, maximum=20.0, value=5.0, step=0.5,
                        label="Distraction Threshold",
                    )
                    alert_thresh = gr.Slider(
                        minimum=5.0, maximum=30.0, value=12.0, step=0.5,
                        label="Alert Threshold",
                    )

                with gr.Row():
                    process_btn = gr.Button("Process (Full API)", variant="primary")
                    demo_btn = gr.Button("Process (Demo Mode)", variant="secondary")

                gr.Markdown(
                    "*Demo mode* uses synthetic gaze scores to demonstrate "
                    "the annotation and reporting without calling the NVIDIA API.",
                    elem_classes="subtitle",
                )

            # --- Right column: outputs ---
            with gr.Column(scale=2):
                video_output = gr.Video(label="Annotated Video")
                chart_output = gr.Image(label="Gaze Deviation Timeline", type="filepath")
                report_output = gr.HTML(label="Safety Report")

        # --- Event handlers ---
        def load_sample():
            if os.path.isfile(SAMPLE_VIDEO):
                return SAMPLE_VIDEO
            return gr.update()

        sample_btn.click(fn=load_sample, outputs=video_input)

        process_btn.click(
            fn=process_video_full,
            inputs=[video_input, api_key, dist_thresh, alert_thresh],
            outputs=[video_output, chart_output, report_output],
        )

        demo_btn.click(
            fn=process_demo,
            inputs=[video_input, dist_thresh, alert_thresh],
            outputs=[video_output, chart_output, report_output],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate"),
        css="""
        .main-title { text-align: center; margin-bottom: 4px; }
        .subtitle { text-align: center; color: #888; font-size: 14px; margin-bottom: 16px; }
        """,
    )
