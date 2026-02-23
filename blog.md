# NVIDIA Eye Contact & Gaze Detection

## Where Conversation Design Meets Operator Safety

I have been thinking about gaze for a while..

Not in the abstract — in the context of how we build conversational interfaces and how we keep people safe around heavy machinery.

These sound like two entirely different problems.

They are not.

Both come down to the same question: **is this person paying attention?**

In a conversation, gaze tells you whether the other person is engaged.

In a Caterpillar haul truck, gaze tells you whether the operator is paying attention.

Same signal.



NVIDIA's [Maxine Eye Contact](https://build.nvidia.com/nvidia/eyecontact) technology reads that signal.

And I built a working prototype that repurposes it for safety monitoring.

---

### Gaze Is The Original Interface

Before touchscreens. Before keyboards. Before language itself — there was gaze.

Research published in PNAS found that [eye contact marks the rise and fall of shared attention in conversation](https://www.pnas.org/doi/10.1073/pnas.2106645118).

When eyes meet, both people receive a cue that shared engagement is intact.

The conversation flows.

When gaze breaks, the signal is equally clear — attention has moved elsewhere.

This is not subtle.

It is foundational.

And most of our technology completely ignores it.

A [scoping review in the Journal of Nonverbal Behaviour](https://link.springer.com/article/10.1007/s10919-020-00333-3) confirms what anyone who has had a conversation already knows — gaze carries the majority of social information in human interaction. More than tone. More than gesture. Where someone is looking tells you more about their state of mind than what they are saying.

Also, research on [turn-taking in conversations](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.616471/full) shows how specific this is. Speakers avert gaze when they begin talking and during hesitation, but look directly at the listener when they are about to finish an utterance. This is how we regulate who speaks next — not through words, but through where we point our eyes.

Every voice interface we have built so far throws this away.

---

### Face Speed

Fjord, the design and innovation arm of Accenture, introduced a concept in their [voice UI work](https://voiceui.fjordnet.com/) that I keep coming back to.

They called it **face speed**.

The idea is that human conversation operates at the speed of the face. Not the speed of text rendering. Not the speed of a loading spinner. The speed at which facial micro-expressions, gaze direction, blinks and head movements carry information between two people.

Think about what gaze does in a conversation:

- **Turn-taking** — I look at you when I am about to finish speaking. You know it is your turn.
- **Attention signalling** — sustained eye contact says "I am here, I am listening."
- **Trust calibration** — direct gaze strengthens perceived trustworthiness. Averted gaze weakens it.
- **Engagement feedback** — your gaze tells me whether I still have the room, or whether I have lost you.

A smart speaker has no face.

A voice assistant in a car has no eyes.

The entire non-verbal channel — the channel that carries the richest social signal in human interaction — goes dark the moment we move from face-to-face conversation to human-machine interaction.

**This is the gap.**

Not just for conversation design, but for safety. A machine that cannot see where the human is looking cannot know whether the human is paying attention. And in some contexts, that is a life-or-death problem.

---

### Two Sides Of The Same Technology

NVIDIA's [Maxine Eye Contact](https://build.nvidia.com/nvidia/eyecontact) does something precise.

It estimates gaze angles from video and optionally redirects them. The model extracts a 256x64 pixel eye patch from each frame using facial landmark tracking and head pose estimation, then computes where the person is looking as a pitch/yaw angle vector.

What I find interesting is that this single capability serves two fundamentally different purposes..

**Enhancing conversation.**

In video calls, the camera sits above the screen. When you look at the person you are talking to, you look at the screen — not the camera. Your gaze appears averted. Eye Contact redirects the eyes to simulate direct gaze, restoring the non-verbal channel that video conferencing breaks.

This is not vanity. It is conversational infrastructure. It brings face speed back to remote interaction.

**Monitoring attention for safety.**

The same gaze estimation that tells a video call "this person is looking at their screen" can tell a safety system "this person is not looking at the road" or "this operator's eyes are closed."

The technology is identical. The application is life-critical.

---

### The Safety Case: Heavy Machinery

The safety application is not hypothetical. It is deployed. At scale. In some of the most dangerous operating environments on earth.

Caterpillar's [MineStar Detect Driver Safety System (DSS)](https://www.cat.com/en_US/by-industry/mining/surface-mining/surface-technology/detect1/fatigue.html) does exactly this. An in-cab camera monitors the operator's eye closure duration and head pose. If the system detects fatigue or distraction, the operator is immediately alerted through seat vibration and an audio alarm.

The current DSS 5.0 includes upgraded algorithms for head position tracking, eye closure tracking, and facial recognition, backed by an improved camera with enhanced granularity — particularly for eye tracking. Data feeds to a 24/7 monitoring centre where safety advisors analyse video to notify onsite personnel about drowsy or distracted operators.

Consider the context. A Caterpillar 797F haul truck weighs 623 tonnes loaded. A momentary lapse in attention at the wrong moment is catastrophic. These are environments where gaze detection is not a feature — it is a safety system on par with seat belts and roll cages.

And it extends well beyond mining. Construction sites. Warehouses with forklifts. Crane operations. Industrial manufacturing. Anywhere a human operates heavy equipment while fatigued, distracted, or in hazardous conditions. The [Guardian 2 system](https://www.cat.com/en_US/news/machine-press-releases/caterpillar-to-deliver-and-support-guardian-fatigue-and-distraction-monitoring-system-for-light-and-on-highway-vehicle-operators.html) extends the same monitoring to light vehicles and highway operators, fitting any vehicle in the fleet with gaze-based distraction detection.

What Caterpillar understood early is what I think the rest of the industry is only now catching up to: **gaze is the most direct measure of human attention that a camera can capture.** If you have a camera pointed at a person's face and you are not reading their gaze, you are leaving the most valuable signal on the table.

---

### How The Demo Works

I built a working prototype that repurposes NVIDIA's Eye Contact API for safety monitoring. The approach is indirect but effective — and it reveals something about how the technology works under the hood.

The Eye Contact NIM is designed to redirect gaze — it takes a video where someone is looking away from the camera and returns a video where their eyes have been corrected to look forward. The API does not expose the raw gaze angle estimates. What it does expose is the corrected video.

So, compare the original against the corrected output. If the frames look the same, the person was already looking at the camera. If the frames differ, the API had to redirect their gaze — which means they were looking away. The magnitude of the difference maps directly to how far they were looking away.


![Architecture](https://github.com/cobusgreyling/gaze-safety-monitor/blob/main/images/2026-02-23_20-47-50.png) 

NVIDIA only does the AI-heavy part — face tracking, gaze estimation, redirection. Everything else runs locally on your laptop: the frame comparison, event detection, annotated video generation, and the Gradio web UI.

The gaze estimation is happening inside the model — I am reading its work by measuring how much correction it applied. If it barely changed anything, the person was attentive. If it had to move the eyes significantly, they were not.

---

### The Annotated Output

The prototype generates an annotated video with real-time overlays:

- **Coloured border** — green when attentive, yellow when gaze is diverted, red for high-severity distraction
- **Status banner** — displays "ATTENTIVE", "GAZE DIVERTED", or "GAZE DIVERTED — HIGH" with a timestamp
- **Score bar** — a visual meter showing the frame-by-frame deviation score against configurable thresholds
- **Timeline chart** — a full-video plot of gaze deviation over time, with distraction events shaded
- **Safety report** — attentiveness percentage, event count, severity breakdown, and a PASS / WARNING / FAIL verdict

I wrapped the whole pipeline in a Gradio web interface. Upload a video, enter an NGC API key, and the system returns the annotated video alongside the timeline and report. A demo mode lets you test the UI with synthetic scores without needing the API.

---

### The Code

The implementation is roughly 300 lines of Python across two files. The core pipeline:

```python
# gaze_safety_monitor.py — the detection engine

# 1. Send video to NVIDIA NIM via gRPC
call_eye_contact_api(video_path, redirected_path, api_key)

# 2. Compare frames between original and redirected
scores, fps, total_frames = extract_frame_scores(video_path, redirected_path)

# 3. Detect distraction events from score timeseries
events = detect_events(scores, fps)

# 4. Generate annotated video with safety overlays
generate_annotated_video(video_path, annotated_path, scores, events, fps)
```

The comparison focuses on the upper third of each frame — the head and eye region — to isolate gaze changes from compression artefacts elsewhere in the frame. Events must exceed both a pixel-difference threshold and a minimum duration to filter out noise from blinks or momentary glances.

Running it:

```bash
export NGC_API_KEY="nvapi-..."

# Full pipeline
python gaze_safety_monitor.py video.mp4

# Web interface
python gaze_safety_app.py
# Opens at http://localhost:7860
```

---

### Two Worlds, One Signal

What I keep coming back to is how the same signal means such different things depending on where you read it..

| Context | What gaze deviation means | Consequence |
|---------|--------------------------|-------------|
| Video call | Poor eye contact, reduced rapport | Weaker communication |
| Voice UI interaction | User distracted, not engaged | Lower task completion |
| Vehicle operation | Driver not watching the road | Collision risk |
| Heavy equipment | Operator fatigued or distracted | Life-threatening |
| Industrial manufacturing | Worker not monitoring process | Equipment damage, injury |

The input is identical in every case: a camera pointed at a person's face. The output is identical: an estimate of where they are looking. The stakes are the variable.

Fjord's insight about face speed applies across all of these. Human attention operates at the speed of the face. Systems that are aware of gaze — that can read the face — adapt to the human, rather than forcing the human to adapt to the system.

For a voice assistant, this means knowing when the user is actually engaged in the conversation versus looking at something else. For a safety system in a Caterpillar haul truck, it means the difference between an alert that saves a life and a catastrophic failure.

Same technology. Same signal. Fundamentally different consequences.

---

### What This Means

Gaze detection is moving from a niche computer vision problem to infrastructure. NVIDIA ships it as a [NIM microservice](https://build.nvidia.com/nvidia/eyecontact) — a containerised API call. Caterpillar deploys it in operator cabs across mining fleets worldwide. Video conferencing platforms are embedding it as a standard feature.

The common thread: **gaze is not optional in human-machine interaction.** It is the primary non-verbal channel. Ignoring it is like building a phone that cannot detect whether you are holding it to your ear.

I think the conversation design community and the industrial safety community have more to learn from each other than either realises. Fjord understood that voice interfaces need to operate at face speed. Caterpillar understood that operator monitoring needs to read the face in real time. They arrived at the same conclusion from opposite ends of the spectrum — and the technology serving both is now a single API call.

For anyone building systems that humans interact with while their eyes are open, gaze is the signal you should be reading.

---

Chief Evangelist @ Kore.ai | I'm passionate about exploring the intersection of AI and language. Language Models, AI Agents, Agentic Apps, Dev Frameworks & Data-Driven Tools shaping tomorrow.

- [NVIDIA Maxine Eye Contact NIM](https://build.nvidia.com/nvidia/eyecontact) — NVIDIA's eye contact and gaze estimation model, available as a cloud API.
- [NVIDIA Maxine Eye Contact Documentation](https://docs.nvidia.com/nim/maxine/eye-contact/latest/overview.html) — Technical documentation for the Eye Contact NIM.
- [NVIDIA Maxine NIM Client Examples (GitHub)](https://github.com/NVIDIA-Maxine/nim-clients) — Sample Python clients for calling the Eye Contact API.
- [NVIDIA Technical Blog — Maxine Eye Contact](https://developer.nvidia.com/blog/improve-human-connection-in-video-conferences-with-nvidia-maxine-eye-contact/) — Technical deep-dive into the encoder-decoder architecture.
- [Cat MineStar Detect — Fatigue and Distraction Monitoring](https://www.cat.com/en_US/by-industry/mining/surface-mining/surface-technology/detect1/fatigue.html) — Caterpillar's operator monitoring system using eye tracking in heavy equipment.
- [Cat Guardian 2 Announcement](https://www.cat.com/en_US/news/machine-press-releases/caterpillar-to-deliver-and-support-guardian-fatigue-and-distraction-monitoring-system-for-light-and-on-highway-vehicle-operators.html) — Extension of gaze monitoring to light vehicles and highway operators.
- [Eye Contact Marks the Rise and Fall of Shared Attention (PNAS)](https://www.pnas.org/doi/10.1073/pnas.2106645118) — Research on gaze and shared attention in conversation.
- [The Measurement of Eye Contact in Human Interactions (Springer)](https://link.springer.com/article/10.1007/s10919-020-00333-3) — Scoping review of eye contact measurement methods.
- [The Role of Eye Gaze in Regulating Turn Taking (Frontiers)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.616471/full) — Research on gaze and conversational turn-taking.
